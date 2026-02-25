use crate::registry::ModelEntry;
use crate::registry::ModelRegistry;
use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::manifest::ModelSpec;
use rusty_genius_core::GeniusError;
use rusty_genius_thinkerv1 as thinkerv1;
use std::fs;
use std::path::PathBuf;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

struct ProgressReader<R> {
    inner: R,
    current: u64,
    total: u64,
    sender: mpsc::Sender<thinkerv1::Response>,
    id: String,
}

impl<R: futures::io::AsyncRead + Unpin> futures::io::AsyncRead for ProgressReader<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        match std::pin::Pin::new(&mut self.inner).poll_read(cx, buf) {
            std::task::Poll::Ready(Ok(n)) => {
                if n > 0 {
                    self.current += n as u64;
                    let progress = if self.total > 0 {
                        Some(self.current as f32 / self.total as f32)
                    } else {
                        None
                    };
                    let id_clone = self.id.clone();
                    let _ = self.sender.try_send(thinkerv1::Response::Status(
                        thinkerv1::StatusResponse {
                            id: id_clone,
                            status: "downloading".to_string(),
                            progress,
                            message: None,
                        },
                    ));
                }
                std::task::Poll::Ready(Ok(n))
            }
            other => other,
        }
    }
}

impl AssetAuthority {
    pub fn new() -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::new()?,
        })
    }

    pub fn list_models(&self) -> Vec<ModelEntry> {
        self.registry.list_models()
    }

    pub async fn ensure_model(
        &self,
        id: String,
        name: &str,
        model_config: Option<thinkerv1::ModelConfig>,
    ) -> Result<PathBuf> {
        let mut stream = self.ensure_model_stream(id, name, model_config);
        let mut final_path = None;
        while let Some(response) = stream.next().await {
            if let thinkerv1::Response::Status(status) = response {
                if status.status == "ready" {
                    final_path = status.message;
                    break;
                }
                if status.status == "error" {
                    return Err(anyhow::anyhow!(
                        "Failed to ensure model: {}",
                        status.message.unwrap_or_default()
                    ));
                }
            }
        }
        final_path
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("Model ensure stream ended without a ready signal."))
    }

    pub fn ensure_model_stream(
        &self,
        id: String,
        name: &str,
        model_config: Option<thinkerv1::ModelConfig>,
    ) -> mpsc::Receiver<thinkerv1::Response> {
        let (tx, rx) = mpsc::channel(100);
        let name = name.to_string();

        async_std::task::spawn(async move {
            let mut err_tx = tx.clone();
            let result: Result<()> = async {
                let auth = AssetAuthority::new()?;
                auth.ensure_model_internal(id.clone(), &name, model_config, tx, false).await?;
                Ok(())
            }
            .await;

            if let Err(e) = result {
                let _ = err_tx
                    .send(thinkerv1::Response::Status(thinkerv1::StatusResponse {
                        id,
                        status: "error".to_string(),
                        progress: None,
                        message: Some(e.to_string()),
                    }))
                    .await;
            }
        });

        rx
    }

    async fn ensure_model_internal(
        &self,
        id: String,
        name: &str,
        model_config: Option<thinkerv1::ModelConfig>,
        mut tx: mpsc::Sender<thinkerv1::Response>,
        _silent: bool,
    ) -> Result<PathBuf> {
        let _ = tx
            .send(thinkerv1::Response::Status(thinkerv1::StatusResponse {
                id: id.clone(),
                status: "resolving".to_string(),
                progress: None,
                message: Some(name.to_string()),
            }))
            .await;

        let mut spec = if let Some(s) = self.registry.resolve(name) {
            s
        } else {
            // If not in registry and not a HuggingFace ID with '/', assume a default repo
            // and treat the name as the filename and a default quantization.
            ModelSpec {
                repo: "ggml-org/models".to_string(), // Default HuggingFace Org/Repo
                filename: format!("{}.gguf", name),
                quantization: "Q4_K_M".to_string(),
            }
        };

        if let Some(config) = &model_config {
            if let Some(quant) = &config.quant {
                spec.quantization = quant.clone();
            }
        }

        let cache_dir = self.registry.get_cache_dir();
        fs::create_dir_all(&cache_dir)?;

        let path = cache_dir.join(&spec.filename);
        if path.exists() {
            let _ = tx
                .send(thinkerv1::Response::Status(thinkerv1::StatusResponse {
                    id,
                    status: "ready".to_string(),
                    progress: None,
                    message: Some(path.display().to_string()),
                }))
                .await;
            return Ok(path);
        }

        self.download_file_with_events(id.clone(), &spec, &path, tx.clone())
            .await?;
        
        let _ = tx
            .send(thinkerv1::Response::Status(thinkerv1::StatusResponse {
                id,
                status: "ready".to_string(),
                progress: None,
                message: Some(path.display().to_string()),
            }))
            .await;

        Ok(path)
    }

    async fn download_file_with_events(
        &self,
        id: String,
        spec: &ModelSpec,
        final_path: &PathBuf,
        sender: mpsc::Sender<thinkerv1::Response>,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );
        let _ = sender.clone().try_send(thinkerv1::Response::Status(
            thinkerv1::StatusResponse {
                id: id.clone(),
                status: "downloading".to_string(),
                progress: Some(0.0),
                message: Some(url.clone()),
            },
        ));

        let partial_path = final_path.with_extension("partial");
        
        let mut current_url = url; // Make url mutable for redirects
        let mut redirect_count = 0;
        const MAX_REDIRECTS: u8 = 5; // A reasonable limit for redirects

        let response = loop {
            let client = surf::Client::new();
            let res = client.get(&current_url).await.map_err(|e| anyhow::anyhow!("Surf request failed: {}", e))?;
            
            let status = res.status();
            if status.is_success() {
                break res; // Successful, exit loop with response
            } else if status.is_redirection() {
                redirect_count += 1;
                if redirect_count > MAX_REDIRECTS {
                    return Err(anyhow::anyhow!("Too many redirects ({}) for URL: {}", MAX_REDIRECTS, current_url));
                }

                let location_header = res.header("Location")
                                            .map(|h| h.last().as_str())
                                            .ok_or_else(|| anyhow::anyhow!("Redirect without Location header for URL: {}", current_url))?;
                
                // Update current_url for the next iteration
                current_url = location_header.to_string();
                eprintln!("DEBUG: Redirecting to: {} (attempt {}/{})", current_url, redirect_count, MAX_REDIRECTS);
                // Continue loop
            } else {
                // Other non-successful status
                let location_header = res.header("Location").map(|h| h.last().as_str()).unwrap_or("N/A");
                eprintln!("DEBUG: Download failed for URL: {} with status: {}. Location header: {}", current_url, status, location_header);
                return Err(anyhow::anyhow!("Download failed with status: {}", status));
            }
        };

        let total_size = response
            .header("Content-Length")
            .and_then(|h| h.last().as_str().parse::<u64>().ok())
            .unwrap_or(0);

        let mut reader = ProgressReader {
            inner: response,
            current: 0,
            total: total_size,
            sender,
            id,
        };

        let mut file = async_std::fs::File::create(&partial_path).await?;
        futures::io::copy(&mut reader, &mut file).await?;
        
        std::fs::rename(&partial_path, final_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[async_std::test]
    async fn test_ensure_model_tiny() {
        let authority = AssetAuthority::new().unwrap();
        let name = "tiny-model";
        let res = authority.ensure_model("test-id-1".to_string(), name, None).await;
        assert!(res.is_ok(), "Should resolve and download (or find) tiny-model");
        let path = res.unwrap();
        assert!(path.exists());
    }

    #[async_std::test]
    async fn test_ensure_model_stream() {
        let authority = AssetAuthority::new().unwrap();
        let name = "tiny-model";

        let mut rx = authority.ensure_model_stream("test-id-2".to_string(), name, None);
        let mut saw_ready = false;

        while let Some(response) = rx.next().await {
            if let thinkerv1::Response::Status(status) = response {
                 if status.status == "ready" {
                    saw_ready = true;
                    assert!(status.message.is_some());
                    assert!(std::path::Path::new(&status.message.clone().unwrap()).exists());
                 }
                 if status.status == "error" {
                    panic!("Download error: {}", status.message.unwrap_or_default());
                 }
            }
        }
        assert!(saw_ready, "Should have received Ready status");
    }
}
