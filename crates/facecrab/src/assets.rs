use crate::registry::ModelRegistry;
use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::manifest::ModelSpec;
use rusty_genius_core::protocol::AssetEvent;
use rusty_genius_core::GeniusError;
use std::fs;
use std::path::PathBuf;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

struct ProgressReader<R> {
    inner: R,
    current: u64,
    total: u64,
    sender: mpsc::Sender<AssetEvent>,
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
                    let current = self.current;
                    let total = self.total;
                    let _ = self.sender.try_send(AssetEvent::Progress(current, total));
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

    /// Download a model and return its local path.
    pub async fn ensure_model(&self, name: &str) -> Result<PathBuf> {
        let (tx, mut rx) = mpsc::channel(1);
        let name = name.to_string();

        let handle = async_std::task::spawn(async move {
            if let Ok(auth) = AssetAuthority::new() {
                auth.ensure_model_internal(&name, tx, true).await
            } else {
                Err(anyhow::anyhow!("Failed to create authority"))
            }
        });

        while let Some(_) = rx.next().await {}
        handle.await
    }

    /// Download a model and return a stream of [AssetEvent]s.
    pub fn ensure_model_stream(&self, name: &str) -> mpsc::Receiver<AssetEvent> {
        let (tx, rx) = mpsc::channel(100);
        let name = name.to_string();

        async_std::task::spawn(async move {
            if let Ok(auth) = AssetAuthority::new() {
                let _ = auth.ensure_model_internal(&name, tx, false).await;
            }
        });

        rx
    }

    async fn ensure_model_internal(
        &self,
        name: &str,
        mut tx: mpsc::Sender<AssetEvent>,
        silent: bool,
    ) -> Result<PathBuf> {
        let _ = tx.send(AssetEvent::Started(name.to_string())).await;

        let spec = self.registry.resolve(name).ok_or_else(|| {
            let err = format!("Model '{}' not found in registry", name);
            let _ = tx.try_send(AssetEvent::Error(err.clone()));
            GeniusError::ManifestError(err)
        })?;

        let cache_dir = self.registry.get_cache_dir();
        fs::create_dir_all(&cache_dir)?;

        let path = cache_dir.join(&spec.filename);
        if path.exists() {
            let _ = tx
                .send(AssetEvent::Complete(path.display().to_string()))
                .await;
            return Ok(path);
        }

        if !silent {
            println!("Downloading {} from {}...", spec.filename, spec.repo);
        }
        self.download_file_with_events(&spec, &path, tx.clone())
            .await?;

        let _ = tx
            .send(AssetEvent::Complete(path.display().to_string()))
            .await;
        Ok(path)
    }

    async fn download_file_with_events(
        &self,
        spec: &ModelSpec,
        final_path: &PathBuf,
        sender: mpsc::Sender<AssetEvent>,
    ) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );

        let partial_path = final_path.with_extension("partial");
        let client = surf::Client::new().with(RedirectMiddleware::new(5));
        let response = client
            .get(&url)
            .await
            .map_err(|e| anyhow::anyhow!("Surf request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("Download failed with status: {}", status));
        }

        let total_size = response
            .header("Content-Length")
            .and_then(|h| h.last().as_str().parse::<u64>().ok())
            .unwrap_or(0);

        let mut reader = ProgressReader {
            inner: response,
            current: 0,
            total: total_size,
            sender,
        };

        {
            let std_file = std::fs::File::create(&partial_path)
                .map_err(|e| anyhow::anyhow!("Failed to create partial file: {}", e))?;
            let mut file: async_std::fs::File = std_file.into();

            if let Err(e) = futures::io::copy(&mut reader, &mut file).await {
                let _ = std::fs::remove_file(&partial_path);
                return Err(anyhow::anyhow!("Streaming failed: {}", e));
            }
        }

        std::fs::rename(&partial_path, final_path)
            .map_err(|e| anyhow::anyhow!("Failed to finalize model file: {}", e))?;
        Ok(())
    }
}

struct RedirectMiddleware {
    max_attempts: u8,
}

impl RedirectMiddleware {
    pub fn new(max_attempts: u8) -> Self {
        Self { max_attempts }
    }
}

#[surf::utils::async_trait]
impl surf::middleware::Middleware for RedirectMiddleware {
    async fn handle(
        &self,
        req: surf::Request,
        client: surf::Client,
        next: surf::middleware::Next<'_>,
    ) -> surf::Result<surf::Response> {
        let mut attempts = 0;
        let mut current_req = req;

        loop {
            // Check attempts
            if attempts > self.max_attempts {
                return Err(surf::Error::from_str(
                    surf::StatusCode::LoopDetected,
                    "Too many redirects",
                ));
            }

            // Clone req for the attempt (body might be an issue if not reusable, but for GET it's fine)
            // surf::Request cloning is usually cheap (Arc-ish for body?).
            // Wait, Request isn't trivially cloneable if body is a naive stream.
            // But `current_req.clone()` works in surf.
            let req_clone = current_req.clone();

            let response = next.run(req_clone, client.clone()).await?;

            if response.status().is_redirection() {
                if let Some(location) = response.header("Location") {
                    let loc_str = location.last().as_str().to_string();
                    // Update URL
                    // Use Url parsing to handle relative redirects?
                    // For HF, usually absolute.
                    // I will assume absolute or handle simple parse.

                    let new_url = match surf::Url::parse(&loc_str) {
                        Ok(u) => u,
                        Err(_) => {
                            // Try joining with base?
                            let base = current_req.url();
                            match base.join(&loc_str) {
                                Ok(u) => u,
                                Err(_) => {
                                    return Err(surf::Error::from_str(
                                        surf::StatusCode::BadGateway,
                                        "Invalid redirect location",
                                    ))
                                }
                            }
                        }
                    };

                    current_req = surf::Request::new(current_req.method(), new_url);
                    // Copy headers? usually yes.
                    // For now, new request is clean. simple GET.
                    // HF auth headers not needed for public models, but if they were, we'd copy.

                    attempts += 1;
                    continue;
                }
            }

            return Ok(response);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[async_std::test]
    async fn test_ensure_model_tiny() {
        let authority = AssetAuthority::new().unwrap();
        // Use a temp dir for testing if possible, but for now we'll just test the resolve logic
        // and assume connectivity is allowed in this environment.
        let name = "tiny-model";
        let res = authority.ensure_model(name).await;
        assert!(
            res.is_ok(),
            "Should resolve and download (or find) tiny-model"
        );
        let path = res.unwrap();
        assert!(path.exists());
    }

    #[async_std::test]
    async fn test_ensure_model_stream() {
        let authority = AssetAuthority::new().unwrap();
        let name = "tiny-model";

        let mut rx = authority.ensure_model_stream(name);
        let mut saw_started = false;
        let mut saw_complete = false;

        while let Some(event) = rx.next().await {
            match event {
                AssetEvent::Started(_) => saw_started = true,
                AssetEvent::Complete(p) => {
                    saw_complete = true;
                    assert!(
                        std::path::Path::new(&p).exists(),
                        "Complete path must exist"
                    );
                }
                AssetEvent::Error(e) => panic!("Download error: {}", e),
                _ => {}
            }
        }

        assert!(saw_started, "Should have received Started event");
        assert!(saw_complete, "Should have received Complete event");
    }
}
