use crate::registry::ModelRegistry;
use anyhow::{Context, Result};
use rusty_genius_core::manifest::ModelSpec;
use rusty_genius_core::GeniusError;
use std::fs;
use std::path::PathBuf;

pub struct AssetAuthority {
    registry: ModelRegistry,
}

impl AssetAuthority {
    pub fn new() -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::new()?,
        })
    }

    pub async fn ensure_model(&self, name: &str) -> Result<PathBuf> {
        // 1. Resolve model spec
        let spec = self.registry.resolve(name).ok_or_else(|| {
            GeniusError::ManifestError(format!("Model '{}' not found in registry", name))
        })?;

        // 2. Check cache
        let cache_dir = self.registry.get_cache_dir();
        fs::create_dir_all(&cache_dir)?;

        let path = cache_dir.join(&spec.filename);
        if path.exists() {
            return Ok(path);
        }

        // 3. Download (Simulated or Real)
        println!("Downloading {} from {}...", spec.filename, spec.repo);

        // Use spec.filename or name.gguf
        let local_filename = format!("{}.gguf", name);
        let path = cache_dir.join(&local_filename);

        if path.exists() {
            return Ok(path);
        }

        self.download_file(&spec, &path).await?;

        Ok(path)
    }

    // Async download using surf with RedirectMiddleware
    async fn download_file(&self, spec: &ModelSpec, path: &PathBuf) -> Result<()> {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            spec.repo, spec.filename
        );

        let client = surf::Client::new().with(RedirectMiddleware::new(5));

        let mut response = client
            .get(&url)
            .await
            .map_err(|e| anyhow::anyhow!("Surf request failed: {}", e))?;

        let status = response.status();
        if status.is_success() {
            println!("Downloading to: {:?}", path);

            // Use std::fs to create file (workaround for potential async fs quirks in some envs)
            let std_file = std::fs::File::create(path)
                .map_err(|e| anyhow::anyhow!("Failed to create file (std): {}", e))?;

            // Convert to async-std file
            let mut file: async_std::fs::File = std_file.into();

            // Copy from response (which implements AsyncRead) to file
            // response implements AsyncRead in surf 2.3+
            futures::io::copy(&mut response, &mut file)
                .await
                .map_err(|e| anyhow::anyhow!("Streaming failed: {}", e))?;

            println!("Download complete.");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Download failed with status: {}", status))
        }
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
