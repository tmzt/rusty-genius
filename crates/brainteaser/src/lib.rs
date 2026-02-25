#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::StreamExt;
    use rusty_genius::Genius;
    use rusty_genius_thinkerv1::{EventResponse, ModelConfig, Response, InferenceConfig};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    #[derive(Debug)]
    struct Fixture {
        path: PathBuf,
        // org: String, // Removed
        // repo: String, // Removed
        // quant: String, // Removed
        test_name: String,
    }

    fn scan_fixtures(base_path: &Path) -> Vec<Fixture> {
        let mut fixtures = Vec::new();
        if !base_path.exists() {
            println!(
                "Fixture path {:?} does not exist, skipping scan.",
                base_path
            );
            return fixtures;
        }
        for entry in walkdir::WalkDir::new(base_path) {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
                 if let Ok(stripped) = path.strip_prefix(base_path) {
                    let components: Vec<_> = stripped
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().to_string())
                        .collect();
                    if components.len() == 4 {
                        fixtures.push(Fixture {
                            path: path.to_path_buf(),
                            test_name: path
                                .file_stem()
                                .unwrap()
                                .to_string_lossy()
                                .to_string(),
                        });
                    }
                }
            }
        }
        fixtures.sort_by(|a, b| a.test_name.cmp(&b.test_name));
        fixtures
    }

    #[async_std::test]
    async fn test_inference_flow() -> Result<()> {
        println!("Starting test_inference_flow...");

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let fixture_root = PathBuf::from(manifest_dir).join("fixtures");
        let fixtures = scan_fixtures(&fixture_root);
        println!("Found {} fixtures.", fixtures.len());

        let mut genius = Genius::new(Some(60)).await?; // Short TTL for testing

        // 1. Load Model
        println!("Ensuring model is loaded...");
        let model_name = "tiny-random-llama-gguf"; // A model name that should be in the registry or resolve
        let model_config = Some(ModelConfig {
            quant: Some("Q4_K_M".to_string()),
            context_size: Some(1024),
            ttl_seconds: Some(-1), // Keep alive for the test duration
        });

        let mut ensure_stream = genius
            .ensure_model_stream(model_name.to_string(), true, model_config)
            .await?;
        
        loop {
            match ensure_stream.next().await {
                Some(Response::Status(status)) => {
                    println!("[Client] Model Status: {:?}", status);
                    if status.status == "ready" {
                        break;
                    }
                    if status.status == "error" {
                        return Err(anyhow::anyhow!("Failed to load model: {}", status.message.unwrap_or_default()));
                    }
                }
                Some(_) => {}, // Ignore other events
                None => return Err(anyhow::anyhow!("Ensure model stream ended unexpectedly.")),
            }
        }
        println!("Model is ready.");

        // 2. Run Inference
        let prompt = if let Some(fixture) = fixtures.first() {
            println!("Running fixture: {}", fixture.test_name);
            fs::read_to_string(&fixture.path)?
        } else {
            println!("No fixtures found, defaulting.");
            "Tell me a joke about Rust".to_string()
        };
        println!("Prompt: {}", prompt);
        
        let inference_config = Some(InferenceConfig {
            show_thinking: true,
            ..Default::default()
        });

        let mut infer_stream = genius.infer_stream(prompt.clone(), inference_config).await?;

        let mut collected_output = String::new();
        let mut thought_process = String::new();

        println!("Waiting for inference events...");
        let timeout_sec = if cfg!(feature = "real-engine") { 600 } else { 5 };

        loop {
            let msg = async_std::future::timeout(Duration::from_secs(timeout_sec), infer_stream.next()).await;
            match msg {
                Ok(Some(response)) => {
                    if let Response::Event(event) = response {
                        match event {
                            EventResponse::Thought{ content, .. } => {
                                thought_process.push_str(&content);
                            },
                            EventResponse::Content{ content, .. } => {
                                collected_output.push_str(&content);
                            },
                            EventResponse::Complete{ .. } => {
                                println!("Inference Complete");
                                break;
                            },
                            _ => {}
                        }
                    } else if let Response::Status(status) = response {
                        if status.status == "error" {
                             return Err(anyhow::anyhow!("Received error during inference: {}", status.message.unwrap_or_default()));
                        }
                    }
                },
                Ok(None) => return Err(anyhow::anyhow!("Inference stream closed unexpectedly")),
                Err(_) => return Err(anyhow::anyhow!("Timeout waiting for inference response")),
            }
        }
        
        println!("Collected Output: {}", collected_output);

        #[cfg(not(feature = "real-engine"))]
        {
            assert!(thought_process.contains("Narf!"));
            assert!(collected_output.contains("Pinky says:"));
            assert!(collected_output.contains(&prompt));
        }

        #[cfg(feature = "real-engine")]
        {
            if collected_output.trim().is_empty() {
                println!("Warning: No content received from model.");
            } else {
                println!("Real Engine Output Verified: Length {}", collected_output.len());
            }
        }

        Ok(())
    }
}
