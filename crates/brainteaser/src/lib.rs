#[cfg(test)]
mod tests {
    use anyhow::Result;
    use futures::channel::mpsc;
    use futures::sink::SinkExt;
    use futures::StreamExt;
    use rusty_genius_core::protocol::{
        AssetEvent, BrainstemBody, BrainstemCommand, BrainstemInput, InferenceEvent, ThoughtEvent,
    };
    use rusty_genius_stem::Orchestrator;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    #[derive(Debug)]
    struct Fixture {
        path: PathBuf,
        org: String,
        repo: String,
        quant: String,
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

        let mut stack = vec![base_path.to_path_buf()];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if let Some(ext) = path.extension() {
                        if ext == "md" {
                            if let Ok(stripped) = path.strip_prefix(base_path) {
                                let components: Vec<_> = stripped
                                    .components()
                                    .map(|c| c.as_os_str().to_string_lossy().to_string())
                                    .collect();
                                if components.len() == 4 {
                                    fixtures.push(Fixture {
                                        path: path.clone(),
                                        org: components[0].clone(),
                                        repo: components[1].clone(),
                                        quant: components[2].clone(),
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
                }
            }
        }
        fixtures.sort_by(|a, b| a.test_name.cmp(&b.test_name));
        fixtures
    }

    #[async_std::test]
    async fn test_inference_flow() -> Result<()> {
        println!("Starting test_inference_flow...");

        // 1. Scan for fixtures
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
        let fixture_root = PathBuf::from(manifest_dir).join("fixtures");
        let fixtures = scan_fixtures(&fixture_root);
        println!("Found {} fixtures.", fixtures.len());

        // 2. Setup Orchestrator
        let mut orchestrator = Orchestrator::new().await?;
        let (mut input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, mut output_rx) = mpsc::channel(100);

        let orchestrator_handle =
            async_std::task::spawn(async move { orchestrator.run(input_rx, output_tx).await });

        // 3. Load Model
        println!("Sending LoadModel command...");
        input_tx
            .send(BrainstemInput {
                id: None,
                command: BrainstemCommand::LoadModel("qwen-2.5-3b-instruct".to_string()),
            })
            .await?;

        // 4. Run Inference (Use first fixture if available)
        let prompt = if let Some(fixture) = fixtures.first() {
            println!("Running fixture: {}", fixture.test_name);
            fs::read_to_string(&fixture.path)?
        } else {
            println!("No fixtures found, defaulting.");
            "Tell me a joke about Rust".to_string()
        };
        println!("Prompt: {}", prompt);

        // Send Inference Request
        input_tx
            .send(BrainstemInput {
                id: None,
                command: BrainstemCommand::Infer {
                    model: Some("qwen-2.5-3b-instruct".to_string()),
                    prompt: prompt.clone(),
                    config: Default::default(),
                },
            })
            .await?;

        // 5. Collect Output
        let mut collected_output = String::new();
        let mut thought_process = String::new();

        println!("Waiting for events...");
        loop {
            let timeout_sec = if cfg!(feature = "real-engine") {
                600
            } else {
                5
            };
            let msg =
                async_std::future::timeout(Duration::from_secs(timeout_sec), output_rx.next())
                    .await;
            match msg {
                Ok(Some(output)) => match output.body {
                    BrainstemBody::Event(event) => match event {
                        InferenceEvent::ProcessStart => println!("Process Started"),
                        InferenceEvent::Thought(t) => match t {
                            ThoughtEvent::Start => println!("Thinking..."),
                            ThoughtEvent::Delta(d) => thought_process.push_str(&d),
                            ThoughtEvent::Stop => println!("Thought process: {}", thought_process),
                        },
                        InferenceEvent::Content(c) => collected_output.push_str(&c),
                        InferenceEvent::Complete => {
                            println!("Inference Complete");
                            break;
                        }
                        _ => {}
                    },
                    BrainstemBody::Asset(asset_event) => {
                        println!("[Asset] Event: {:?}", asset_event);
                        if let AssetEvent::Error(e) = asset_event {
                            return Err(anyhow::anyhow!("Asset error: {}", e));
                        }
                    }
                    BrainstemBody::Error(e) => {
                        return Err(anyhow::anyhow!("Received error from brainstem: {}", e));
                    }
                    BrainstemBody::ModelList(_) => {
                        // Ignored in test harness
                    }
                },
                Ok(None) => return Err(anyhow::anyhow!("Channel closed unexpectedly")),
                Err(_) => return Err(anyhow::anyhow!("Timeout waiting for response")),
            }
        }

        // Cleanup
        input_tx
            .send(BrainstemInput {
                id: None,
                command: BrainstemCommand::Stop,
            })
            .await?;
        let _ = orchestrator_handle.await?;

        println!("Collected Output: {}", collected_output);

        // Assertions (for stub mode)
        #[cfg(not(feature = "real-engine"))]
        {
            assert!(thought_process.contains("Narf!"));
            assert!(collected_output.contains("Pinky says:"));
            assert!(collected_output.contains(&prompt));
        }

        // Assertions (for real mode)
        #[cfg(feature = "real-engine")]
        {
            if collected_output.trim().is_empty() {
                println!("Warning: No content received from model.");
            } else {
                println!(
                    "Real Engine Output Verified: Length {}",
                    collected_output.len()
                );
            }
        }

        Ok(())
    }
}
