#![cfg(not(feature = "real-engine"))]

use crate::Engine;
use anyhow::{anyhow, Result};
use async_std::task;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::manifest::EngineConfig;
use rusty_genius_thinkerv1::{EventResponse, Response};
use std::time::Duration;

#[derive(Default)]
pub struct Pinky {
    model_loaded: bool,
}

impl Pinky {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Engine for Pinky {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        // Simulate loading time
        task::sleep(Duration::from_millis(100)).await;
        self.model_loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model_loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    fn default_model(&self) -> String {
        "tiny-model".to_string()
    }

    async fn infer(
        &mut self,
        id: String,
        prompt: &str,
        config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<Response>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let prompt_owned = prompt.to_string();
        let id_cloned = id.clone();
        eprintln!("DEBUG: Pinky::infer prompt: {}", prompt_owned);
        task::spawn(async move {
            if config.show_thinking {
                // Emit a "thought"
                let _ = tx
                    .send(Ok(Response::Event(EventResponse::Thought {
                        id: id_cloned.clone(),
                        content: "Narf!".to_string(),
                    })))
                    .await;
                task::sleep(Duration::from_millis(50)).await;
            }

            // Emit content (echo prompt mostly)
            let _ = tx
                .send(Ok(Response::Event(EventResponse::Content {
                    id: id_cloned.clone(),
                    content: format!("Pinky says: {}", prompt_owned),
                })))
                .await;

            let _ = tx
                .send(Ok(Response::Event(EventResponse::Complete {
                    id: id_cloned,
                })))
                .await;
        });

        Ok(rx)
    }

    async fn embed(
        &mut self,
        id: String,
        input: &str,
        _config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<Response>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let input_owned = input.to_string();
        let id_cloned = id.clone();
        eprintln!("DEBUG: Pinky::embed input: {}", input_owned);
        task::spawn(async move {
            // Generate a simple mock embedding (384 dimensions with random-ish values)
            let mock_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
            let mock_embedding_hex = mock_embedding
                .iter()
                .map(|f| f.to_bits().to_be_bytes())
                .flatten()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();

            let _ = tx
                .send(Ok(Response::Event(EventResponse::Embedding {
                    id: id_cloned.clone(),
                    vector_hex: mock_embedding_hex,
                })))
                .await;
            let _ = tx
                .send(Ok(Response::Event(EventResponse::Complete {
                    id: id_cloned,
                })))
                .await;
        });

        Ok(rx)
    }
}
