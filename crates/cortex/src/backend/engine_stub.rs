#![cfg(not(feature = "real-engine"))]

use crate::Engine;
use anyhow::{anyhow, Result};
use async_std::task;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
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
        prompt: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let prompt_owned = prompt.to_string();
        eprintln!("DEBUG: Pinky::infer prompt: {}", prompt_owned);
        task::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            task::sleep(Duration::from_millis(50)).await;

            // Emit a "thought"
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                .await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                    "Narf!".to_string(),
                ))))
                .await;
            task::sleep(Duration::from_millis(50)).await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                .await;

            // Emit content (echo prompt mostly)
            let _ = tx
                .send(Ok(InferenceEvent::Content(format!(
                    "Pinky says: {}",
                    prompt_owned
                ))))
                .await;

            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        });

        Ok(rx)
    }

    async fn embed(
        &mut self,
        input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let input_owned = input.to_string();
        eprintln!("DEBUG: Pinky::embed input: {}", input_owned);
        task::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            task::sleep(Duration::from_millis(50)).await;

            // Generate a simple mock embedding (384 dimensions with random-ish values)
            let mock_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();

            let _ = tx.send(Ok(InferenceEvent::Embedding(mock_embedding))).await;
            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        });

        Ok(rx)
    }
}
