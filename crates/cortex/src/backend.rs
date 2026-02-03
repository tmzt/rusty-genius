use crate::Engine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

// --- Pinky (Stub) ---

pub struct Pinky {
    model_loaded: bool,
}

impl Pinky {
    pub fn new() -> Self {
        Self {
            model_loaded: false,
        }
    }
}

#[async_trait]
impl Engine for Pinky {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        // Simulate loading time
        sleep(Duration::from_millis(100)).await;
        self.model_loaded = true;
        Ok(())
    }

    async fn infer(&mut self, prompt: &str) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (tx, rx) = mpsc::channel(100);
        let prompt = prompt.to_string();

        tokio::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            sleep(Duration::from_millis(50)).await;

            // Emit a "thought"
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                .await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                    "Narf!".to_string(),
                ))))
                .await;
            sleep(Duration::from_millis(50)).await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                .await;

            // Emit content (echo prompt mostly)
            let _ = tx
                .send(Ok(InferenceEvent::Content(format!(
                    "Pinky says: {}",
                    prompt
                ))))
                .await;

            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        });

        Ok(rx)
    }
}

// --- Brain (Real) ---

#[cfg(feature = "real-engine")]
pub struct Brain {
    // actual llama-cpp state would go here
}

#[cfg(feature = "real-engine")]
impl Brain {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(feature = "real-engine")]
#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        // TODO: Implement actual llama.cpp loading
        Ok(())
    }

    async fn infer(&mut self, _prompt: &str) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let (tx, rx) = mpsc::channel(100);

        // TODO: Implement actual inference loop
        // For now, fail or dummy since I don't have the weights
        tokio::spawn(async move {
            let _ = tx
                .send(Err(anyhow!(
                    "Brain implementation pending wiring to llama-cpp-2"
                )))
                .await;
        });

        Ok(rx)
    }
}
