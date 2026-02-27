use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::error::GeniusError;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::memory::EmbeddingProvider;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceEvent,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Embedding provider that routes through the brainstem orchestrator.
/// Sends `BrainstemCommand::Embed` and waits for `InferenceEvent::Embedding`.
pub struct BrainstemEmbedder {
    input_tx: mpsc::Sender<BrainstemInput>,
    output_rx: Arc<futures::lock::Mutex<mpsc::Receiver<BrainstemOutput>>>,
}

impl BrainstemEmbedder {
    pub fn new(
        input_tx: mpsc::Sender<BrainstemInput>,
        output_rx: mpsc::Receiver<BrainstemOutput>,
    ) -> Self {
        Self {
            input_tx,
            output_rx: Arc::new(futures::lock::Mutex::new(output_rx)),
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EmbeddingProvider for BrainstemEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, GeniusError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let request_id = format!("emb-{}", timestamp);

        let input = BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::Embed {
                model: None, // orchestrator auto-selects "embedding-gemma"
                input: text.to_string(),
                config: InferenceConfig::default(),
            },
        };

        self.input_tx
            .clone()
            .send(input)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Failed to send embed request: {}", e)))?;

        // Wait for matching response
        let mut rx = self.output_rx.lock().await;
        while let Some(output) = rx.next().await {
            // Match by request ID
            if output.id.as_deref() != Some(&request_id) {
                continue;
            }

            match output.body {
                BrainstemBody::Event(InferenceEvent::Embedding(vec)) => {
                    return Ok(vec);
                }
                BrainstemBody::Error(e) => {
                    return Err(GeniusError::MemoryError(format!("Embedding error: {}", e)));
                }
                // Skip intermediate events (ProcessStart, Complete, etc.)
                _ => continue,
            }
        }

        Err(GeniusError::MemoryError(
            "Embedding channel closed without response".to_string(),
        ))
    }
}
