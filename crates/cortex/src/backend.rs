use crate::Engine;
use anyhow::{anyhow, Result};
use async_std::task::{self, sleep};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
use std::time::Duration;

#[cfg(feature = "real-engine")]
use llama_cpp_2::context::params::LlamaContextParams;
#[cfg(feature = "real-engine")]
use llama_cpp_2::llama_backend::LlamaBackend;
#[cfg(feature = "real-engine")]
use llama_cpp_2::llama_batch::LlamaBatch;
#[cfg(feature = "real-engine")]
use llama_cpp_2::model::params::LlamaModelParams;
#[cfg(feature = "real-engine")]
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
#[cfg(feature = "real-engine")]
use llama_cpp_2::token::LlamaToken;
#[cfg(feature = "real-engine")]
use std::num::NonZeroU32;
use std::sync::Arc;

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

    async fn unload_model(&mut self) -> Result<()> {
        self.model_loaded = false;
        Ok(())
    }

    async fn infer(&mut self, prompt: &str) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let prompt = prompt.to_string();

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
    model: Option<Arc<LlamaModel>>,
    backend: Arc<LlamaBackend>,
}

#[cfg(feature = "real-engine")]
impl Brain {
    pub fn new() -> Self {
        Self {
            model: None,
            backend: Arc::new(LlamaBackend::init().expect("Failed to init llama backend")),
        }
    }
}

#[cfg(feature = "real-engine")]
#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Load model
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load model from {}: {}", model_path, e))?;
        self.model = Some(Arc::new(model));
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model = None;
        Ok(())
    }

    async fn infer(&mut self, prompt: &str) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .clone();

        // Share the backend reference
        let backend = self.backend.clone();

        let prompt_str = prompt.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        task::spawn_blocking(move || {
            // Send ProcessStart
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            // Use the shared backend (no re-init)
            let backend_ref = &backend;

            // Create context
            let ctx_params =
                LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));

            let mut ctx = match model.new_context(backend_ref, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Context creation failed: {}", e))),
                    );
                    return;
                }
            };

            // Tokenize
            let tokens_list = match model.str_to_token(&prompt_str, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Tokenize failed: {}", e))),
                    );
                    return;
                }
            };

            // Prepare Batch
            // LlamaBatch::new(n_tokens, n_seq_max)
            let mut batch = LlamaBatch::new(512, 1);

            // Load prompt into batch
            let last_index = tokens_list.len() as i32 - 1;
            for (i, token) in tokens_list.iter().enumerate() {
                // add(token, pos, &[seq_id], logits)
                let _ = batch.add(*token, i as i32, &[0], i as i32 == last_index);
            }

            // Decode Prompt
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Decode prompt failed: {}", e))),
                );
                return;
            }

            let _ = futures::executor::block_on(
                tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Start))),
            );

            // Emit success message demonstrating Real Engine loaded and Decoded.
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Content(format!("(Real Engine) Model Loaded & Decoded {} tokens. [Sampling implementation simplified]", tokens_list.len())))));

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        });

        Ok(rx)
    }
}
