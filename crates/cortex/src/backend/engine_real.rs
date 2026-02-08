#![cfg(feature = "real-engine")]

use crate::Engine;
use anyhow::{anyhow, Result};
use async_std::task;
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn get_llama_backend() -> Arc<LlamaBackend> {
    LLAMA_BACKEND
        .get_or_init(|| Arc::new(LlamaBackend::init().expect("Failed to init llama backend")))
        .clone()
}

pub struct Brain {
    model: Option<Arc<LlamaModel>>,
    backend: Arc<LlamaBackend>,
    model_loaded: bool,
}

impl Brain {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self {
            model: None,
            backend: get_llama_backend(),
            model_loaded: false,
        }
    }
}

#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Load model
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load model from {}: {}", model_path, e))?;
        self.model = Some(Arc::new(model));
        self.model_loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model_loaded = false;
        self.model = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn default_model(&self) -> String {
        "Qwen/Qwen2.5-1.5B-Instruct".to_string()
    }

    async fn infer(
        &mut self,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
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
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(config.context_size.and_then(|s| NonZeroU32::new(s)));

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

            // Prepare Batch for Prompt
            let n_tokens = tokens_list.len();
            let mut batch = LlamaBatch::new(2048, 1); // Ensure batch size can handle context

            // Load prompt into batch
            let last_index = n_tokens as i32 - 1;
            for (i, token) in tokens_list.iter().enumerate() {
                // add(token, pos, &[seq_id], logits)
                // We only need logits for the very last token to predict the next one
                let _ = batch.add(*token, i as i32, &[0], i as i32 == last_index);
            }

            // Decode Prompt
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Decode prompt failed: {}", e))),
                );
                return;
            }

            // Generation Loop
            let mut n_cur = n_tokens as i32;
            let n_decode = 0; // generated tokens count
            let max_tokens = 512; // Hard limit for safety

            let mut in_think_block = false;
            let mut token_str_buffer = String::new();

            loop {
                // Sample next token
                let mut sampler = LlamaSampler::greedy();
                let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

                // Decode token to string
                let token_str = match model.token_to_str(next_token, Special::Plaintext) {
                    Ok(s) => s.to_string(),
                    Err(_) => "??".to_string(),
                };

                // Check for EOS
                if next_token == model.token_eos() || n_decode >= max_tokens {
                    break;
                }

                // Parse Logic for <think> tags
                // Simple stream parsing
                token_str_buffer.push_str(&token_str);

                // If we are NOT in a think block, check if one is starting
                if !in_think_block && config.show_thinking {
                    if token_str_buffer.contains("<think>") {
                        in_think_block = true;
                        // Emit Start Thought event
                        let _ = futures::executor::block_on(
                            tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Start))),
                        );

                        // If there was content before <think>, we should emit it?
                        // For simplicity assuming distinct blocks or just consuming tag.
                        // Remove <think> from buffer to find remainder
                        token_str_buffer = token_str_buffer.replace("<think>", "");
                    }
                }

                // If we ARE in a think block
                if in_think_block {
                    if token_str_buffer.contains("</think>") {
                        in_think_block = false;
                        // Emit Stop Thought event
                        let parts: Vec<&str> = token_str_buffer.split("</think>").collect();
                        if let Some(think_content) = parts.first() {
                            if !think_content.is_empty() {
                                let _ = futures::executor::block_on(tx.send(Ok(
                                    InferenceEvent::Thought(ThoughtEvent::Delta(
                                        think_content.to_string(),
                                    )),
                                )));
                            }
                        }

                        let _ = futures::executor::block_on(
                            tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop))),
                        );

                        // Remainder after </think> should be content?
                        if parts.len() > 1 {
                            token_str_buffer = parts[1].to_string();
                            // Fallthrough to emit content
                        } else {
                            token_str_buffer.clear();
                        }
                    } else {
                        // Stream delta
                        if !token_str_buffer.is_empty() {
                            let _ =
                                futures::executor::block_on(tx.send(Ok(InferenceEvent::Thought(
                                    ThoughtEvent::Delta(token_str_buffer.clone()),
                                ))));
                            token_str_buffer.clear();
                        }
                    }
                }

                // If NOT in think block (anymore), emit as content
                if !in_think_block && !token_str_buffer.is_empty() {
                    let _ = futures::executor::block_on(
                        tx.send(Ok(InferenceEvent::Content(token_str_buffer.clone()))),
                    );
                    token_str_buffer.clear();
                }

                // Prepare next batch
                batch.clear();
                let _ = batch.add(next_token, n_cur, &[0], true);
                n_cur += 1;

                if let Err(e) = ctx.decode(&mut batch) {
                    let _ =
                        futures::executor::block_on(tx.send(Err(anyhow!("Decode failed: {}", e))));
                    break;
                }
            }

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        });

        Ok(rx)
    }

    async fn embed(
        &mut self,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .clone();

        let backend = self.backend.clone();
        let input_str = input.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        task::spawn_blocking(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            // Create context for embeddings
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(config.context_size.and_then(|s| NonZeroU32::new(s)))
                .with_embeddings(true); // Enable embedding mode

            let mut ctx = match model.new_context(backend_ref, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Context creation failed: {}", e))),
                    );
                    return;
                }
            };

            // Tokenize input
            let tokens_list = match model.str_to_token(&input_str, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Tokenize failed: {}", e))),
                    );
                    return;
                }
            };

            // Prepare batch
            let mut batch = LlamaBatch::new(2048, 1);

            // Add all tokens to batch (no need for logits in embedding mode)
            for (i, token) in tokens_list.iter().enumerate() {
                let _ = batch.add(*token, i as i32, &[0], false);
            }

            // Decode to get embeddings
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(tx.send(Err(anyhow!("Decode failed: {}", e))));
                return;
            }

            // Extract embeddings from the context
            // The embeddings are typically available after decode
            // Extract embeddings from the context
            let embeddings = match ctx.embeddings_seq_ith(0) {
                Ok(e) => e.to_vec(),
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Failed to get embeddings from context: {}", e))),
                    );
                    return;
                }
            };

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Embedding(embeddings))));
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        });

        Ok(rx)
    }
}
