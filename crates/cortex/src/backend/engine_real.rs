#![cfg(feature = "real-engine")]

use rusty_genius_core::engine::Engine;
use anyhow::{anyhow, Result};
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
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn get_llama_backend() -> Arc<LlamaBackend> {
    LLAMA_BACKEND
        .get_or_init(|| Arc::new(LlamaBackend::init().expect("Failed to init llama backend")))
        .clone()
}

pub struct Brain {
    /// Cached models for inference, keyed by model path.
    infer_models: HashMap<String, Arc<LlamaModel>>,
    /// Cached models for embedding, keyed by model path.
    embed_models: HashMap<String, Arc<LlamaModel>>,
    /// The currently active model path (set by load_model).
    current_model_path: Option<String>,
    backend: Arc<LlamaBackend>,
}

impl Brain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or load a model for inference. Only uses already-loaded models
    /// or the model at `current_model_path`. Does not pull/download.
    fn get_infer_model(&mut self, model_path: &str) -> Result<Arc<LlamaModel>> {
        if let Some(model) = self.infer_models.get(model_path) {
            return Ok(model.clone());
        }
        // Load from disk (must already be downloaded via ensure_models).
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load infer model {}: {}", model_path, e))?;
        let model = Arc::new(model);
        self.infer_models.insert(model_path.to_string(), model.clone());
        Ok(model)
    }

    /// Get or load a model for embedding. Only uses already-loaded models
    /// or the model at `current_model_path`. Does not pull/download.
    fn get_embed_model(&mut self, model_path: &str) -> Result<Arc<LlamaModel>> {
        if let Some(model) = self.embed_models.get(model_path) {
            return Ok(model.clone());
        }
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load embed model {}: {}", model_path, e))?;
        let model = Arc::new(model);
        self.embed_models.insert(model_path.to_string(), model.clone());
        Ok(model)
    }

    /// Resolve which model path to use: explicit override or current default.
    fn resolve_model_path(&self, override_path: Option<&str>) -> Result<String> {
        if let Some(p) = override_path {
            return Ok(p.to_string());
        }
        self.current_model_path
            .clone()
            .ok_or_else(|| anyhow!("No model loaded and no override specified"))
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self {
            infer_models: HashMap::new(),
            embed_models: HashMap::new(),
            current_model_path: None,
            backend: get_llama_backend(),
        }
    }
}

#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Just set the current model path. Actual loading is lazy
        // (happens on first infer/embed call) so we don't need to
        // guess whether it's for inference or embedding here.
        self.current_model_path = Some(model_path.to_string());
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        // Remove current model from both caches if present.
        if let Some(path) = self.current_model_path.take() {
            self.infer_models.remove(&path);
            self.embed_models.remove(&path);
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.current_model_path.is_some()
    }

    fn default_model(&self) -> String {
        "Qwen/Qwen2.5-1.5B-Instruct".to_string()
    }

    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model_path = self.resolve_model_path(model)?;
        let model = self.get_infer_model(&model_path)?;

        let backend = self.backend.clone();
        let prompt_str = prompt.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            // Create context — inference uses default (causal) mode.
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

            let n_tokens = tokens_list.len();
            let mut batch = LlamaBatch::new(2048, 1);

            let last_index = n_tokens as i32 - 1;
            for (i, token) in tokens_list.iter().enumerate() {
                let _ = batch.add(*token, i as i32, &[0], i as i32 == last_index);
            }

            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Decode prompt failed: {}", e))),
                );
                return;
            }

            let mut n_cur = n_tokens as i32;
            let n_decode = 0;
            let max_tokens = config.max_tokens.unwrap_or(512);

            let mut in_think_block = false;
            let mut token_str_buffer = String::new();

            loop {
                let mut sampler = LlamaSampler::greedy();
                let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

                let token_str = match model.token_to_str(next_token, Special::Plaintext) {
                    Ok(s) => s.to_string(),
                    Err(_) => "??".to_string(),
                };

                if next_token == model.token_eos() || n_decode >= max_tokens {
                    break;
                }

                token_str_buffer.push_str(&token_str);

                if !in_think_block && config.show_thinking {
                    if token_str_buffer.contains("<think>") {
                        in_think_block = true;
                        let _ = futures::executor::block_on(
                            tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Start))),
                        );
                        token_str_buffer = token_str_buffer.replace("<think>", "");
                    }
                }

                if in_think_block {
                    if token_str_buffer.contains("</think>") {
                        in_think_block = false;
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

                        if parts.len() > 1 {
                            token_str_buffer = parts[1].to_string();
                        } else {
                            token_str_buffer.clear();
                        }
                    } else {
                        if !token_str_buffer.is_empty() {
                            let _ =
                                futures::executor::block_on(tx.send(Ok(InferenceEvent::Thought(
                                    ThoughtEvent::Delta(token_str_buffer.clone()),
                                ))));
                            token_str_buffer.clear();
                        }
                    }
                }

                if !in_think_block && !token_str_buffer.is_empty() {
                    let _ = futures::executor::block_on(
                        tx.send(Ok(InferenceEvent::Content(token_str_buffer.clone()))),
                    );
                    token_str_buffer.clear();
                }

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
        }))
        .detach();

        Ok(rx)
    }

    async fn embed(
        &mut self,
        model: Option<&str>,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model_path = self.resolve_model_path(model)?;
        let model = self.get_embed_model(&model_path)?;

        let backend = self.backend.clone();
        let input_str = input.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            // Create context — embedding mode.
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(config.context_size.and_then(|s| NonZeroU32::new(s)))
                .with_embeddings(true);

            let mut ctx = match model.new_context(backend_ref, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Context creation failed: {}", e))),
                    );
                    return;
                }
            };

            let tokens_list = match model.str_to_token(&input_str, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Tokenize failed: {}", e))),
                    );
                    return;
                }
            };

            let mut batch = LlamaBatch::new(2048, 1);

            // Mark all tokens for output so per-token embeddings are
            // available for mean-pooling with POOLING_TYPE_NONE models.
            for (i, token) in tokens_list.iter().enumerate() {
                let _ = batch.add(*token, i as i32, &[0], true);
            }

            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(tx.send(Err(anyhow!("Decode failed: {}", e))));
                return;
            }

            // Try sequence-level pooled embeddings first; fall back to
            // mean-pooling per-token embeddings for POOLING_TYPE_NONE.
            let embeddings = match ctx.embeddings_seq_ith(0) {
                Ok(e) => e.to_vec(),
                Err(_) => {
                    let n_tokens = tokens_list.len();
                    if n_tokens == 0 {
                        let _ = futures::executor::block_on(
                            tx.send(Err(anyhow!("No tokens to embed"))),
                        );
                        return;
                    }
                    let dim = match ctx.embeddings_ith(0) {
                        Ok(e) => e.len(),
                        Err(e) => {
                            let _ = futures::executor::block_on(
                                tx.send(Err(anyhow!("Failed to get token embedding: {}", e))),
                            );
                            return;
                        }
                    };
                    let mut mean = vec![0.0f32; dim];
                    for i in 0..n_tokens {
                        if let Ok(tok_emb) = ctx.embeddings_ith(i as i32) {
                            for (j, val) in tok_emb.iter().enumerate() {
                                mean[j] += val;
                            }
                        }
                    }
                    let scale = 1.0 / n_tokens as f32;
                    for v in &mut mean {
                        *v *= scale;
                    }
                    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-8 {
                        for v in &mut mean {
                            *v /= norm;
                        }
                    }
                    mean
                }
            };

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Embedding(embeddings))));
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        }))
        .detach();

        Ok(rx)
    }
}
