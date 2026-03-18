#![cfg(feature = "real-engine")]

use rusty_genius_core::engine::Engine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
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

// ── CortexContext: typed wrapper for inference vs embedding ──

/// A context that knows whether it should decode (inference/causal) or
/// encode (embedding/encoder-only).
enum CortexContext<'a> {
    /// Causal decoder context — uses `decode()` for autoregressive generation.
    Inference(LlamaContext<'a>),
    /// Encoder context — uses `encode()` for embedding extraction.
    Embedding(LlamaContext<'a>),
}

impl<'a> CortexContext<'a> {
    /// Create an inference (causal decoder) context.
    fn new_inference(
        model: &'a LlamaModel,
        backend: &'a LlamaBackend,
        config: &InferenceConfig,
    ) -> Result<Self> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(config.context_size.and_then(NonZeroU32::new));
        let ctx = model.new_context(backend, ctx_params)
            .map_err(|e| anyhow!("Inference context creation failed: {}", e))?;
        Ok(CortexContext::Inference(ctx))
    }

    /// Create an embedding (encoder) context.
    fn new_embedding(
        model: &'a LlamaModel,
        backend: &'a LlamaBackend,
        config: &InferenceConfig,
    ) -> Result<Self> {
        let n_ctx = config.context_size.unwrap_or(512);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::None)
            .with_flash_attention_policy(0);
        let ctx = model.new_context(backend, ctx_params)
            .map_err(|e| anyhow!("Embedding context creation failed: {}", e))?;
        Ok(CortexContext::Embedding(ctx))
    }

    /// Process a batch through the appropriate path.
    fn process(&mut self, batch: &mut LlamaBatch) -> Result<()> {
        match self {
            CortexContext::Inference(ctx) => {
                ctx.decode(batch)
                    .map_err(|e| anyhow!("decode failed: {}", e))
            }
            CortexContext::Embedding(ctx) => {
                ctx.encode(batch)
                    .map_err(|e| anyhow!("encode failed: {}", e))
            }
        }
    }

    /// Get the inner LlamaContext (for sampling, embedding extraction, etc.).
    fn inner(&self) -> &LlamaContext<'a> {
        match self {
            CortexContext::Inference(ctx) | CortexContext::Embedding(ctx) => ctx,
        }
    }
}

// ── LlamaCppEngine: the Engine implementation ──

pub struct LlamaCppEngine {
    /// Cached models for inference, keyed by model path.
    infer_models: HashMap<String, Arc<LlamaModel>>,
    /// Cached models for embedding, keyed by model path.
    embed_models: HashMap<String, Arc<LlamaModel>>,
    backend: Arc<LlamaBackend>,
}

impl LlamaCppEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or load a model for inference.
    fn get_infer_model(&mut self, model_path: &str) -> Result<Arc<LlamaModel>> {
        if let Some(model) = self.infer_models.get(model_path) {
            return Ok(model.clone());
        }
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load infer model {}: {}", model_path, e))?;
        let model = Arc::new(model);
        self.infer_models.insert(model_path.to_string(), model.clone());
        Ok(model)
    }

    /// Get or load a model for embedding.
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
}

impl Default for LlamaCppEngine {
    fn default() -> Self {
        Self {
            infer_models: HashMap::new(),
            embed_models: HashMap::new(),
            backend: get_llama_backend(),
        }
    }
}

impl Drop for LlamaCppEngine {
    fn drop(&mut self) {
        self.infer_models.clear();
        self.embed_models.clear();
    }
}

#[async_trait]
impl Engine for LlamaCppEngine {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.infer_models.clear();
        self.embed_models.clear();
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        !self.infer_models.is_empty() || !self.embed_models.is_empty()
    }

    fn default_model(&self) -> String {
        "Qwen/Qwen2.5-1.5B-Instruct".to_string()
    }

    async fn preload_model(&mut self, model_path: &str, purpose: &str) -> Result<()> {
        match purpose {
            "embed" => { self.get_embed_model(model_path)?; }
            _ => { self.get_infer_model(model_path)?; }
        }
        Ok(())
    }

    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model_path = model.ok_or_else(|| anyhow!("model path required for infer"))?;
        let model = self.get_infer_model(model_path)?;

        let backend = self.backend.clone();
        let prompt_str = prompt.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            let mut cortex = match CortexContext::new_inference(&model, backend_ref, &config) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(tx.send(Err(e)));
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

            if let Err(e) = cortex.process(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Decode prompt failed: {}", e))),
                );
                return;
            }

            let mut n_cur = n_tokens as i32;
            let mut n_decode: usize = 0;
            let max_tokens = config.max_tokens.unwrap_or(512);

            let mut in_think_block = false;
            let mut token_str_buffer = String::new();

            loop {
                let mut sampler = LlamaSampler::greedy();
                let next_token = sampler.sample(cortex.inner(), batch.n_tokens() - 1);

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

                n_decode += 1;
                batch.clear();
                let _ = batch.add(next_token, n_cur, &[0], true);
                n_cur += 1;

                if let Err(e) = cortex.process(&mut batch) {
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
        let model_path = model.ok_or_else(|| anyhow!("model path required for embed"))?;
        let model = self.get_embed_model(model_path)?;

        let backend = self.backend.clone();
        let input_str = input.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            let mut cortex = match CortexContext::new_embedding(&model, backend_ref, &config) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(tx.send(Err(e)));
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

            // Encoder path: use encode() directly for embedding models.
            if let Err(e) = cortex.process(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Encode failed: {}", e))),
                );
                return;
            }

            // Try sequence-level pooled embeddings first; fall back to
            // mean-pooling per-token embeddings for POOLING_TYPE_NONE.
            let ctx = cortex.inner();
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
