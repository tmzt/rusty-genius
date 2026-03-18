#![cfg(feature = "mlx")]

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use hf_hub::api::sync::Api;
use mlx_rs::module::{Module, ModuleParametersExt};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::Array;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use super::qwen35;

pub const MLX_DEFAULT_MODEL: &str = "mlx-community/Qwen3.5-9B-MLX-4bit";

// ── Model config (loaded from config.json) ──

#[derive(Debug, Deserialize)]
struct QwenConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

// ── Qwen model layers ──

fn make_linear_no_bias(in_dim: i32, out_dim: i32) -> Result<nn::Linear> {
    // Linear::new creates with bias; we manually create without bias
    let scale = f32::sqrt(1.0 / in_dim as f32);
    let weight = mlx_rs::random::uniform::<_, f32>(-scale, scale, &[out_dim, in_dim], None)
        .map_err(|e| anyhow!("Linear weight init: {e}"))?;
    Ok(nn::Linear {
        weight: mlx_rs::module::Param::new(weight),
        bias: mlx_rs::module::Param::new(None),
    })
}

fn make_rms_norm(dims: i32, eps: f32) -> Result<nn::RmsNorm> {
    let weight = mlx_rs::ops::ones::<f32>(&[dims])
        .map_err(|e| anyhow!("RmsNorm weight: {e}"))?;
    Ok(nn::RmsNorm {
        weight: mlx_rs::module::Param::new(weight),
        eps,
    })
}

struct QwenRotaryEmbedding {
    inv_freq: Array,
}

impl QwenRotaryEmbedding {
    fn new(dim: usize, theta: f64) -> Result<Self> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Array::from_slice(&inv_freq, &[half_dim as i32]),
        })
    }

    fn apply(&self, x: &Array, offset: usize) -> Result<Array> {
        let seq_len = x.shape()[1] as usize;
        let half = (*x.shape().last().unwrap() / 2) as i32;
        let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
        let t = Array::from_slice(&positions, &[seq_len as i32]);

        // freqs: [seq_len, half_dim]
        let freqs = ops::outer(&t, &self.inv_freq)
            .map_err(|e| anyhow!("outer failed: {e}"))?;
        let cos_f = freqs.cos().map_err(|e| anyhow!("cos: {e}"))?;
        let sin_f = freqs.sin().map_err(|e| anyhow!("sin: {e}"))?;

        // Reshape for broadcasting: [1, seq_len, 1, half_dim]
        let cos_f = cos_f.reshape(&[1, seq_len as i32, 1, half])?;
        let sin_f = sin_f.reshape(&[1, seq_len as i32, 1, half])?;

        // Split x into halves along last dim
        // x shape: [batch, seq_len, heads, head_dim]
        let head_dim = *x.shape().last().unwrap();
        let x_flat = x.reshape(&[-1, head_dim])?;
        let x1_flat = x_flat.index((.., ..half));
        let x2_flat = x_flat.index((.., half..));
        let orig_shape = x.shape().to_vec();
        let mut half_shape = orig_shape.clone();
        *half_shape.last_mut().unwrap() = half;
        let x1 = x1_flat.reshape(&half_shape)?;
        let x2 = x2_flat.reshape(&half_shape)?;

        let rotated = ops::concatenate_axis(
            &[
                &x1.multiply(&cos_f)?.subtract(&x2.multiply(&sin_f)?)?,
                &x2.multiply(&cos_f)?.add(&x1.multiply(&sin_f)?)?,
            ],
            -1,
        ).map_err(|e| anyhow!("concatenate: {e}"))?;
        Ok(rotated)
    }
}

#[derive(ModuleParameters)]
struct QwenAttention {
    #[param]
    q_proj: nn::Linear,
    #[param]
    k_proj: nn::Linear,
    #[param]
    v_proj: nn::Linear,
    #[param]
    o_proj: nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: QwenRotaryEmbedding,
}

impl QwenAttention {
    fn new(config: &QwenConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_dim = (config.num_attention_heads * head_dim) as i32;
        let kv_dim = (config.num_key_value_heads * head_dim) as i32;
        let h = config.hidden_size as i32;
        Ok(Self {
            q_proj: make_linear_no_bias(h, q_dim)?,
            k_proj: make_linear_no_bias(h, kv_dim)?,
            v_proj: make_linear_no_bias(h, kv_dim)?,
            o_proj: make_linear_no_bias(q_dim, h)?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope: QwenRotaryEmbedding::new(head_dim, config.rope_theta)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&Array, &Array)>,
    ) -> Result<(Array, Array, Array)> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        let q = self.q_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let k = self.k_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let v = self.v_proj.forward(x).map_err(|e| anyhow!("{e}"))?;

        let hd = self.head_dim as i32;
        let q = q.reshape(&[batch, seq_len, self.num_heads as i32, hd])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads as i32, hd])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads as i32, hd])?;

        let offset = cache.map(|(ck, _)| ck.shape()[1] as usize).unwrap_or(0);
        let q = self.rope.apply(&q, offset)?;
        let k = self.rope.apply(&k, offset)?;

        // Concatenate with KV cache
        let (k, v) = if let Some((ck, cv)) = cache {
            (
                ops::concatenate_axis(&[ck, &k], 1).map_err(|e| anyhow!("{e}"))?,
                ops::concatenate_axis(&[cv, &v], 1).map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            (k, v)
        };

        // Transpose to [batch, heads, seq, dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // GQA: repeat KV heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeats = (self.num_heads / self.num_kv_heads) as i32;
            // Expand via broadcast: insert repeat dim, reshape
            let kv_seq = k.shape()[2];
            let k = k.reshape(&[batch, self.num_kv_heads as i32, 1, kv_seq, hd])?;
            let k = ops::broadcast_to(&k, &[batch, self.num_kv_heads as i32, repeats, kv_seq, hd])
                .map_err(|e| anyhow!("{e}"))?;
            let k = k.reshape(&[batch, self.num_heads as i32, kv_seq, hd])?;

            let v = v.reshape(&[batch, self.num_kv_heads as i32, 1, kv_seq, hd])?;
            let v = ops::broadcast_to(&v, &[batch, self.num_kv_heads as i32, repeats, kv_seq, hd])
                .map_err(|e| anyhow!("{e}"))?;
            let v = v.reshape(&[batch, self.num_heads as i32, kv_seq, hd])?;
            (k, v)
        } else {
            (k, v)
        };

        // Scaled dot-product attention
        let scale = Array::from_slice(&[(self.head_dim as f32).sqrt()], &[1]);
        let kt = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&kt)?.divide(&scale)?;

        let scores = if let Some(m) = mask {
            scores.add(m)?
        } else {
            scores
        };

        let weights = ops::softmax_axis(&scores, -1, None)
            .map_err(|e| anyhow!("{e}"))?;
        let out = weights.matmul(&v)?;

        // Transpose back: [batch, seq, heads, dim]
        let out = out.transpose_axes(&[0, 2, 1, 3])?;
        let out = out.reshape(&[batch, seq_len, (self.num_heads * self.head_dim) as i32])?;
        let out = self.o_proj.forward(&out).map_err(|e| anyhow!("{e}"))?;

        // Return k,v in [batch, seq, heads, dim] form for cache
        let k_cache = k.transpose_axes(&[0, 2, 1, 3])?;
        let v_cache = v.transpose_axes(&[0, 2, 1, 3])?;

        Ok((out, k_cache, v_cache))
    }
}

#[derive(ModuleParameters)]
struct QwenMLP {
    #[param]
    gate_proj: nn::Linear,
    #[param]
    up_proj: nn::Linear,
    #[param]
    down_proj: nn::Linear,
}

impl QwenMLP {
    fn new(config: &QwenConfig) -> Result<Self> {
        let h = config.hidden_size as i32;
        let i = config.intermediate_size as i32;
        Ok(Self {
            gate_proj: make_linear_no_bias(h, i)?,
            up_proj: make_linear_no_bias(h, i)?,
            down_proj: make_linear_no_bias(i, h)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        // SiLU = x * sigmoid(x)
        let gate = gate.multiply(&ops::sigmoid(&gate).map_err(|e| anyhow!("{e}"))?)?;
        let up = self.up_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden).map_err(|e| anyhow!("{e}"))
    }
}

#[derive(ModuleParameters)]
struct QwenLayer {
    #[param]
    self_attn: QwenAttention,
    #[param]
    mlp: QwenMLP,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl QwenLayer {
    fn new(config: &QwenConfig) -> Result<Self> {
        Ok(Self {
            self_attn: QwenAttention::new(config)?,
            mlp: QwenMLP::new(config)?,
            input_layernorm: make_rms_norm(config.hidden_size as i32, config.rms_norm_eps as f32)?,
            post_attention_layernorm: make_rms_norm(config.hidden_size as i32, config.rms_norm_eps as f32)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<(&Array, &Array)>,
    ) -> Result<(Array, Array, Array)> {
        let residual = x.clone();
        let hidden = self.input_layernorm.forward(x).map_err(|e| anyhow!("{e}"))?;
        let (attn_out, k_cache, v_cache) = self.self_attn.forward(&hidden, mask, cache)?;
        let hidden = residual.add(&attn_out)?;

        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden).map_err(|e| anyhow!("{e}"))?;
        let hidden = self.mlp.forward(&hidden)?;
        let hidden = residual.add(&hidden)?;

        Ok((hidden, k_cache, v_cache))
    }
}

#[derive(ModuleParameters)]
struct QwenModel {
    #[param]
    embed_tokens: nn::Embedding,
    #[param]
    layers: Vec<QwenLayer>,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    lm_head: nn::Linear,
    config: QwenConfig,
}

impl QwenModel {
    fn new(config: QwenConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(QwenLayer::new(&config)?);
        }

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)
                .map_err(|e| anyhow!("Embedding: {e}"))?,
            layers,
            norm: make_rms_norm(config.hidden_size as i32, config.rms_norm_eps as f32)?,
            lm_head: make_linear_no_bias(config.hidden_size as i32, config.vocab_size as i32)?,
            config,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Array,
        caches: &[Option<(Array, Array)>],
    ) -> Result<(Array, Vec<(Array, Array)>)> {
        let mut hidden = self.embed_tokens.forward(input_ids).map_err(|e| anyhow!("{e}"))?;

        let seq_len = input_ids.shape()[1];
        let past_len = caches.first()
            .and_then(|c| c.as_ref())
            .map(|(k, _)| k.shape()[1])
            .unwrap_or(0);

        // Causal mask
        let mask = if seq_len > 1 {
            let total_len = past_len + seq_len;
            let mut mask_data = vec![0.0f32; (seq_len * total_len) as usize];
            for i in 0..seq_len as usize {
                for j in (past_len as usize + i + 1)..total_len as usize {
                    mask_data[i * total_len as usize + j] = f32::NEG_INFINITY;
                }
            }
            Some(Array::from_slice(&mask_data, &[1, 1, seq_len, total_len]))
        } else {
            None
        };

        let mut new_caches = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache = caches.get(i).and_then(|c| c.as_ref()).map(|(k, v)| (k, v));
            let (h, k, v) = layer.forward(&hidden, mask.as_ref(), cache)?;
            hidden = h;
            new_caches.push((k, v));
        }

        let hidden = self.norm.forward(&hidden).map_err(|e| anyhow!("{e}"))?;
        let logits = self.lm_head.forward(&hidden).map_err(|e| anyhow!("{e}"))?;

        Ok((logits, new_caches))
    }
}

// ── HuggingFace model loading ──

fn download_model(repo_id: &str) -> Result<PathBuf> {
    log::info!("downloading MLX model from {}", repo_id);
    let api = Api::new().map_err(|e| anyhow!("HF API init failed: {e}"))?;
    let repo = api.model(repo_id.to_string());

    for file in &["config.json", "tokenizer.json"] {
        repo.get(file).map_err(|e| anyhow!("failed to download {file}: {e}"))?;
    }

    let model_dir = repo.get("config.json")
        .map_err(|e| anyhow!("failed to get model dir: {e}"))?
        .parent()
        .unwrap()
        .to_path_buf();

    log::info!("MLX model cached at {:?}", model_dir);
    Ok(model_dir)
}

fn load_config(model_dir: &std::path::Path) -> Result<QwenConfig> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("failed to read config.json: {e}"))?;
    let config: QwenConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("failed to parse config.json: {e}"))?;
    log::info!(
        "MLX model config: {}L, {}H, {}KV, {}D",
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size,
    );
    Ok(config)
}

fn load_qwen35_config(model_dir: &std::path::Path) -> Result<qwen35::Qwen35Config> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("failed to read config.json: {e}"))?;
    let outer: qwen35::Qwen35ConfigOuter = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("failed to parse Qwen3.5 config: {e}"))?;
    let config = outer.text_config;
    log::info!(
        "Qwen3.5 config: {}L ({}lin/{}full), {}H, head_dim={}, key_dim={}, value_dim={}",
        config.num_hidden_layers,
        config.layer_types.iter().filter(|t| *t == "linear_attention").count(),
        config.layer_types.iter().filter(|t| *t == "full_attention").count(),
        config.num_attention_heads,
        config.head_dim,
        config.key_dim(),
        config.value_dim(),
    );
    Ok(config)
}

fn load_tokenizer(model_dir: &std::path::Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))
}

// ── Model type detection ──

/// Detect whether config.json describes a Qwen3.5 (hybrid) or Qwen2 (basic) model.
fn is_qwen35_config(model_dir: &std::path::Path) -> bool {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let val: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Qwen3.5 has text_config with layer_types
    val.get("text_config")
        .and_then(|tc| tc.get("layer_types"))
        .is_some()
}

// ── Loaded model enum ──

enum LoadedModel {
    Qwen2(QwenModel),
    Qwen35 {
        model: qwen35::model::Qwen35Model,
        caches: Vec<qwen35::cache::LayerCache>,
    },
}

// ── MlxEngine: Engine trait implementation ──

pub struct MlxEngine {
    loaded: Option<LoadedModel>,
    tokenizer: Option<Tokenizer>,
    model_dir: Option<PathBuf>,
    repo_id: String,
}

// SAFETY: MlxEngine is only used from a single thread (the orchestrator task).
// MLX arrays are backed by Metal buffers which are thread-safe on Apple platforms.
unsafe impl Send for MlxEngine {}
unsafe impl Sync for MlxEngine {}

impl MlxEngine {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            loaded: None,
            tokenizer: None,
            model_dir: None,
            repo_id: String::new(),
        })
    }

    fn ensure_loaded(&mut self, repo_id: &str) -> Result<()> {
        if self.loaded.is_some() && self.repo_id == repo_id {
            return Ok(());
        }

        let model_dir = download_model(repo_id)?;
        let tokenizer = load_tokenizer(&model_dir)?;

        let loaded = if is_qwen35_config(&model_dir) {
            log::info!("detected Qwen3.5 hybrid architecture");
            let config = load_qwen35_config(&model_dir)?;
            let mut model = qwen35::model::Qwen35Model::new(config)?;
            qwen35::model::load_qwen35_weights(&mut model, &model_dir)?;
            let caches = model.init_caches();
            LoadedModel::Qwen35 { model, caches }
        } else {
            log::info!("detected Qwen2 basic architecture");
            let config = load_config(&model_dir)?;
            let mut model = QwenModel::new(config)?;
            let weight_files: Vec<_> = std::fs::read_dir(&model_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
                .collect();
            if weight_files.is_empty() {
                return Err(anyhow!("no .safetensors files found in {:?}", model_dir));
            }
            for shard in &weight_files {
                log::info!("loading weights from {:?}", shard.file_name().unwrap_or_default());
                model.load_safetensors(shard)
                    .map_err(|e| anyhow!("failed to load {:?}: {e}", shard))?;
            }
            LoadedModel::Qwen2(model)
        };

        self.loaded = Some(loaded);
        self.tokenizer = Some(tokenizer);
        self.model_dir = Some(model_dir);
        self.repo_id = repo_id.to_string();

        log::info!("MLX engine ready ({})", repo_id);
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        tx: &mut mpsc::Sender<Result<InferenceEvent>>,
    ) -> Result<()> {
        let loaded = self.loaded.as_mut().ok_or_else(|| anyhow!("model not loaded"))?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow!("tokenizer not loaded"))?;

        let encoding = tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;
        let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let seq_len = token_ids.len() as i32;

        log::info!("MLX infer: {} tokens prompt, max_tokens={}", seq_len, max_tokens);

        let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

        let mut input = Array::from_slice(&token_ids, &[1, seq_len]);
        let mut in_think_block = false;
        let mut token_buffer = String::new();

        let eos_token = tokenizer.token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .unwrap_or(2);

        // For Qwen2, we still use the old cache style
        let mut qwen2_caches: Vec<Option<(Array, Array)>> = match loaded {
            LoadedModel::Qwen2(m) => vec![None; m.config.num_hidden_layers],
            LoadedModel::Qwen35 { .. } => vec![],
        };

        for _step in 0..max_tokens {
            let next_token = match loaded {
                LoadedModel::Qwen2(model) => {
                    let (logits, new_caches) = model.forward(&input, &qwen2_caches)?;
                    qwen2_caches = new_caches.into_iter().map(Some).collect();
                    let last_logits = logits.index((.., -1, ..));
                    let arr = argmax_axis(&last_logits, -1, None).map_err(|e| anyhow!("{e}"))?;
                    arr.eval().map_err(|e| anyhow!("{e}"))?;
                    arr.as_slice::<i32>()[0]
                }
                LoadedModel::Qwen35 { model, caches } => {
                    model.next_token(&input, caches)?
                }
            };

            if next_token as u32 == eos_token {
                break;
            }

            let token_str = tokenizer
                .decode(&[next_token as u32], true)
                .unwrap_or_else(|_| "??".to_string());

            token_buffer.push_str(&token_str);

            // Handle <think> blocks
            if !in_think_block && token_buffer.contains("<think>") {
                in_think_block = true;
                let _ = futures::executor::block_on(
                    tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Start))),
                );
                token_buffer = token_buffer.replace("<think>", "");
            }

            if in_think_block {
                if token_buffer.contains("</think>") {
                    in_think_block = false;
                    let parts: Vec<&str> = token_buffer.split("</think>").collect();
                    if let Some(think_content) = parts.first() {
                        if !think_content.is_empty() {
                            let _ = futures::executor::block_on(tx.send(Ok(
                                InferenceEvent::Thought(ThoughtEvent::Delta(think_content.to_string())),
                            )));
                        }
                    }
                    let _ = futures::executor::block_on(
                        tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop))),
                    );
                    token_buffer = parts.get(1).unwrap_or(&"").to_string();
                } else if !token_buffer.is_empty() {
                    let _ = futures::executor::block_on(tx.send(Ok(
                        InferenceEvent::Thought(ThoughtEvent::Delta(token_buffer.clone())),
                    )));
                    token_buffer.clear();
                }
            }

            if !in_think_block && !token_buffer.is_empty() {
                let _ = futures::executor::block_on(
                    tx.send(Ok(InferenceEvent::Content(token_buffer.clone()))),
                );
                token_buffer.clear();
            }

            input = Array::from_slice(&[next_token], &[1, 1]);
        }

        let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        Ok(())
    }
}

#[async_trait]
impl Engine for MlxEngine {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        self.repo_id = model_path.to_string();
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.loaded = None;
        self.tokenizer = None;
        self.model_dir = None;
        self.repo_id.clear();
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded.is_some()
    }

    fn default_model(&self) -> String {
        MLX_DEFAULT_MODEL.to_string()
    }

    async fn preload_model(&mut self, model_path: &str, _purpose: &str) -> Result<()> {
        self.ensure_loaded(model_path)
    }

    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let repo_id = model
            .unwrap_or(MLX_DEFAULT_MODEL)
            .to_string();

        self.ensure_loaded(&repo_id)?;

        let (mut tx, rx) = mpsc::channel(100);
        let max_tokens = config.max_tokens.unwrap_or(512);

        if let Err(e) = self.generate(prompt, max_tokens, &mut tx) {
            let _ = futures::executor::block_on(tx.send(Err(e)));
        }

        Ok(rx)
    }

    async fn embed(
        &mut self,
        _model: Option<&str>,
        _input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        Err(anyhow!("MLX engine does not support embedding — use DispatchEngine to route to llama.cpp"))
    }
}
