//! Qwen3.5 top-level model: embedding → 32 hybrid layers → norm → lm_head.
//! Includes weight loading with key remapping for quantized safetensors.

use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;

use anyhow::{anyhow, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
#[allow(unused_imports)]
use mlx_rs::module::ModuleParametersExt;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};
use mlx_rs::{nn, Array};

use super::cache::LayerCache;
use super::layer::{FullAttnLayer, LinearAttnLayer, Qwen35Layer};
use super::{make_quantized_linear, make_quantized_embedding, make_rms_norm, Qwen35Config};

// ── Model ──

#[derive(ModuleParameters)]
pub struct Qwen35Model {
    #[param]
    pub embed_tokens: nn::QuantizedEmbedding,
    #[param]
    pub layers: Vec<Qwen35Layer>,
    #[param]
    pub norm: nn::RmsNorm,
    #[param]
    pub lm_head: nn::QuantizedLinear,
    pub config: Qwen35Config,
}

impl Qwen35Model {
    pub fn new(config: Qwen35Config) -> Result<Self> {
        let h = config.hidden_size as i32;
        let v = config.vocab_size as i32;
        let eps = config.rms_norm_eps as f32;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_type = &config.layer_types[i];
            match layer_type.as_str() {
                "full_attention" => layers.push(Qwen35Layer::Full(FullAttnLayer::new(&config)?)),
                "linear_attention" => {
                    layers.push(Qwen35Layer::Linear(LinearAttnLayer::new(&config)?))
                }
                other => return Err(anyhow!("unknown layer_type: {other}")),
            }
        }

        Ok(Self {
            embed_tokens: make_quantized_embedding(v, h)?,
            layers,
            norm: make_rms_norm(h, eps)?,
            lm_head: make_quantized_linear(h, v)?,
            config,
        })
    }

    /// Build initial caches (one per layer, matching layer type).
    pub fn init_caches(&self) -> Vec<LayerCache> {
        self.config
            .layer_types
            .iter()
            .map(|lt| match lt.as_str() {
                "full_attention" => LayerCache::full(),
                _ => LayerCache::linear(),
            })
            .collect()
    }

    /// Forward pass. Returns logits [batch, seq, vocab].
    pub fn forward(
        &mut self,
        input_ids: &Array,
        caches: &mut [LayerCache],
    ) -> Result<Array> {
        let mut hidden = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| anyhow!("{e}"))?;

        let seq_len = input_ids.shape()[1];

        // Causal mask (only needed for full-attention layers, and only for seq > 1)
        let mask = if seq_len > 1 {
            // Determine past_len from first full-attention cache
            let past_len = caches
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(kv) => Some(kv.offset()),
                    _ => None,
                })
                .unwrap_or(0) as i32;

            let total_len = past_len + seq_len;
            let mut mask_data = vec![0.0f32; (seq_len * total_len) as usize];
            for i in 0..seq_len as usize {
                for j in (past_len as usize + i + 1)..total_len as usize {
                    mask_data[i * total_len as usize + j] = f32::NEG_INFINITY;
                }
            }
            Some(Array::from_slice(
                &mask_data,
                &[1, 1, seq_len, total_len],
            ))
        } else {
            None
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer.forward(&hidden, mask.as_ref(), &mut caches[i])?;
        }

        let hidden = self.norm.forward(&hidden).map_err(|e| anyhow!("{e}"))?;
        self.lm_head.forward(&hidden).map_err(|e| anyhow!("{e}"))
    }
}

// ── Weight loading with key remapping ──

/// Load quantized safetensors weights into a Qwen35Model.
///
/// Handles two key transforms:
/// 1. Strip common prefix (`language_model.model.`, `model.`, etc.)
/// 2. Remap `*.weight` → `*.inner.weight` for QuantizedLinear/QuantizedEmbedding
///    (detected by presence of a sibling `*.scales` key).
pub fn load_qwen35_weights(model: &mut Qwen35Model, model_dir: &Path) -> Result<()> {
    let weight_files: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    if weight_files.is_empty() {
        return Err(anyhow!(
            "no .safetensors files found in {:?}",
            model_dir
        ));
    }

    // Load all shards
    let mut all_weights: Vec<(String, Array)> = Vec::new();
    for shard in &weight_files {
        log::info!(
            "loading weights from {:?}",
            shard.file_name().unwrap_or_default()
        );
        let loaded = Array::load_safetensors(shard)
            .map_err(|e| anyhow!("load {:?}: {e}", shard))?;
        for (k, v) in loaded {
            all_weights.push((k.to_string(), v));
        }
    }

    log::info!("loaded {} weight tensors total", all_weights.len());

    // Strip prefix and build key set for quantized detection
    let stripped: Vec<(String, Array)> = all_weights
        .into_iter()
        .map(|(k, v)| (strip_prefix(&k).to_string(), v))
        .collect();

    let key_set: HashSet<String> = stripped.iter().map(|(k, _)| k.clone()).collect();

    // Remap quantized keys and collect
    let remapped: Vec<(Rc<str>, Array)> = stripped
        .into_iter()
        .map(|(k, v)| {
            let remapped = remap_quantized_key(&k, &key_set);
            (Rc::from(remapped.as_str()), v)
        })
        .collect();

    let n = remapped.len();
    model.update_flattened(remapped.into_iter().collect());
    log::info!("applied {} weight tensors to model", n);

    Ok(())
}

/// Strip common HuggingFace / MLX community prefixes.
fn strip_prefix(key: &str) -> &str {
    // Order matters: check longer prefix first
    if let Some(rest) = key.strip_prefix("language_model.model.") {
        rest
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        rest
    } else if let Some(rest) = key.strip_prefix("model.") {
        rest
    } else {
        key
    }
}

/// For QuantizedLinear/QuantizedEmbedding, safetensors stores `weight`
/// but the mlx-rs struct flattens to `inner.weight`. Detect by checking
/// whether a sibling `scales` key exists at the same prefix.
fn remap_quantized_key(key: &str, all_keys: &HashSet<String>) -> String {
    if let Some(prefix) = key.strip_suffix(".weight") {
        let scales_key = format!("{prefix}.scales");
        if all_keys.contains(&scales_key) {
            return format!("{prefix}.inner.weight");
        }
    }
    key.to_string()
}

impl Qwen35Model {
    /// Greedy next-token prediction (for use by MlxEngine::generate).
    pub fn next_token(
        &mut self,
        input: &Array,
        caches: &mut [LayerCache],
    ) -> Result<i32> {
        let logits = self.forward(input, caches)?;
        let last_logits = logits.index((.., -1, ..));
        let next_token = argmax_axis(&last_logits, -1, None)
            .map_err(|e| anyhow!("argmax: {e}"))?;
        next_token.eval().map_err(|e| anyhow!("eval: {e}"))?;
        Ok(next_token.as_slice::<i32>()[0])
    }
}
