//! Qwen3.5 hybrid architecture: Gated DeltaNet (linear) + Gated full attention.

pub mod attention_full;
pub mod attention_linear;
pub mod cache;
pub mod layer;
pub mod mlp;
pub mod model;
pub mod rope;

use anyhow::{anyhow, Result};
use mlx_rs::builder::Builder;
use mlx_rs::module::Param;
use mlx_rs::nn;
use serde::Deserialize;

// ── Config ──

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Qwen35ConfigOuter {
    pub text_config: Qwen35Config,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub attn_output_gate: bool,
}

#[derive(Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
}

impl Qwen35Config {
    pub fn key_dim(&self) -> usize {
        self.linear_key_head_dim * self.linear_num_key_heads
    }

    pub fn value_dim(&self) -> usize {
        self.linear_value_head_dim * self.linear_num_value_heads
    }

    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }
}

// ── Helpers ──

pub fn make_quantized_linear(in_dim: i32, out_dim: i32) -> Result<nn::QuantizedLinear> {
    nn::QuantizedLinearBuilder::new(in_dim, out_dim)
        .bias(false)
        .build()
        .map_err(|e| anyhow!("QuantizedLinear({in_dim},{out_dim}): {e}"))
}

pub fn make_rms_norm(dims: i32, eps: f32) -> Result<nn::RmsNorm> {
    let weight = mlx_rs::ops::ones::<f32>(&[dims])
        .map_err(|e| anyhow!("RmsNorm weight: {e}"))?;
    Ok(nn::RmsNorm {
        weight: Param::new(weight),
        eps,
    })
}

pub fn make_quantized_embedding(vocab: i32, dims: i32) -> Result<nn::QuantizedEmbedding> {
    nn::QuantizedEmbeddingBuilder::new(vocab, dims)
        .build()
        .map_err(|e| anyhow!("QuantizedEmbedding: {e}"))
}

/// Depthwise Conv1d (groups == channels, no bias) for linear attention.
pub fn make_depthwise_conv1d(channels: i32, kernel_size: i32) -> Result<nn::Conv1d> {
    let scale = f32::sqrt(1.0 / kernel_size as f32);
    let weight = mlx_rs::random::uniform::<_, f32>(
        -scale,
        scale,
        &[channels, kernel_size, 1], // MLX: [out, kernel, in/groups]
        None,
    )
    .map_err(|e| anyhow!("Conv1d weight init: {e}"))?;
    Ok(nn::Conv1d {
        weight: Param::new(weight),
        bias: Param::new(None),
        stride: 1,
        padding: 0, // causal padding handled manually
        dilation: 1,
        groups: channels,
    })
}
