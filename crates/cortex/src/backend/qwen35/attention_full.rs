//! Gated full attention with output gate (Qwen3.5 full-attention layers).
//!
//! q_proj outputs 2× for [Q, gate]. After SDPA, output *= sigmoid(gate).

use anyhow::{anyhow, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::{nn, ops, Array};
use mlx_rs::ops::indexing::IndexOp;

use super::cache::KvCache;
use super::rope::PartialRotaryEmbedding;
use super::{make_quantized_linear, make_rms_norm, Qwen35Config};

#[derive(ModuleParameters)]
pub struct FullAttention {
    #[param]
    pub q_proj: nn::QuantizedLinear,
    #[param]
    pub k_proj: nn::QuantizedLinear,
    #[param]
    pub v_proj: nn::QuantizedLinear,
    #[param]
    pub o_proj: nn::QuantizedLinear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: PartialRotaryEmbedding,
}

impl FullAttention {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        let h = config.hidden_size as i32;
        let hd = config.head_dim;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        // q_proj outputs 2× head_dim for [Q, gate]
        let q_out = (nh * hd * 2) as i32;
        let kv_out = (nkv * hd) as i32;
        let q_in = (nh * hd) as i32;
        let eps = config.rms_norm_eps as f32;
        let rotary_dim = config.rotary_dim();

        Ok(Self {
            q_proj: make_quantized_linear(h, q_out)?,
            k_proj: make_quantized_linear(h, kv_out)?,
            v_proj: make_quantized_linear(h, kv_out)?,
            o_proj: make_quantized_linear(q_in, h)?,
            q_norm: make_rms_norm(hd as i32, eps)?,
            k_norm: make_rms_norm(hd as i32, eps)?,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            rope: PartialRotaryEmbedding::new(hd, rotary_dim, config.rope_parameters.rope_theta)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KvCache,
    ) -> Result<Array> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];
        let hd = self.head_dim as i32;
        let nh = self.num_heads as i32;
        let nkv = self.num_kv_heads as i32;
        let q_total = nh * hd; // half of q_proj output

        // Project
        let qg = self.q_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let k = self.k_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let v = self.v_proj.forward(x).map_err(|e| anyhow!("{e}"))?;

        // Split q_proj output into q and gate
        let q = qg.index((.., .., ..q_total));
        let gate = qg.index((.., .., q_total..));

        // Reshape to [batch, seq, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, nh, hd])?;
        let gate = gate.reshape(&[batch, seq_len, nh, hd])?;
        let k = k.reshape(&[batch, seq_len, nkv, hd])?;
        let v = v.reshape(&[batch, seq_len, nkv, hd])?;

        // QK norms
        let q = self.q_norm.forward(&q).map_err(|e| anyhow!("{e}"))?;
        let k = self.k_norm.forward(&k).map_err(|e| anyhow!("{e}"))?;

        // Partial RoPE
        let offset = cache.offset();
        let q = self.rope.apply(&q, offset)?;
        let k = self.rope.apply(&k, offset)?;

        // Concatenate with KV cache
        let (k, v) = if let Some(ck) = &cache.k {
            let cv = cache.v.as_ref().unwrap();
            (
                ops::concatenate_axis(&[ck, &k], 1).map_err(|e| anyhow!("{e}"))?,
                ops::concatenate_axis(&[cv, &v], 1).map_err(|e| anyhow!("{e}"))?,
            )
        } else {
            (k, v)
        };
        cache.k = Some(k.clone());
        cache.v = Some(v.clone());

        // Transpose to [batch, heads, seq, dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // GQA: repeat KV heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeats = (self.num_heads / self.num_kv_heads) as i32;
            let kv_seq = k.shape()[2];
            let expand_k = k.reshape(&[batch, nkv, 1, kv_seq, hd])?;
            let expand_k =
                ops::broadcast_to(&expand_k, &[batch, nkv, repeats, kv_seq, hd])
                    .map_err(|e| anyhow!("{e}"))?;
            let k = expand_k.reshape(&[batch, nh, kv_seq, hd])?;

            let expand_v = v.reshape(&[batch, nkv, 1, kv_seq, hd])?;
            let expand_v =
                ops::broadcast_to(&expand_v, &[batch, nkv, repeats, kv_seq, hd])
                    .map_err(|e| anyhow!("{e}"))?;
            let v = expand_v.reshape(&[batch, nh, kv_seq, hd])?;
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

        let weights =
            ops::softmax_axis(&scores, -1, None).map_err(|e| anyhow!("{e}"))?;
        let attn_out = weights.matmul(&v)?;

        // Transpose back to [batch, seq, heads, dim]
        let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3])?;

        // Output gate: sigmoid(gate) * attn_out
        let gate_sig = ops::sigmoid(&gate).map_err(|e| anyhow!("{e}"))?;
        let attn_out = attn_out.multiply(&gate_sig)?;

        // Reshape and project
        let attn_out = attn_out.reshape(&[batch, seq_len, nh * hd])?;
        self.o_proj
            .forward(&attn_out)
            .map_err(|e| anyhow!("{e}"))
    }
}
