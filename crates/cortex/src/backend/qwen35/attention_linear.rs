//! Gated DeltaNet linear attention (Qwen3.5 linear-attention layers).
//!
//! Fixed-size recurrent state instead of growing KV cache.
//! Forward pass: project → conv1d → delta rule recurrence → gated-norm output.

use anyhow::{anyhow, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{nn, Array};

use super::cache::LinearCache;
use super::{make_depthwise_conv1d, make_quantized_linear, Qwen35Config};

// ── RmsNormGated: rms_norm(x) * silu(gate) ──

#[derive(ModuleParameters)]
pub struct RmsNormGated {
    #[param]
    pub weight: Param<Array>,
    eps: f32,
}

impl RmsNormGated {
    pub fn new(dims: i32, eps: f32) -> Result<Self> {
        let weight =
            mlx_rs::ops::ones::<f32>(&[dims]).map_err(|e| anyhow!("RmsNormGated: {e}"))?;
        Ok(Self {
            weight: Param::new(weight),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, gate: &Array) -> Result<Array> {
        let normed = mlx_rs::fast::rms_norm(x, &self.weight, self.eps)
            .map_err(|e| anyhow!("rms_norm: {e}"))?;
        // silu(gate) = gate * sigmoid(gate)
        let gate_act = gate.multiply(&ops::sigmoid(gate).map_err(|e| anyhow!("{e}"))?)?;
        normed
            .multiply(&gate_act)
            .map_err(|e| anyhow!("gated mul: {e}"))
    }
}

// ── LinearAttention (Gated DeltaNet) ──

#[derive(ModuleParameters)]
#[allow(non_snake_case)]
pub struct LinearAttention {
    #[param]
    pub in_proj_qkv: nn::QuantizedLinear,
    #[param]
    pub in_proj_z: nn::QuantizedLinear,
    #[param]
    pub in_proj_b: nn::QuantizedLinear,
    #[param]
    pub in_proj_a: nn::QuantizedLinear,
    #[param]
    pub conv1d: nn::Conv1d,
    #[param]
    pub A_log: Param<Array>,
    #[param]
    pub dt_bias: Param<Array>,
    #[param]
    pub norm: RmsNormGated,
    #[param]
    pub out_proj: nn::QuantizedLinear,

    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    #[allow(dead_code)]
    conv_dim: usize,
    conv_kernel: usize,
}

impl LinearAttention {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        let h = config.hidden_size as i32;
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();
        let conv_dim = config.conv_dim();
        let nv = config.linear_num_value_heads;
        let eps = config.rms_norm_eps as f32;

        // in_proj_qkv: hidden → key_dim*2 + value_dim
        let qkv_out = (key_dim * 2 + value_dim) as i32;
        // A_log, dt_bias: learnable per-head scalars
        let a_log_data = vec![0.0f32; nv];
        let dt_bias_data = vec![1.0f32; nv];

        Ok(Self {
            in_proj_qkv: make_quantized_linear(h, qkv_out)?,
            in_proj_z: make_quantized_linear(h, value_dim as i32)?,
            in_proj_b: make_quantized_linear(h, nv as i32)?,
            in_proj_a: make_quantized_linear(h, nv as i32)?,
            conv1d: make_depthwise_conv1d(conv_dim as i32, config.linear_conv_kernel_dim as i32)?,
            A_log: Param::new(Array::from_slice(&a_log_data, &[nv as i32])),
            dt_bias: Param::new(Array::from_slice(&dt_bias_data, &[nv as i32])),
            norm: RmsNormGated::new(config.linear_value_head_dim as i32, eps)?,
            out_proj: make_quantized_linear(value_dim as i32, h)?,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: nv,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel: config.linear_conv_kernel_dim,
        })
    }

    pub fn forward(&mut self, x: &Array, cache: &mut LinearCache) -> Result<Array> {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1] as usize;

        // ── Projections ──
        let qkv = self
            .in_proj_qkv
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?; // [B, S, qkv_dim]
        let z = self
            .in_proj_z
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?; // [B, S, value_dim]
        let b = self
            .in_proj_b
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?; // [B, S, num_v_heads]
        let a = self
            .in_proj_a
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?; // [B, S, num_v_heads]

        // ── Causal Conv1d ──
        // Input is NLC [batch, seq, conv_dim]. Left-pad for causal.
        let qkv_conv = self.causal_conv1d(&qkv, cache)?;

        // SiLU activation after conv
        let qkv_conv = qkv_conv.multiply(
            &ops::sigmoid(&qkv_conv).map_err(|e| anyhow!("{e}"))?,
        )?;

        // ── Split into Q, K, V ──
        let kd = self.key_dim as i32;
        let vd = self.value_dim as i32;
        let q = qkv_conv.index((.., .., ..kd));
        let k = qkv_conv.index((.., .., kd..kd * 2));
        let v = qkv_conv.index((.., .., kd * 2..));

        let nk = self.num_k_heads as i32;
        let nv = self.num_v_heads as i32;
        let hkd = self.head_k_dim as i32;
        let hvd = self.head_v_dim as i32;
        let sl = seq_len as i32;

        let q = q.reshape(&[batch, sl, nk, hkd])?;
        let k = k.reshape(&[batch, sl, nk, hkd])?;
        let v = v.reshape(&[batch, sl, nv, hvd])?;
        let z = z.reshape(&[batch, sl, nv, hvd])?;

        // ── Decay factor ──
        // g = -exp(A_log) * softplus(a + dt_bias)
        let a_exp = self.A_log.exp().map_err(|e| anyhow!("{e}"))?;
        let a_neg = a_exp.negative()?;
        let a_plus_bias = a.add(&self.dt_bias)?; // [B, S, nv]
        // softplus(x) = log(1 + exp(x))
        let sp = a_plus_bias
            .exp()
            .map_err(|e| anyhow!("{e}"))?
            .add(&Array::from_slice(&[1.0f32], &[1]))?
            .log()
            .map_err(|e| anyhow!("{e}"))?;
        let g = a_neg.multiply(&sp)?; // [B, S, nv]

        // Beta (delta coefficient)
        let beta = ops::sigmoid(&b).map_err(|e| anyhow!("{e}"))?; // [B, S, nv]

        // ── GQA expansion if needed ──
        let (q, k) = if self.num_v_heads > self.num_k_heads {
            let rep = (self.num_v_heads / self.num_k_heads) as i32;
            let q = q.reshape(&[batch, sl, nk, 1, hkd])?;
            let q = ops::broadcast_to(&q, &[batch, sl, nk, rep, hkd])
                .map_err(|e| anyhow!("{e}"))?;
            let q = q.reshape(&[batch, sl, nv, hkd])?;
            let k = k.reshape(&[batch, sl, nk, 1, hkd])?;
            let k = ops::broadcast_to(&k, &[batch, sl, nk, rep, hkd])
                .map_err(|e| anyhow!("{e}"))?;
            let k = k.reshape(&[batch, sl, nv, hkd])?;
            (q, k)
        } else {
            (q, k)
        };

        // ── L2 normalize Q, K ──
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;

        // ── Recurrent delta rule (per-token) ──
        let output = self.recurrent_forward(&q, &k, &v, &g, &beta, cache)?;

        // ── Gated RMS norm + output projection ──
        // output: [B, S, nv, hvd], z: [B, S, nv, hvd]
        let output = self.norm.forward(&output, &z)?;
        let output = output.reshape(&[batch, sl, vd])?;
        self.out_proj
            .forward(&output)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Causal conv1d: left-pad input, apply depthwise conv.
    fn causal_conv1d(&mut self, x: &Array, cache: &mut LinearCache) -> Result<Array> {
        let pad_len = (self.conv_kernel - 1) as i32;
        let seq_len = x.shape()[1];

        // If we have cached conv state, prepend it
        let x_padded = if let Some(ref cs) = cache.conv_state {
            ops::concatenate_axis(&[cs, x], 1).map_err(|e| anyhow!("{e}"))?
        } else {
            // Zero-pad on the left
            let batch = x.shape()[0];
            let channels = x.shape()[2];
            let pad = mlx_rs::ops::zeros::<f32>(&[batch, pad_len, channels])
                .map_err(|e| anyhow!("{e}"))?;
            ops::concatenate_axis(&[&pad, x], 1).map_err(|e| anyhow!("{e}"))?
        };

        // Save last (kernel-1) timesteps as conv state
        let total = x_padded.shape()[1];
        cache.conv_state = Some(x_padded.index((.., (total - pad_len).., ..)));

        // Apply conv (padding=0, so output length = total - kernel + 1 = seq_len)
        let out = self
            .conv1d
            .forward(&x_padded)
            .map_err(|e| anyhow!("conv1d: {e}"))?;

        // Slice to seq_len (in case of rounding)
        if out.shape()[1] != seq_len {
            Ok(out.index((.., ..seq_len, ..)))
        } else {
            Ok(out)
        }
    }

    /// Per-token recurrent delta rule.
    fn recurrent_forward(
        &self,
        q: &Array,   // [B, S, nv, hkd]
        k: &Array,   // [B, S, nv, hkd]
        v: &Array,   // [B, S, nv, hvd]
        g: &Array,   // [B, S, nv]
        beta: &Array, // [B, S, nv]
        cache: &mut LinearCache,
    ) -> Result<Array> {
        let batch = q.shape()[0];
        let seq_len = q.shape()[1] as usize;
        let nv = self.num_v_heads as i32;
        let hkd = self.head_k_dim as i32;
        let hvd = self.head_v_dim as i32;

        // Initialize state: [B, nv, hkd, hvd]
        let mut state = if let Some(ref s) = cache.state {
            s.clone()
        } else {
            mlx_rs::ops::zeros::<f32>(&[batch, nv, hkd, hvd])
                .map_err(|e| anyhow!("state init: {e}"))?
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let ti = t as i32;
            // Extract timestep t: squeeze seq dim
            let q_t = q.index((.., ti..ti + 1, .., ..)).squeeze_axes(&[1])?; // [B, nv, hkd]
            let k_t = k.index((.., ti..ti + 1, .., ..)).squeeze_axes(&[1])?;
            let v_t = v.index((.., ti..ti + 1, .., ..)).squeeze_axes(&[1])?; // [B, nv, hvd]
            let g_t = g.index((.., ti..ti + 1, ..)).squeeze_axes(&[1])?; // [B, nv]
            let beta_t = beta.index((.., ti..ti + 1, ..)).squeeze_axes(&[1])?; // [B, nv]

            // Decay: exp(g_t) broadcast to state shape [B, nv, 1, 1]
            let decay = g_t
                .exp()
                .map_err(|e| anyhow!("{e}"))?
                .reshape(&[batch, nv, 1, 1])?;
            state = state.multiply(&decay)?;

            // kv_mem = einsum("bnkd,bnk -> bnd", state, k_t) = sum(state * k_t[...,None], dim=-2)
            let k_expanded = k_t.reshape(&[batch, nv, hkd, 1])?; // [B, nv, hkd, 1]
            let kv_mem = state
                .multiply(&k_expanded)?
                .sum_axis(-2, None)
                .map_err(|e| anyhow!("{e}"))?; // [B, nv, hvd]

            // delta = (v_t - kv_mem) * beta_t
            let beta_expanded = beta_t.reshape(&[batch, nv, 1])?; // [B, nv, 1]
            let delta = v_t.subtract(&kv_mem)?.multiply(&beta_expanded)?; // [B, nv, hvd]

            // state += outer(k_t, delta): k_t[...,None] * delta[...,None,:]
            let k_col = k_t.reshape(&[batch, nv, hkd, 1])?; // [B, nv, hkd, 1]
            let d_row = delta.reshape(&[batch, nv, 1, hvd])?; // [B, nv, 1, hvd]
            state = state.add(&k_col.multiply(&d_row)?)?;

            // output_t = einsum("bnkd,bnk -> bnd", state, q_t) = sum(state * q_t[...,None], dim=-2)
            let q_expanded = q_t.reshape(&[batch, nv, hkd, 1])?;
            let out_t = state
                .multiply(&q_expanded)?
                .sum_axis(-2, None)
                .map_err(|e| anyhow!("{e}"))?; // [B, nv, hvd]

            outputs.push(out_t);
        }

        cache.state = Some(state);

        // Stack outputs: [B, S, nv, hvd]
        let output_refs: Vec<&Array> = outputs.iter().collect();
        let stacked = ops::stack_axis(&output_refs, 1).map_err(|e| anyhow!("stack: {e}"))?;
        Ok(stacked)
    }
}

/// L2 normalize along last dimension.
fn l2_normalize(x: &Array) -> Result<Array> {
    let sq = x.multiply(x)?;
    let sum_sq = sq
        .sum_axis(-1, Some(true))
        .map_err(|e| anyhow!("sum: {e}"))?;
    let eps = Array::from_slice(&[1e-12f32], &[1]);
    let norm = sum_sq
        .add(&eps)?
        .sqrt()
        .map_err(|e| anyhow!("sqrt: {e}"))?;
    x.divide(&norm).map_err(|e| anyhow!("div: {e}"))
}
