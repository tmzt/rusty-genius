//! Partial rotary position embedding — applies RoPE to a fraction of head dims.

use anyhow::{anyhow, Result};
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

pub struct PartialRotaryEmbedding {
    inv_freq: Array,
    rotary_dim: usize,
}

impl PartialRotaryEmbedding {
    pub fn new(_head_dim: usize, rotary_dim: usize, theta: f64) -> Result<Self> {
        let half_rot = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / rotary_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Array::from_slice(&inv_freq, &[half_rot as i32]),
            rotary_dim,
        })
    }

    /// Apply partial RoPE to x: [batch, seq, heads, head_dim].
    /// Rotates first `rotary_dim` dims, passes through the rest.
    pub fn apply(&self, x: &Array, offset: usize) -> Result<Array> {
        let seq_len = x.shape()[1] as usize;
        let head_dim = *x.shape().last().unwrap();
        let rot_dim = self.rotary_dim as i32;

        if rot_dim == 0 || rot_dim > head_dim {
            return Ok(x.clone());
        }

        let half = rot_dim / 2;
        let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
        let t = Array::from_slice(&positions, &[seq_len as i32]);

        let freqs = ops::outer(&t, &self.inv_freq).map_err(|e| anyhow!("outer: {e}"))?;
        let cos_f = freqs.cos().map_err(|e| anyhow!("cos: {e}"))?;
        let sin_f = freqs.sin().map_err(|e| anyhow!("sin: {e}"))?;

        // [1, seq_len, 1, half_rot]
        let cos_f = cos_f.reshape(&[1, seq_len as i32, 1, half])?;
        let sin_f = sin_f.reshape(&[1, seq_len as i32, 1, half])?;

        // Split into rotary and passthrough
        let x_rot = x.index((.., .., .., ..rot_dim));
        let x_pass = x.index((.., .., .., rot_dim..));

        // Split rotary part into halves
        let batch = x.shape()[0];
        let n_heads = x.shape()[2];
        let x_flat = x_rot.reshape(&[-1, rot_dim])?;
        let x1_flat = x_flat.index((.., ..half));
        let x2_flat = x_flat.index((.., half..));
        let half_shape = &[batch, seq_len as i32, n_heads, half];
        let x1 = x1_flat.reshape(half_shape)?;
        let x2 = x2_flat.reshape(half_shape)?;

        let rotated = ops::concatenate_axis(
            &[
                &x1.multiply(&cos_f)?.subtract(&x2.multiply(&sin_f)?)?,
                &x2.multiply(&cos_f)?.add(&x1.multiply(&sin_f)?)?,
            ],
            -1,
        )
        .map_err(|e| anyhow!("concat rot: {e}"))?;

        ops::concatenate_axis(&[&rotated, &x_pass], -1).map_err(|e| anyhow!("concat pass: {e}"))
    }
}
