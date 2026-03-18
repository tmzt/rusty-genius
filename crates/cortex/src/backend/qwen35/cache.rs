//! Per-layer caches for Qwen3.5 hybrid architecture.

use mlx_rs::Array;

/// KV cache for full-attention layers.
pub struct KvCache {
    pub k: Option<Array>,
    pub v: Option<Array>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn offset(&self) -> usize {
        self.k
            .as_ref()
            .map(|k| k.shape()[1] as usize)
            .unwrap_or(0)
    }
}

/// Recurrent + conv state for linear-attention layers.
pub struct LinearCache {
    /// Recurrent state: [batch, num_v_heads, head_k_dim, head_v_dim]
    pub state: Option<Array>,
    /// Conv1d state: [batch, kernel_size - 1, conv_dim] (NLC)
    pub conv_state: Option<Array>,
}

impl LinearCache {
    pub fn new() -> Self {
        Self {
            state: None,
            conv_state: None,
        }
    }
}

/// Per-layer cache enum.
pub enum LayerCache {
    Full(KvCache),
    Linear(LinearCache),
}

impl LayerCache {
    pub fn full() -> Self {
        Self::Full(KvCache::new())
    }

    pub fn linear() -> Self {
        Self::Linear(LinearCache::new())
    }

    pub fn as_full_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            Self::Full(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_linear_mut(&mut self) -> Option<&mut LinearCache> {
        match self {
            Self::Linear(c) => Some(c),
            _ => None,
        }
    }
}
