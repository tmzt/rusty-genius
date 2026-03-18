//! Qwen3.5 transformer layers — both full-attention and linear-attention variants.
//!
//! Both share: RmsNorm → Attention → RmsNorm → SwiGLU MLP (residual connections).
//! Uses a macro to generate identical struct layouts so safetensors key paths align.

use anyhow::{anyhow, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
#[allow(unused_imports)]
use mlx_rs::module::ModuleParameters;
use mlx_rs::{nn, Array};

use super::attention_full::FullAttention;
use super::attention_linear::LinearAttention;
use super::cache::LayerCache;
use super::mlp::Qwen35MLP;
use super::{make_rms_norm, Qwen35Config};

// ── Macro-generated layer structs ──

macro_rules! qwen35_layer {
    ($name:ident, $attn_ty:ty) => {
        #[derive(ModuleParameters)]
        pub struct $name {
            #[param]
            pub self_attn: $attn_ty,
            #[param]
            pub mlp: Qwen35MLP,
            #[param]
            pub input_layernorm: nn::RmsNorm,
            #[param]
            pub post_attention_layernorm: nn::RmsNorm,
        }
    };
}

qwen35_layer!(FullAttnLayer, FullAttention);
qwen35_layer!(LinearAttnLayer, LinearAttention);

impl FullAttnLayer {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        let eps = config.rms_norm_eps as f32;
        let h = config.hidden_size as i32;
        Ok(Self {
            self_attn: FullAttention::new(config)?,
            mlp: Qwen35MLP::new(config)?,
            input_layernorm: make_rms_norm(h, eps)?,
            post_attention_layernorm: make_rms_norm(h, eps)?,
        })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array> {
        let kv = cache.as_full_mut().expect("FullAttnLayer needs Full cache");
        let residual = x.clone();
        let hidden = self
            .input_layernorm
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?;
        let hidden = self.self_attn.forward(&hidden, mask, kv)?;
        let hidden = residual.add(&hidden)?;

        let residual = hidden.clone();
        let hidden = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| anyhow!("{e}"))?;
        let hidden = self.mlp.forward(&hidden)?;
        hidden.add(&residual).map_err(|e| anyhow!("residual: {e}"))
    }
}

impl LinearAttnLayer {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        let eps = config.rms_norm_eps as f32;
        let h = config.hidden_size as i32;
        Ok(Self {
            self_attn: LinearAttention::new(config)?,
            mlp: Qwen35MLP::new(config)?,
            input_layernorm: make_rms_norm(h, eps)?,
            post_attention_layernorm: make_rms_norm(h, eps)?,
        })
    }

    pub fn forward(&mut self, x: &Array, cache: &mut LayerCache) -> Result<Array> {
        let lc = cache
            .as_linear_mut()
            .expect("LinearAttnLayer needs Linear cache");
        let residual = x.clone();
        let hidden = self
            .input_layernorm
            .forward(x)
            .map_err(|e| anyhow!("{e}"))?;
        let hidden = self.self_attn.forward(&hidden, lc)?;
        let hidden = residual.add(&hidden)?;

        let residual = hidden.clone();
        let hidden = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| anyhow!("{e}"))?;
        let hidden = self.mlp.forward(&hidden)?;
        hidden.add(&residual).map_err(|e| anyhow!("residual: {e}"))
    }
}

// ── Layer enum with manual ModuleParameters ──

pub enum Qwen35Layer {
    Full(FullAttnLayer),
    Linear(LinearAttnLayer),
}

impl Qwen35Layer {
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>, cache: &mut LayerCache) -> Result<Array> {
        match self {
            Self::Full(l) => l.forward(x, mask, cache),
            Self::Linear(l) => l.forward(x, cache),
        }
    }
}

impl mlx_rs::module::ModuleParameters for Qwen35Layer {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Full(l) => l.num_parameters(),
            Self::Linear(l) => l.num_parameters(),
        }
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Full(l) => l.parameters(),
            Self::Linear(l) => l.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            Self::Full(l) => l.parameters_mut(),
            Self::Linear(l) => l.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            Self::Full(l) => l.trainable_parameters(),
            Self::Linear(l) => l.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Full(l) => l.freeze_parameters(recursive),
            Self::Linear(l) => l.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Full(l) => l.unfreeze_parameters(recursive),
            Self::Linear(l) => l.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Full(l) => l.all_frozen(),
            Self::Linear(l) => l.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Full(l) => l.any_frozen(),
            Self::Linear(l) => l.any_frozen(),
        }
    }
}
