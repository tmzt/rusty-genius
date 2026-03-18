//! SwiGLU MLP shared by both full-attention and linear-attention layers.

use anyhow::{anyhow, Result};
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::{nn, ops, Array};

use super::{make_quantized_linear, Qwen35Config};

#[derive(ModuleParameters)]
pub struct Qwen35MLP {
    #[param]
    pub gate_proj: nn::QuantizedLinear,
    #[param]
    pub up_proj: nn::QuantizedLinear,
    #[param]
    pub down_proj: nn::QuantizedLinear,
}

impl Qwen35MLP {
    pub fn new(config: &Qwen35Config) -> Result<Self> {
        let h = config.hidden_size as i32;
        let i = config.intermediate_size as i32;
        Ok(Self {
            gate_proj: make_quantized_linear(h, i)?,
            up_proj: make_quantized_linear(h, i)?,
            down_proj: make_quantized_linear(i, h)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = self.gate_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        // SiLU = x * sigmoid(x)
        let gate = gate.multiply(&ops::sigmoid(&gate).map_err(|e| anyhow!("{e}"))?)?;
        let up = self.up_proj.forward(x).map_err(|e| anyhow!("{e}"))?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden).map_err(|e| anyhow!("{e}"))
    }
}
