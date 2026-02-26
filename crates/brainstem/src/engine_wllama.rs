use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;

/// Skeleton wllama engine for WASM builds.
/// Will use wasm-bindgen FFI to call into wllama JS at runtime.
pub struct WllamaEngine {
    loaded: bool,
}

impl WllamaEngine {
    pub fn new() -> Self {
        Self { loaded: false }
    }
}

#[async_trait]
impl Engine for WllamaEngine {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        Err(anyhow!("wllama engine not yet connected"))
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn default_model(&self) -> String {
        "wllama-default".to_string()
    }

    async fn infer(
        &mut self,
        _prompt: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        Err(anyhow!("wllama engine not yet connected"))
    }

    async fn embed(
        &mut self,
        _input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        Err(anyhow!("wllama engine not yet connected"))
    }
}
