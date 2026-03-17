use anyhow::Result;
use async_trait::async_trait;
use futures_channel::mpsc;

use crate::manifest::InferenceConfig;
use crate::protocol::InferenceEvent;

#[async_trait]
pub trait Engine: Send + Sync {
    /// Load a model from a path
    async fn load_model(&mut self, model_path: &str) -> Result<()>;

    /// Unload the currently loaded model to free resources
    async fn unload_model(&mut self) -> Result<()>;

    /// Check if a model is currently loaded
    fn is_loaded(&self) -> bool;

    /// Get the default model name for this engine
    fn default_model(&self) -> String;

    /// Preload a model into memory so it's ready for immediate use.
    /// The `purpose` hint ("infer" or "embed") tells the engine which
    /// slot to load into. The model must already be on disk.
    async fn preload_model(&mut self, model_path: &str, purpose: &str) -> Result<()>;

    /// Run inference. If `model` is Some, use that model (must already
    /// be loaded or on disk). If None, use the current/default model.
    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>>;

    /// Generate embeddings. If `model` is Some, use that model (must already
    /// be loaded or on disk). If None, use the current/default model.
    async fn embed(
        &mut self,
        model: Option<&str>,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>>;
}
