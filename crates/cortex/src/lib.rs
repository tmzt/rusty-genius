use anyhow::Result;
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;

pub mod backend;

pub use backend::create_engine;

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

    /// Run inference
    /// Returns a channel of InferenceEvents
    async fn infer(
        &mut self,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>>;

    /// Generate embeddings
    /// Returns a channel of InferenceEvents (will emit Embedding event)
    async fn embed(
        &mut self,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>>;
}
