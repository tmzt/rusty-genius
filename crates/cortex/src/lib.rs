use anyhow::Result;
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::manifest::EngineConfig;
use rusty_genius_thinkerv1 as thinkerv1;

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
    /// Returns a channel of thinkerv1::Response (Event type)
    async fn infer(
        &mut self,
        id: String, // New parameter
        prompt: &str,
        config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<thinkerv1::Response>>>;

    /// Generate embeddings
    /// Returns a channel of thinkerv1::Response (Event type)
    async fn embed(
        &mut self,
        id: String, // New parameter
        input: &str,
        config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<thinkerv1::Response>>>;
}
