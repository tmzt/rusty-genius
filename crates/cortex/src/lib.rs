use anyhow::Result;
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;

pub mod backend;

#[async_trait]
pub trait Engine: Send + Sync {
    /// Load a model from a path
    async fn load_model(&mut self, model_path: &str) -> Result<()>;

    /// Unload the currently loaded model to free resources
    async fn unload_model(&mut self) -> Result<()>;

    /// Check if a model is currently loaded
    fn is_loaded(&self) -> bool;

    /// Run inference
    /// Returns a channel of InferenceEvents
    async fn infer(
        &mut self,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>>;
}

pub async fn create_engine() -> Box<dyn Engine> {
    #[cfg(feature = "real-engine")]
    {
        Box::new(backend::Brain::new())
    }

    #[cfg(not(feature = "real-engine"))]
    {
        Box::new(backend::Pinky::new())
    }
}
