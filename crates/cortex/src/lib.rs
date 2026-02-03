use anyhow::Result;
use async_trait::async_trait;
use rusty_genius_core::protocol::InferenceEvent;
use tokio::sync::mpsc;

pub mod backend;

#[async_trait]
pub trait Engine: Send + Sync {
    /// Load a model from a path
    async fn load_model(&mut self, model_path: &str) -> Result<()>;

    /// Run inference
    /// Returns a channel of InferenceEvents
    async fn infer(
        &mut self,
        prompt: &str,
        // config: InferenceConfig, // We might need to import this
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
