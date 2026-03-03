use anyhow::Result;
use async_trait::async_trait;
use futures::channel::mpsc;

use crate::manifest::InferenceConfig;
use crate::protocol::{ChatContent, ChatMessage, InferenceEvent, ToolDefinition};

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

    /// Run inference with tool-use support.
    /// Default implementation extracts the last user text and falls back to `infer()`.
    async fn infer_with_tools(
        &mut self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        // Extract last user text message as the prompt
        let prompt = messages
            .iter()
            .rev()
            .find_map(|m| match &m.content {
                ChatContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();
        self.infer(&prompt, config).await
    }

    /// Whether this engine supports native tool use
    fn supports_tool_use(&self) -> bool {
        false
    }
}
