pub use crate::manifest::InferenceConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEvent {
    ProcessStart,
    Thought(ThoughtEvent),
    Content(String),
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThoughtEvent {
    Start,
    Delta(String),
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemInput {
    LoadModel(String),
    Infer {
        prompt: String,
        config: InferenceConfig,
    },
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssetEvent {
    /// Starting resolution and download process
    Started(String),
    /// Download progress in bytes (current, total)
    Progress(u64, u64),
    /// Successfully downloaded
    Complete(String),
    /// Error during asset handling
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemOutput {
    /// Standard inference and thought events
    Event(InferenceEvent),
    /// Progress/status of asset management
    Asset(AssetEvent),
    /// Catch-all for engine or orchestrator errors
    Error(String),
}
