pub use crate::manifest::InferenceConfig;
use crate::memory::{MemoryObject, MemoryObjectType};
use serde::{Deserialize, Serialize};

// ── Context protocol types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInput {
    pub id: Option<String>,
    pub command: ContextCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextCommand {
    Set { key: String, value: String },
    Get { key: String },
    Delete { key: String },
    ListKeys { pattern: String },
    FlushAll,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOutput {
    pub id: Option<String>,
    pub body: ContextBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextBody {
    Value(Option<String>),
    Keys(Vec<String>),
    Ack,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEvent {
    ProcessStart,
    Thought(ThoughtEvent),
    Content(String),
    Embedding(Vec<f32>),
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThoughtEvent {
    Start,
    Delta(String),
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainstemInput {
    pub id: Option<String>,
    pub command: BrainstemCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemCommand {
    LoadModel(String),
    /// Preload a model into memory for immediate use.
    /// `purpose` is "infer" or "embed".
    PreloadModel { model: String, purpose: String },
    Infer {
        model: Option<String>,
        prompt: String,
        config: InferenceConfig,
    },
    Embed {
        model: Option<String>,
        input: String,
        config: InferenceConfig,
    },
    /// Keep a model resident in memory. Preloads the engine and switches
    /// the cortex strategy so the model stays loaded.
    /// `duration_secs: None` means keep alive forever; `Some(n)` keeps it
    /// resident for `n` seconds before reverting to the default hibernate
    /// strategy.
    KeepResident {
        model: String,
        purpose: String,
        duration_secs: Option<u64>,
    },
    ListModels,
    Reset,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub purpose: String,
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
pub struct BrainstemOutput {
    pub id: Option<String>,
    pub body: BrainstemBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemBody {
    /// Standard inference and thought events
    Event(InferenceEvent),
    /// Progress/status of asset management
    Asset(AssetEvent),
    /// List of available models
    ModelList(Vec<ModelDescriptor>),
    /// Catch-all for engine or orchestrator errors
    Error(String),
}

// ── Memory protocol types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInput {
    pub id: Option<String>,
    pub command: MemoryCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryCommand {
    Store(MemoryObject),
    Recall {
        query: String,
        limit: usize,
        object_type: Option<MemoryObjectType>,
    },
    RecallByVector {
        embedding: Vec<f32>,
        limit: usize,
        object_type: Option<MemoryObjectType>,
    },
    Get {
        object_id: String,
    },
    Forget {
        object_id: String,
    },
    ListByType {
        object_type: MemoryObjectType,
    },
    /// Consolidate PFC → Neocortex
    Ship,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOutput {
    pub id: Option<String>,
    pub body: MemoryBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryBody {
    Stored(String),
    Recalled(Vec<MemoryObject>),
    Object(Option<MemoryObject>),
    Ack,
    Error(String),
}
