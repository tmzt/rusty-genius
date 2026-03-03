pub mod context;
pub mod cosine;
pub mod engine;
pub mod error;
pub mod manifest;
pub mod memory;
pub mod protocol;
pub mod tools;

pub use context::{ContextStore, InMemoryContextStore};
pub use cosine::cosine_similarity;
pub use engine::Engine;
pub use error::GeniusError;
pub use memory::{
    EmbeddingProvider, InMemoryMemoryStore, MemoryObject, MemoryObjectType, MemoryStore,
    MockEmbeddingProvider,
};
pub use protocol::{
    ChatContent, ChatMessage, ChatRole, ToolCall, ToolDefinition, ToolResult,
};
pub use tools::{CompositeToolExecutor, ToolExecutor};
