use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::GeniusError;

// ── Memory object types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryObjectType {
    LogicElement(LogicElement),
    Fact,
    Observation,
    Preference,
    Skill,
    Entity,
    Relationship,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicElement {
    OneShotExamples(LogicElementSubtype),
    FewShotExamples(LogicElementSubtype),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicElementSubtype {
    ActiveQuery,
    ActiveFilter,
    UICard,
    UIComponent,
    Shader,
    ShaderPortion,
    MtsmOps,
}

// ── Memory object ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryObject {
    pub id: String,
    pub short_name: String,
    pub long_name: String,
    pub description: String,
    pub object_type: MemoryObjectType,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Option<String>, // JSON blob
    pub created_at: u64,
    pub updated_at: u64,
    pub ttl: Option<u64>, // PFC expiry (seconds from created_at)
}

// ── Traits ──

#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError>;

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError>;

    async fn recall_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError>;

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError>;

    async fn forget(&self, id: &str) -> Result<(), GeniusError>;

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError>;

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError>;

    async fn flush_all(&self) -> Result<(), GeniusError>;
}

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, GeniusError>;
}
