use async_trait::async_trait;

use crate::types::MemoryObject;

/// Persistent memory store with full-text and vector search.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait MemoryStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn store(&self, object: MemoryObject) -> Result<String, Self::Error>;

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&str>,
    ) -> Result<Vec<MemoryObject>, Self::Error>;

    async fn recall_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&str>,
    ) -> Result<Vec<MemoryObject>, Self::Error>;

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, Self::Error>;

    async fn forget(&self, id: &str) -> Result<(), Self::Error>;

    async fn list_by_type(&self, object_type: &str) -> Result<Vec<MemoryObject>, Self::Error>;

    async fn list_all(&self) -> Result<Vec<MemoryObject>, Self::Error>;

    async fn flush_all(&self) -> Result<(), Self::Error>;
}

/// Provider that generates embedding vectors from text.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait EmbeddingProvider: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn embed(&self, text: &str) -> Result<Vec<f32>, Self::Error>;
}
