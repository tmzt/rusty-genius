use serde::{Deserialize, Serialize};

/// A generic memory object with `object_type` stored as a plain `String`.
///
/// This makes the crate usable by any application — callers can store
/// domain-specific type enums by serializing them to/from this String field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryObject {
    pub id: String,
    pub short_name: String,
    pub long_name: String,
    pub description: String,
    pub object_type: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub ttl: Option<u64>,
}
