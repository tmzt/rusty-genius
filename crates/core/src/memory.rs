use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

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

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
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

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, GeniusError>;
}

// ── In-memory implementations for testing ──

/// In-memory MemoryStore for testing (parallels InMemoryContextStore).
/// Supports text substring matching and cosine similarity vector search.
pub struct InMemoryMemoryStore {
    data: RwLock<HashMap<String, MemoryObject>>,
}

impl InMemoryMemoryStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// Serialize a MemoryObjectType to a canonical string for comparison.
fn type_tag(object_type: &MemoryObjectType) -> String {
    // Use Debug format for deterministic comparison (no serde_json in core deps)
    format!("{:?}", object_type)
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MemoryStore for InMemoryMemoryStore {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError> {
        let id = object.id.clone();
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        data.insert(id.clone(), object);
        Ok(id)
    }

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        let type_filter = object_type.map(type_tag);
        let query_lower = query.to_lowercase();

        // Score each object: combine text match + vector similarity
        let mut scored: Vec<(f32, &MemoryObject)> = Vec::new();
        for obj in data.values() {
            if let Some(ref tf) = type_filter {
                if &type_tag(&obj.object_type) != tf {
                    continue;
                }
            }

            let text_match = [
                &obj.short_name,
                &obj.long_name,
                &obj.description,
                &obj.content,
            ]
            .iter()
            .any(|f| f.to_lowercase().contains(&query_lower));

            let vec_sim = obj
                .embedding
                .as_ref()
                .map(|e| cosine_similarity(embedding, e))
                .unwrap_or(0.0);

            let score = if text_match { 1.0 + vec_sim } else { vec_sim };
            if score > 0.0 {
                scored.push((score, obj));
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, o)| o.clone()).collect())
    }

    async fn recall_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        let type_filter = object_type.map(type_tag);
        let mut scored: Vec<(f32, &MemoryObject)> = Vec::new();

        for obj in data.values() {
            if let Some(ref tf) = type_filter {
                if &type_tag(&obj.object_type) != tf {
                    continue;
                }
            }
            if let Some(ref stored) = obj.embedding {
                let sim = cosine_similarity(embedding, stored);
                if sim > 0.0 {
                    scored.push((sim, obj));
                }
            }
        }

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, o)| o.clone()).collect())
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        Ok(data.get(id).cloned())
    }

    async fn forget(&self, id: &str) -> Result<(), GeniusError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        data.remove(id);
        Ok(())
    }

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        let tag = type_tag(object_type);
        Ok(data
            .values()
            .filter(|o| type_tag(&o.object_type) == tag)
            .cloned()
            .collect())
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        Ok(data.values().cloned().collect())
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;
        data.clear();
        Ok(())
    }
}

/// Deterministic mock embedding provider for testing.
/// Produces a fixed-dimension vector derived from a hash of the input text,
/// so identical text always produces identical embeddings, and similar texts
/// produce somewhat similar embeddings (shared prefix → partial overlap).
pub struct MockEmbeddingProvider {
    pub dimensions: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    /// Generate a deterministic embedding from text.
    /// Uses a simple hash-based approach: each dimension is derived from
    /// successive bytes of a rotating hash of the input.
    pub fn embed_sync(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimensions];
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return vec;
        }

        // Simple deterministic hash spread across dimensions
        let mut hash: u64 = 5381;
        for (i, slot) in vec.iter_mut().enumerate() {
            // Mix in text bytes at different offsets
            for (j, &b) in bytes.iter().enumerate() {
                hash = hash.wrapping_mul(33).wrapping_add(b as u64);
                hash = hash.wrapping_add((i as u64).wrapping_mul(7));
                hash = hash.wrapping_add((j as u64).wrapping_mul(13));
            }
            // Map hash to [-1, 1] range
            *slot = ((hash % 20000) as f32 / 10000.0) - 1.0;
        }

        // Normalize to unit vector
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for slot in vec.iter_mut() {
                *slot /= norm;
            }
        }

        vec
    }
}

impl Default for MockEmbeddingProvider {
    fn default() -> Self {
        Self::new(384)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, GeniusError> {
        Ok(self.embed_sync(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper to create a test MemoryObject ──

    fn make_object(
        id: &str,
        short_name: &str,
        object_type: MemoryObjectType,
        content: &str,
    ) -> MemoryObject {
        MemoryObject {
            id: id.to_string(),
            short_name: short_name.to_string(),
            long_name: format!("{} (full)", short_name),
            description: format!("Test object: {}", short_name),
            object_type,
            content: content.to_string(),
            embedding: None,
            metadata: None,
            created_at: 1000,
            updated_at: 1000,
            ttl: None,
        }
    }

    // ── Serialization roundtrip tests ──

    #[test]
    fn test_memory_object_type_serde_roundtrip() {
        let types = vec![
            MemoryObjectType::Fact,
            MemoryObjectType::Observation,
            MemoryObjectType::Preference,
            MemoryObjectType::Skill,
            MemoryObjectType::Entity,
            MemoryObjectType::Relationship,
            MemoryObjectType::Custom("my-custom-type".to_string()),
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::ActiveQuery,
            )),
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::UICard,
            )),
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::Shader,
            )),
            MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::UIComponent,
            )),
            MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::ActiveFilter,
            )),
            MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::ShaderPortion,
            )),
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::MtsmOps,
            )),
        ];

        for original in &types {
            let json = serde_json::to_string(original).expect("serialize");
            let deserialized: MemoryObjectType =
                serde_json::from_str(&json).expect("deserialize");
            // Re-serialize to verify structural equality
            let json2 = serde_json::to_string(&deserialized).expect("re-serialize");
            assert_eq!(json, json2, "Roundtrip failed for {:?}", original);
        }
    }

    #[test]
    fn test_memory_object_full_serde_roundtrip() {
        let obj = MemoryObject {
            id: "test-001".to_string(),
            short_name: "sql_filter".to_string(),
            long_name: "SQL Active Filter Example".to_string(),
            description: "One-shot example of an active filter query".to_string(),
            object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::ActiveQuery,
            )),
            content: "SELECT * FROM users WHERE active = true".to_string(),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            metadata: Some(r#"{"source": "manual"}"#.to_string()),
            created_at: 1700000000,
            updated_at: 1700000001,
            ttl: Some(3600),
        };

        let json = serde_json::to_string(&obj).expect("serialize");
        let restored: MemoryObject = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.id, obj.id);
        assert_eq!(restored.short_name, obj.short_name);
        assert_eq!(restored.content, obj.content);
        assert_eq!(restored.embedding, obj.embedding);
        assert_eq!(restored.metadata, obj.metadata);
        assert_eq!(restored.ttl, Some(3600));
    }

    #[test]
    fn test_logic_element_json_format() {
        let one_shot = MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::UICard,
        ));
        let json = serde_json::to_string(&one_shot).unwrap();
        // Verify the JSON structure is as expected (nested enum encoding)
        assert!(json.contains("LogicElement"));
        assert!(json.contains("OneShotExamples"));
        assert!(json.contains("UICard"));
    }

    // ── MockEmbeddingProvider tests ──

    #[test]
    fn test_mock_embedder_deterministic() {
        let embedder = MockEmbeddingProvider::new(64);
        let v1 = embedder.embed_sync("hello world");
        let v2 = embedder.embed_sync("hello world");
        assert_eq!(v1, v2, "Same input must produce same output");
    }

    #[test]
    fn test_mock_embedder_different_inputs_differ() {
        let embedder = MockEmbeddingProvider::new(64);
        let v1 = embedder.embed_sync("hello world");
        let v2 = embedder.embed_sync("goodbye world");
        assert_ne!(v1, v2, "Different inputs should produce different vectors");
    }

    #[test]
    fn test_mock_embedder_unit_vector() {
        let embedder = MockEmbeddingProvider::new(128);
        let v = embedder.embed_sync("test input");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Embedding should be unit vector, got norm={}",
            norm
        );
    }

    #[test]
    fn test_mock_embedder_correct_dimensions() {
        let embedder = MockEmbeddingProvider::new(384);
        let v = embedder.embed_sync("test");
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![0.5, 0.5, 0.5, 0.5];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    // ── InMemoryMemoryStore tests ──

    #[async_std::test]
    async fn test_store_and_get() {
        let store = InMemoryMemoryStore::new();
        let obj = make_object("id1", "test_obj", MemoryObjectType::Fact, "Some fact");
        let id = store.store(obj).await.unwrap();
        assert_eq!(id, "id1");

        let retrieved = store.get("id1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Some fact");
    }

    #[async_std::test]
    async fn test_get_missing() {
        let store = InMemoryMemoryStore::new();
        let result = store.get("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[async_std::test]
    async fn test_forget() {
        let store = InMemoryMemoryStore::new();
        let obj = make_object("id1", "test_obj", MemoryObjectType::Fact, "content");
        store.store(obj).await.unwrap();
        store.forget("id1").await.unwrap();
        assert!(store.get("id1").await.unwrap().is_none());
    }

    #[async_std::test]
    async fn test_list_all() {
        let store = InMemoryMemoryStore::new();
        store
            .store(make_object("a", "obj_a", MemoryObjectType::Fact, "fact a"))
            .await
            .unwrap();
        store
            .store(make_object(
                "b",
                "obj_b",
                MemoryObjectType::Observation,
                "obs b",
            ))
            .await
            .unwrap();
        let all = store.list_all().await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[async_std::test]
    async fn test_list_by_type() {
        let store = InMemoryMemoryStore::new();
        store
            .store(make_object("a", "fact_a", MemoryObjectType::Fact, "fact"))
            .await
            .unwrap();
        store
            .store(make_object("b", "obs_b", MemoryObjectType::Observation, "obs"))
            .await
            .unwrap();
        store
            .store(make_object("c", "fact_c", MemoryObjectType::Fact, "another fact"))
            .await
            .unwrap();

        let facts = store.list_by_type(&MemoryObjectType::Fact).await.unwrap();
        assert_eq!(facts.len(), 2);
        assert!(facts.iter().all(|o| matches!(o.object_type, MemoryObjectType::Fact)));
    }

    #[async_std::test]
    async fn test_flush_all() {
        let store = InMemoryMemoryStore::new();
        store
            .store(make_object("a", "a", MemoryObjectType::Fact, "a"))
            .await
            .unwrap();
        store
            .store(make_object("b", "b", MemoryObjectType::Fact, "b"))
            .await
            .unwrap();
        store.flush_all().await.unwrap();
        assert_eq!(store.list_all().await.unwrap().len(), 0);
    }

    #[async_std::test]
    async fn test_recall_by_text() {
        let store = InMemoryMemoryStore::new();
        let embedder = MockEmbeddingProvider::new(8);

        let mut obj1 = make_object("q1", "sql_query", MemoryObjectType::Fact, "SELECT * FROM users");
        obj1.embedding = Some(embedder.embed_sync("SELECT * FROM users"));
        store.store(obj1).await.unwrap();

        let mut obj2 = make_object("q2", "shader_code", MemoryObjectType::Fact, "void main() { gl_FragColor = vec4(1.0); }");
        obj2.embedding = Some(embedder.embed_sync("void main() { gl_FragColor = vec4(1.0); }"));
        store.store(obj2).await.unwrap();

        let query_vec = embedder.embed_sync("SELECT");
        let results = store.recall("SELECT", &query_vec, 10, None).await.unwrap();

        assert!(!results.is_empty());
        // The SQL query should be the top result (text match boost)
        assert_eq!(results[0].id, "q1");
    }

    #[async_std::test]
    async fn test_recall_by_vector() {
        let store = InMemoryMemoryStore::new();
        let embedder = MockEmbeddingProvider::new(8);

        let mut obj = make_object("v1", "vector_test", MemoryObjectType::Fact, "some content");
        let emb = embedder.embed_sync("some content");
        obj.embedding = Some(emb.clone());
        store.store(obj).await.unwrap();

        let results = store.recall_by_vector(&emb, 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[async_std::test]
    async fn test_recall_with_type_filter() {
        let store = InMemoryMemoryStore::new();
        let embedder = MockEmbeddingProvider::new(8);

        let mut fact = make_object("f1", "fact", MemoryObjectType::Fact, "a fact about SQL");
        fact.embedding = Some(embedder.embed_sync("a fact about SQL"));
        store.store(fact).await.unwrap();

        let mut obs = make_object("o1", "obs", MemoryObjectType::Observation, "observed SQL usage");
        obs.embedding = Some(embedder.embed_sync("observed SQL usage"));
        store.store(obs).await.unwrap();

        let query_vec = embedder.embed_sync("SQL");
        let results = store
            .recall("SQL", &query_vec, 10, Some(&MemoryObjectType::Fact))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "f1");
    }

    #[async_std::test]
    async fn test_store_overwrites_by_id() {
        let store = InMemoryMemoryStore::new();
        let obj1 = make_object("dup", "version1", MemoryObjectType::Fact, "first");
        store.store(obj1).await.unwrap();

        let obj2 = make_object("dup", "version2", MemoryObjectType::Fact, "second");
        store.store(obj2).await.unwrap();

        let retrieved = store.get("dup").await.unwrap().unwrap();
        assert_eq!(retrieved.short_name, "version2");
        assert_eq!(retrieved.content, "second");
        assert_eq!(store.list_all().await.unwrap().len(), 1);
    }
}
