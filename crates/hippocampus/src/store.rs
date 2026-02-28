use std::cell::RefCell;
use std::collections::HashSet;

use async_trait::async_trait;
use rusty_genius_core::error::GeniusError;
use rusty_genius_core::memory::{MemoryObject, MemoryObjectType, MemoryStore};

use crate::cosine::cosine_similarity;
use crate::fts::FtsIndex;
use crate::idb::ContentStore;
use crate::wrapper::WasmSendSync;

/// Browser-compatible `MemoryStore` backed by IndexedDB (content + embeddings)
/// and SQLite FTS5 (full-text search), both persisted in IndexedDB.
pub struct IdbMemoryStore {
    content: WasmSendSync<ContentStore>,
    fts: WasmSendSync<RefCell<FtsIndex>>,
}

impl IdbMemoryStore {
    /// Create a new IdbMemoryStore. Both the IndexedDB database and the
    /// SQLite FTS5 database (persisted via RelaxedIdbVFS) are initialized.
    pub async fn open() -> Result<Self, String> {
        let content = ContentStore::open().await?;
        let fts = FtsIndex::open("hippocampus.db").await?;

        Ok(Self {
            content: WasmSendSync(content),
            fts: WasmSendSync(RefCell::new(fts)),
        })
    }

    /// Perform vector search: load all embeddings, compute cosine similarity,
    /// return top-N IDs sorted by descending similarity.
    async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        let all_embeddings = self.content.get_all_embeddings().await?;

        let mut scored: Vec<(String, f32)> = all_embeddings
            .into_iter()
            .map(|(id, emb)| {
                let sim = cosine_similarity(query_embedding, &emb);
                (id, sim)
            })
            .filter(|(_, sim)| *sim > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }
}

/// Serialize a MemoryObjectType to a canonical JSON string for comparison.
fn type_json(object_type: &MemoryObjectType) -> String {
    serde_json::to_string(object_type).unwrap_or_default()
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MemoryStore for IdbMemoryStore {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError> {
        let id = object.id.clone();
        let type_json = type_json(&object.object_type);

        // Store in IndexedDB (content + embedding)
        self.content
            .put(&object)
            .await
            .map_err(|e| GeniusError::MemoryError(e))?;

        // Index in SQLite FTS
        self.fts
            .borrow()
            .index_object(
                &object.id,
                &object.short_name,
                &object.long_name,
                &object.description,
                &type_json,
                &object.content,
                object.metadata.as_deref(),
                object.created_at,
                object.updated_at,
            )
            .map_err(|e| GeniusError::MemoryError(e))?;

        Ok(id)
    }

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_filter = object_type.map(type_json);

        // 1. Vector search
        let vector_ids = self
            .vector_search(embedding, limit * 2)
            .await
            .map_err(|e| GeniusError::MemoryError(e))?;

        // 2. FTS5 text search
        let fts_ids = self
            .fts
            .borrow()
            .search(query, limit * 2)
            .map_err(|e| GeniusError::MemoryError(e))?;

        // 3. Merge: vector results first, then FTS results, deduplicate
        let mut seen = HashSet::new();
        let mut merged_ids = Vec::new();

        for (id, _score) in &vector_ids {
            if seen.insert(id.clone()) {
                merged_ids.push(id.clone());
            }
        }
        for id in &fts_ids {
            if seen.insert(id.clone()) {
                merged_ids.push(id.clone());
            }
        }

        // 4. Load full objects, apply type filter, truncate to limit
        let mut results = Vec::new();
        for id in &merged_ids {
            if results.len() >= limit {
                break;
            }
            if let Some(obj) = self
                .content
                .get(id)
                .await
                .map_err(|e| GeniusError::MemoryError(e))?
            {
                if let Some(ref tf) = type_filter {
                    if &type_json(&obj.object_type) != tf {
                        continue;
                    }
                }
                results.push(obj);
            }
        }

        Ok(results)
    }

    async fn recall_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_filter = object_type.map(type_json);

        let vector_ids = self
            .vector_search(embedding, limit * 2)
            .await
            .map_err(|e| GeniusError::MemoryError(e))?;

        let mut results = Vec::new();
        for (id, _score) in &vector_ids {
            if results.len() >= limit {
                break;
            }
            if let Some(obj) = self
                .content
                .get(id)
                .await
                .map_err(|e| GeniusError::MemoryError(e))?
            {
                if let Some(ref tf) = type_filter {
                    if &type_json(&obj.object_type) != tf {
                        continue;
                    }
                }
                results.push(obj);
            }
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        self.content
            .get(id)
            .await
            .map_err(|e| GeniusError::MemoryError(e))
    }

    async fn forget(&self, id: &str) -> Result<(), GeniusError> {
        self.content
            .delete(id)
            .await
            .map_err(|e| GeniusError::MemoryError(e))?;
        self.fts
            .borrow()
            .remove_object(id)
            .map_err(|e| GeniusError::MemoryError(e))?;
        Ok(())
    }

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_str = type_json(object_type);
        let ids = self
            .fts
            .borrow()
            .list_by_type(&type_str)
            .map_err(|e| GeniusError::MemoryError(e))?;

        let mut results = Vec::new();
        for id in &ids {
            if let Some(obj) = self
                .content
                .get(id)
                .await
                .map_err(|e| GeniusError::MemoryError(e))?
            {
                results.push(obj);
            }
        }
        Ok(results)
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError> {
        self.content
            .get_all()
            .await
            .map_err(|e| GeniusError::MemoryError(e))
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        self.content
            .clear()
            .await
            .map_err(|e| GeniusError::MemoryError(e))?;
        self.fts
            .borrow()
            .clear()
            .map_err(|e| GeniusError::MemoryError(e))?;
        Ok(())
    }
}
