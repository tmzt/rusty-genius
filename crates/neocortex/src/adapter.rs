use async_trait::async_trait;
use rusty_genius_core::error::GeniusError;
use rusty_genius_core::memory::{MemoryObject, MemoryObjectType, MemoryStore};

use gyrus::SqliteMemoryStore as GyrusSqliteStore;
use gyrus::MemoryObject as GyrusMemoryObject;
use gyrus::traits::MemoryStore as GyrusMemoryStore;

/// Adapter that wraps `gyrus::SqliteMemoryStore` and implements
/// `rusty_genius_core::memory::MemoryStore` with `MemoryObjectType` enum.
pub struct SqliteMemoryStore {
    inner: GyrusSqliteStore,
}

impl SqliteMemoryStore {
    pub async fn new(database_url: &str) -> Result<Self, GeniusError> {
        let inner = GyrusSqliteStore::new(database_url)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("SQLite connection error: {}", e)))?;
        Ok(Self { inner })
    }
}

fn core_to_gyrus(obj: MemoryObject) -> GyrusMemoryObject {
    GyrusMemoryObject {
        id: obj.id,
        short_name: obj.short_name,
        long_name: obj.long_name,
        description: obj.description,
        object_type: serde_json::to_string(&obj.object_type)
            .unwrap_or_else(|_| "\"Fact\"".to_string()),
        content: obj.content,
        embedding: obj.embedding,
        metadata: obj.metadata,
        created_at: obj.created_at,
        updated_at: obj.updated_at,
        ttl: obj.ttl,
    }
}

fn gyrus_to_core(obj: GyrusMemoryObject) -> Result<MemoryObject, GeniusError> {
    let object_type: MemoryObjectType = serde_json::from_str(&obj.object_type)
        .map_err(|e| GeniusError::MemoryError(format!("Deserialize object_type: {}", e)))?;

    Ok(MemoryObject {
        id: obj.id,
        short_name: obj.short_name,
        long_name: obj.long_name,
        description: obj.description,
        object_type,
        content: obj.content,
        embedding: obj.embedding,
        metadata: obj.metadata,
        created_at: obj.created_at,
        updated_at: obj.updated_at,
        ttl: obj.ttl,
    })
}

fn type_json(object_type: &MemoryObjectType) -> String {
    serde_json::to_string(object_type).unwrap_or_else(|_| "\"Fact\"".to_string())
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MemoryStore for SqliteMemoryStore {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError> {
        self.inner
            .store(core_to_gyrus(object))
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))
    }

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_str = object_type.map(type_json);
        let results = self
            .inner
            .recall(query, embedding, limit, type_str.as_deref())
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        results.into_iter().map(gyrus_to_core).collect()
    }

    async fn recall_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_str = object_type.map(type_json);
        let results = self
            .inner
            .recall_by_vector(embedding, limit, type_str.as_deref())
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        results.into_iter().map(gyrus_to_core).collect()
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        let obj = self
            .inner
            .get(id)
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        match obj {
            None => Ok(None),
            Some(o) => Ok(Some(gyrus_to_core(o)?)),
        }
    }

    async fn forget(&self, id: &str) -> Result<(), GeniusError> {
        self.inner
            .forget(id)
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))
    }

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_str = type_json(object_type);
        let results = self
            .inner
            .list_by_type(&type_str)
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        results.into_iter().map(gyrus_to_core).collect()
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError> {
        let results = self
            .inner
            .list_all()
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))?;

        results.into_iter().map(gyrus_to_core).collect()
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        self.inner
            .flush_all()
            .await
            .map_err(|e| GeniusError::MemoryError(e.to_string()))
    }
}
