use rexie::{ObjectStore, Rexie, TransactionMode};
use rusty_genius_core::memory::{MemoryObject, MemoryObjectType};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

const DB_NAME: &str = "rusty-genius-memory";
const DB_VERSION: u32 = 1;
const STORE_OBJECTS: &str = "memory_objects";
const STORE_EMBEDDINGS: &str = "memory_embeddings";

/// MemoryObject without the embedding field — stored in the "memory_objects" IDB store.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredObject {
    pub id: String,
    pub short_name: String,
    pub long_name: String,
    pub description: String,
    pub object_type: MemoryObjectType,
    pub content: String,
    pub metadata: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub ttl: Option<u64>,
}

/// Embedding entry stored in the "memory_embeddings" IDB store.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEmbedding {
    pub id: String,
    pub embedding: Vec<f32>,
}

impl From<&MemoryObject> for StoredObject {
    fn from(obj: &MemoryObject) -> Self {
        Self {
            id: obj.id.clone(),
            short_name: obj.short_name.clone(),
            long_name: obj.long_name.clone(),
            description: obj.description.clone(),
            object_type: obj.object_type.clone(),
            content: obj.content.clone(),
            metadata: obj.metadata.clone(),
            created_at: obj.created_at,
            updated_at: obj.updated_at,
            ttl: obj.ttl,
        }
    }
}

/// IndexedDB content + embedding layer using `rexie`.
pub struct ContentStore {
    db: Rexie,
}

impl ContentStore {
    /// Open (or create) the IndexedDB database.
    pub async fn open() -> Result<Self, String> {
        let db = Rexie::builder(DB_NAME)
            .version(DB_VERSION)
            .add_object_store(
                ObjectStore::new(STORE_OBJECTS)
                    .key_path("id")
                    .add_index(rexie::Index::new("object_type", "object_type")),
            )
            .add_object_store(
                ObjectStore::new(STORE_EMBEDDINGS).key_path("id"),
            )
            .build()
            .await
            .map_err(|e| format!("Failed to open IndexedDB: {:?}", e))?;

        Ok(Self { db })
    }

    /// Store a MemoryObject: content goes to "memory_objects", embedding to "memory_embeddings".
    pub async fn put(&self, object: &MemoryObject) -> Result<(), String> {
        let stored_obj = StoredObject::from(object);
        let obj_js = serde_wasm_bindgen::to_value(&stored_obj)
            .map_err(|e| format!("serialize object: {:?}", e))?;

        // Write object
        let tx = self
            .db
            .transaction(&[STORE_OBJECTS], TransactionMode::ReadWrite)
            .map_err(|e| format!("tx open (objects): {:?}", e))?;
        let store = tx
            .store(STORE_OBJECTS)
            .map_err(|e| format!("store open: {:?}", e))?;
        store
            .put(&obj_js, None)
            .await
            .map_err(|e| format!("put object: {:?}", e))?;
        tx.done()
            .await
            .map_err(|e| format!("tx done (objects): {:?}", e))?;

        // Write embedding (if present)
        if let Some(ref emb) = object.embedding {
            let stored_emb = StoredEmbedding {
                id: object.id.clone(),
                embedding: emb.clone(),
            };
            let emb_js = serde_wasm_bindgen::to_value(&stored_emb)
                .map_err(|e| format!("serialize embedding: {:?}", e))?;

            let tx = self
                .db
                .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadWrite)
                .map_err(|e| format!("tx open (embeddings): {:?}", e))?;
            let store = tx
                .store(STORE_EMBEDDINGS)
                .map_err(|e| format!("store open (embeddings): {:?}", e))?;
            store
                .put(&emb_js, None)
                .await
                .map_err(|e| format!("put embedding: {:?}", e))?;
            tx.done()
                .await
                .map_err(|e| format!("tx done (embeddings): {:?}", e))?;
        }

        Ok(())
    }

    /// Get a single MemoryObject by ID (reassembles from object + embedding stores).
    pub async fn get(&self, id: &str) -> Result<Option<MemoryObject>, String> {
        let key = JsValue::from_str(id);

        // Load object
        let tx = self
            .db
            .transaction(&[STORE_OBJECTS], TransactionMode::ReadOnly)
            .map_err(|e| format!("tx open: {:?}", e))?;
        let store = tx
            .store(STORE_OBJECTS)
            .map_err(|e| format!("store open: {:?}", e))?;
        let js_val = store
            .get(&key)
            .await
            .map_err(|e| format!("get object: {:?}", e))?;

        if js_val.is_undefined() || js_val.is_null() {
            return Ok(None);
        }

        let stored: StoredObject = serde_wasm_bindgen::from_value(js_val)
            .map_err(|e| format!("deserialize object: {:?}", e))?;

        // Load embedding
        let tx = self
            .db
            .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadOnly)
            .map_err(|e| format!("tx open (emb): {:?}", e))?;
        let emb_store = tx
            .store(STORE_EMBEDDINGS)
            .map_err(|e| format!("store open (emb): {:?}", e))?;
        let emb_js = emb_store
            .get(&key)
            .await
            .map_err(|e| format!("get embedding: {:?}", e))?;

        let embedding = if !emb_js.is_undefined() && !emb_js.is_null() {
            let stored_emb: StoredEmbedding = serde_wasm_bindgen::from_value(emb_js)
                .map_err(|e| format!("deserialize embedding: {:?}", e))?;
            Some(stored_emb.embedding)
        } else {
            None
        };

        Ok(Some(MemoryObject {
            id: stored.id,
            short_name: stored.short_name,
            long_name: stored.long_name,
            description: stored.description,
            object_type: stored.object_type,
            content: stored.content,
            embedding,
            metadata: stored.metadata,
            created_at: stored.created_at,
            updated_at: stored.updated_at,
            ttl: stored.ttl,
        }))
    }

    /// Delete a MemoryObject (and its embedding) by ID.
    pub async fn delete(&self, id: &str) -> Result<(), String> {
        let key = JsValue::from_str(id);

        let tx = self
            .db
            .transaction(&[STORE_OBJECTS], TransactionMode::ReadWrite)
            .map_err(|e| format!("tx open: {:?}", e))?;
        let store = tx
            .store(STORE_OBJECTS)
            .map_err(|e| format!("store open: {:?}", e))?;
        store
            .delete(&key)
            .await
            .map_err(|e| format!("delete object: {:?}", e))?;
        tx.done()
            .await
            .map_err(|e| format!("tx done: {:?}", e))?;

        let tx = self
            .db
            .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadWrite)
            .map_err(|e| format!("tx open (emb): {:?}", e))?;
        let store = tx
            .store(STORE_EMBEDDINGS)
            .map_err(|e| format!("store open (emb): {:?}", e))?;
        store
            .delete(&key)
            .await
            .map_err(|e| format!("delete embedding: {:?}", e))?;
        tx.done()
            .await
            .map_err(|e| format!("tx done (emb): {:?}", e))?;

        Ok(())
    }

    /// Get all MemoryObjects from IndexedDB (reassembles embeddings).
    pub async fn get_all(&self) -> Result<Vec<MemoryObject>, String> {
        let tx = self
            .db
            .transaction(&[STORE_OBJECTS], TransactionMode::ReadOnly)
            .map_err(|e| format!("tx open: {:?}", e))?;
        let store = tx
            .store(STORE_OBJECTS)
            .map_err(|e| format!("store open: {:?}", e))?;
        let entries = store
            .get_all(None, None, None, None)
            .await
            .map_err(|e| format!("get_all objects: {:?}", e))?;

        let mut objects = Vec::with_capacity(entries.len());
        for (_key, val) in &entries {
            let stored: StoredObject = serde_wasm_bindgen::from_value(val.clone())
                .map_err(|e| format!("deserialize object: {:?}", e))?;

            // Load embedding for this object
            let emb = self.get_embedding(&stored.id).await?;

            objects.push(MemoryObject {
                id: stored.id,
                short_name: stored.short_name,
                long_name: stored.long_name,
                description: stored.description,
                object_type: stored.object_type,
                content: stored.content,
                embedding: emb,
                metadata: stored.metadata,
                created_at: stored.created_at,
                updated_at: stored.updated_at,
                ttl: stored.ttl,
            });
        }

        Ok(objects)
    }

    /// Load all embeddings as (id, Vec<f32>) pairs for vector search.
    pub async fn get_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, String> {
        let tx = self
            .db
            .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadOnly)
            .map_err(|e| format!("tx open: {:?}", e))?;
        let store = tx
            .store(STORE_EMBEDDINGS)
            .map_err(|e| format!("store open: {:?}", e))?;
        let entries = store
            .get_all(None, None, None, None)
            .await
            .map_err(|e| format!("get_all embeddings: {:?}", e))?;

        let mut result = Vec::with_capacity(entries.len());
        for (_key, val) in &entries {
            let stored: StoredEmbedding = serde_wasm_bindgen::from_value(val.clone())
                .map_err(|e| format!("deserialize embedding: {:?}", e))?;
            result.push((stored.id, stored.embedding));
        }
        Ok(result)
    }

    /// Clear all data from both object stores.
    pub async fn clear(&self) -> Result<(), String> {
        let tx = self
            .db
            .transaction(&[STORE_OBJECTS], TransactionMode::ReadWrite)
            .map_err(|e| format!("tx open: {:?}", e))?;
        let store = tx
            .store(STORE_OBJECTS)
            .map_err(|e| format!("store open: {:?}", e))?;
        store
            .clear()
            .await
            .map_err(|e| format!("clear objects: {:?}", e))?;
        tx.done()
            .await
            .map_err(|e| format!("tx done: {:?}", e))?;

        let tx = self
            .db
            .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadWrite)
            .map_err(|e| format!("tx open (emb): {:?}", e))?;
        let store = tx
            .store(STORE_EMBEDDINGS)
            .map_err(|e| format!("store open (emb): {:?}", e))?;
        store
            .clear()
            .await
            .map_err(|e| format!("clear embeddings: {:?}", e))?;
        tx.done()
            .await
            .map_err(|e| format!("tx done (emb): {:?}", e))?;

        Ok(())
    }

    // ── Private helpers ──

    async fn get_embedding(&self, id: &str) -> Result<Option<Vec<f32>>, String> {
        let key = JsValue::from_str(id);
        let tx = self
            .db
            .transaction(&[STORE_EMBEDDINGS], TransactionMode::ReadOnly)
            .map_err(|e| format!("tx open (emb): {:?}", e))?;
        let store = tx
            .store(STORE_EMBEDDINGS)
            .map_err(|e| format!("store open (emb): {:?}", e))?;
        let js_val = store
            .get(&key)
            .await
            .map_err(|e| format!("get embedding: {:?}", e))?;

        if js_val.is_undefined() || js_val.is_null() {
            Ok(None)
        } else {
            let stored: StoredEmbedding = serde_wasm_bindgen::from_value(js_val)
                .map_err(|e| format!("deserialize embedding: {:?}", e))?;
            Ok(Some(stored.embedding))
        }
    }
}
