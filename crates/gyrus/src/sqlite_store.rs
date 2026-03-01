use async_trait::async_trait;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

use rusty_genius_core::cosine::cosine_similarity;
use crate::error::GyrusError;
use crate::schema;
use crate::traits::MemoryStore;
use crate::types::MemoryObject;

pub struct SqliteMemoryStore {
    pool: SqlitePool,
}

impl SqliteMemoryStore {
    pub async fn new(database_url: &str) -> Result<Self, GyrusError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        schema::init_db(&pool).await?;

        Ok(Self { pool })
    }

    /// Create from an existing pool (useful for testing or shared connections).
    pub async fn from_pool(pool: SqlitePool) -> Result<Self, GyrusError> {
        schema::init_db(&pool).await?;
        Ok(Self { pool })
    }

    async fn store_embedding(&self, id: &str, embedding: &[f32]) -> Result<(), GyrusError> {
        let vec_json = serde_json::to_string(embedding)
            .map_err(|e| GyrusError::Serialization(format!("Serialize vec error: {}", e)))?;

        #[cfg(feature = "vec0")]
        {
            let _ = sqlx::query("DELETE FROM memory_vec WHERE id = ?")
                .bind(id)
                .execute(&self.pool)
                .await;
            sqlx::query("INSERT INTO memory_vec (id, embedding) VALUES (?, ?)")
                .bind(id)
                .bind(&vec_json)
                .execute(&self.pool)
                .await?;
        }

        #[cfg(not(feature = "vec0"))]
        {
            sqlx::query(
                "INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)",
            )
            .bind(id)
            .bind(&vec_json)
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    async fn load_embedding(&self, id: &str) -> Option<Vec<f32>> {
        #[cfg(feature = "vec0")]
        let table = "memory_vec";
        #[cfg(not(feature = "vec0"))]
        let table = "memory_embeddings";

        let query = format!("SELECT embedding FROM {} WHERE id = ?", table);
        sqlx::query(&query)
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .ok()
            .flatten()
            .and_then(|r| {
                let raw: String = r.get("embedding");
                serde_json::from_str(&raw).ok()
            })
    }

    async fn delete_embedding(&self, id: &str) -> Result<(), GyrusError> {
        #[cfg(feature = "vec0")]
        let table = "memory_vec";
        #[cfg(not(feature = "vec0"))]
        let table = "memory_embeddings";

        let query = format!("DELETE FROM {} WHERE id = ?", table);
        sqlx::query(&query)
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    #[cfg(feature = "vec0")]
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>, GyrusError> {
        let vec_json = serde_json::to_string(embedding)
            .map_err(|e| GyrusError::Serialization(format!("Serialize query vec: {}", e)))?;

        let rows = sqlx::query(
            "SELECT id, distance FROM memory_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        )
        .bind(&vec_json)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    #[cfg(not(feature = "vec0"))]
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>, GyrusError> {
        let rows = sqlx::query("SELECT id, embedding FROM memory_embeddings")
            .fetch_all(&self.pool)
            .await?;

        let mut scored: Vec<(String, f32)> = Vec::new();
        for row in &rows {
            let id: String = row.get("id");
            let raw: String = row.get("embedding");
            if let Ok(stored) = serde_json::from_str::<Vec<f32>>(&raw) {
                let sim = cosine_similarity(embedding, &stored);
                scored.push((id, sim));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(id, _)| id).collect())
    }

    fn row_to_object(row: &sqlx::sqlite::SqliteRow) -> Result<MemoryObject, GyrusError> {
        Ok(MemoryObject {
            id: row.get("id"),
            short_name: row.get("short_name"),
            long_name: row.get("long_name"),
            description: row.get("description"),
            object_type: row.get("object_type"),
            content: row.get("content"),
            embedding: None,
            metadata: row.get("metadata"),
            created_at: row.get::<i64, _>("created_at") as u64,
            updated_at: row.get::<i64, _>("updated_at") as u64,
            ttl: None,
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MemoryStore for SqliteMemoryStore {
    type Error = GyrusError;

    async fn store(&self, object: MemoryObject) -> Result<String, GyrusError> {
        sqlx::query(
            r#"INSERT OR REPLACE INTO memory_objects
               (id, short_name, long_name, description, object_type, content, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&object.id)
        .bind(&object.short_name)
        .bind(&object.long_name)
        .bind(&object.description)
        .bind(&object.object_type)
        .bind(&object.content)
        .bind(&object.metadata)
        .bind(object.created_at as i64)
        .bind(object.updated_at as i64)
        .execute(&self.pool)
        .await?;

        if let Some(ref embedding) = object.embedding {
            self.store_embedding(&object.id, embedding).await?;
        }

        Ok(object.id)
    }

    async fn recall(
        &self,
        query: &str,
        embedding: &[f32],
        limit: usize,
        object_type: Option<&str>,
    ) -> Result<Vec<MemoryObject>, GyrusError> {
        let fts_ids: Vec<String> = sqlx::query(
            r#"SELECT mo.id
               FROM memory_fts
               JOIN memory_objects mo ON memory_fts.rowid = mo.rowid
               WHERE memory_fts MATCH ?
               ORDER BY rank
               LIMIT ?"#,
        )
        .bind(query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default()
        .iter()
        .map(|r| r.get::<String, _>("id"))
        .collect();

        let vec_ids = self.vector_search(embedding, limit).await?;

        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();

        for id in vec_ids.iter().chain(fts_ids.iter()) {
            if results.len() >= limit {
                break;
            }
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(obj) = self.get(id).await? {
                if let Some(tf) = object_type {
                    if obj.object_type != tf {
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
        object_type: Option<&str>,
    ) -> Result<Vec<MemoryObject>, GyrusError> {
        let ids = self.vector_search(embedding, limit).await?;
        let mut results = Vec::new();

        for id in &ids {
            if let Some(obj) = self.get(id).await? {
                if let Some(tf) = object_type {
                    if obj.object_type != tf {
                        continue;
                    }
                }
                results.push(obj);
            }
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GyrusError> {
        let row = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            None => Ok(None),
            Some(row) => {
                let mut obj = Self::row_to_object(&row)?;
                obj.embedding = self.load_embedding(id).await;
                Ok(Some(obj))
            }
        }
    }

    async fn forget(&self, id: &str) -> Result<(), GyrusError> {
        self.delete_embedding(id).await?;

        sqlx::query("DELETE FROM memory_objects WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    async fn list_by_type(&self, object_type: &str) -> Result<Vec<MemoryObject>, GyrusError> {
        let rows = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects WHERE object_type = ?"#,
        )
        .bind(object_type)
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(Self::row_to_object).collect()
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GyrusError> {
        let rows = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects"#,
        )
        .fetch_all(&self.pool)
        .await?;

        rows.iter().map(Self::row_to_object).collect()
    }

    async fn flush_all(&self) -> Result<(), GyrusError> {
        #[cfg(feature = "vec0")]
        sqlx::query("DELETE FROM memory_vec")
            .execute(&self.pool)
            .await?;

        #[cfg(not(feature = "vec0"))]
        sqlx::query("DELETE FROM memory_embeddings")
            .execute(&self.pool)
            .await?;

        sqlx::query("DELETE FROM memory_objects")
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_store() -> SqliteMemoryStore {
        SqliteMemoryStore::new("sqlite::memory:").await.unwrap()
    }

    fn make_object(id: &str, short_name: &str, object_type: &str, content: &str) -> MemoryObject {
        MemoryObject {
            id: id.to_string(),
            short_name: short_name.to_string(),
            long_name: format!("{} (full)", short_name),
            description: format!("Test object: {}", short_name),
            object_type: object_type.to_string(),
            content: content.to_string(),
            embedding: None,
            metadata: None,
            created_at: 1000,
            updated_at: 1000,
            ttl: None,
        }
    }

    #[async_std::test]
    async fn store_and_get() {
        let store = make_store().await;
        let obj = make_object("id1", "test", "Fact", "Some fact");
        let id = store.store(obj).await.unwrap();
        assert_eq!(id, "id1");

        let retrieved = store.get("id1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Some fact");
    }

    #[async_std::test]
    async fn get_missing() {
        let store = make_store().await;
        assert!(store.get("nonexistent").await.unwrap().is_none());
    }

    #[async_std::test]
    async fn forget() {
        let store = make_store().await;
        store.store(make_object("id1", "test", "Fact", "content")).await.unwrap();
        store.forget("id1").await.unwrap();
        assert!(store.get("id1").await.unwrap().is_none());
    }

    #[async_std::test]
    async fn list_all() {
        let store = make_store().await;
        store.store(make_object("a", "a", "Fact", "fact")).await.unwrap();
        store.store(make_object("b", "b", "Observation", "obs")).await.unwrap();
        assert_eq!(store.list_all().await.unwrap().len(), 2);
    }

    #[async_std::test]
    async fn list_by_type() {
        let store = make_store().await;
        store.store(make_object("a", "a", "Fact", "fact")).await.unwrap();
        store.store(make_object("b", "b", "Observation", "obs")).await.unwrap();
        store.store(make_object("c", "c", "Fact", "another")).await.unwrap();

        let facts = store.list_by_type("Fact").await.unwrap();
        assert_eq!(facts.len(), 2);
    }

    #[async_std::test]
    async fn flush_all() {
        let store = make_store().await;
        store.store(make_object("a", "a", "Fact", "a")).await.unwrap();
        store.store(make_object("b", "b", "Fact", "b")).await.unwrap();
        store.flush_all().await.unwrap();
        assert_eq!(store.list_all().await.unwrap().len(), 0);
    }

    #[async_std::test]
    async fn store_with_embedding() {
        let store = make_store().await;
        let mut obj = make_object("e1", "emb", "Fact", "content");
        obj.embedding = Some(vec![0.1, 0.2, 0.3]);
        store.store(obj).await.unwrap();

        let retrieved = store.get("e1").await.unwrap().unwrap();
        assert!(retrieved.embedding.is_some());
        let emb = retrieved.embedding.unwrap();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 0.1).abs() < 0.001);
    }

    #[async_std::test]
    async fn vector_recall() {
        let store = make_store().await;

        let mut obj1 = make_object("v1", "vec1", "Fact", "content1");
        obj1.embedding = Some(vec![1.0, 0.0, 0.0]);
        store.store(obj1).await.unwrap();

        let mut obj2 = make_object("v2", "vec2", "Fact", "content2");
        obj2.embedding = Some(vec![0.0, 1.0, 0.0]);
        store.store(obj2).await.unwrap();

        let results = store.recall_by_vector(&[1.0, 0.0, 0.0], 10, None).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "v1");
    }
}
