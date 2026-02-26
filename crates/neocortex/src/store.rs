use async_trait::async_trait;
use rusty_genius_core::error::GeniusError;
use rusty_genius_core::memory::{MemoryObject, MemoryObjectType, MemoryStore};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

use crate::schema;

pub struct SqliteMemoryStore {
    pool: SqlitePool,
}

impl SqliteMemoryStore {
    pub async fn new(database_url: &str) -> Result<Self, GeniusError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("SQLite connection error: {}", e)))?;

        schema::init_db(&pool)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Schema init error: {}", e)))?;

        Ok(Self { pool })
    }

    fn type_json(object_type: &MemoryObjectType) -> String {
        serde_json::to_string(object_type).unwrap_or_else(|_| "\"Fact\"".to_string())
    }

    /// Store embedding in the appropriate table (vec0 or fallback).
    async fn store_embedding(&self, id: &str, embedding: &[f32]) -> Result<(), GeniusError> {
        let vec_json = serde_json::to_string(embedding)
            .map_err(|e| GeniusError::MemoryError(format!("Serialize vec error: {}", e)))?;

        // Delete existing entry first
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
                .await
                .map_err(|e| {
                    GeniusError::MemoryError(format!("INSERT memory_vec error: {}", e))
                })?;
        }

        #[cfg(not(feature = "vec0"))]
        {
            sqlx::query(
                "INSERT OR REPLACE INTO memory_embeddings (id, embedding) VALUES (?, ?)",
            )
            .bind(id)
            .bind(&vec_json)
            .execute(&self.pool)
            .await
            .map_err(|e| {
                GeniusError::MemoryError(format!("INSERT memory_embeddings error: {}", e))
            })?;
        }

        Ok(())
    }

    /// Load embedding for a given object ID.
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

    /// Delete embedding for a given object ID.
    async fn delete_embedding(&self, id: &str) -> Result<(), GeniusError> {
        #[cfg(feature = "vec0")]
        let table = "memory_vec";
        #[cfg(not(feature = "vec0"))]
        let table = "memory_embeddings";

        let query = format!("DELETE FROM {} WHERE id = ?", table);
        sqlx::query(&query)
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("DELETE embedding error: {}", e)))?;

        Ok(())
    }

    /// Vector search using vec0 extension.
    #[cfg(feature = "vec0")]
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>, GeniusError> {
        let vec_json = serde_json::to_string(embedding)
            .map_err(|e| GeniusError::MemoryError(format!("Serialize query vec: {}", e)))?;

        let rows = sqlx::query(
            "SELECT id, distance FROM memory_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        )
        .bind(&vec_json)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("Vec0 search error: {}", e)))?;

        Ok(rows.iter().map(|r| r.get::<String, _>("id")).collect())
    }

    /// Fallback vector search: load all embeddings and compute cosine similarity in Rust.
    #[cfg(not(feature = "vec0"))]
    async fn vector_search(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>, GeniusError> {
        let rows = sqlx::query("SELECT id, embedding FROM memory_embeddings")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("SELECT embeddings error: {}", e)))?;

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

    fn row_to_object(row: &sqlx::sqlite::SqliteRow) -> Result<MemoryObject, GeniusError> {
        let obj_type_str: String = row.get("object_type");
        let object_type: MemoryObjectType = serde_json::from_str(&obj_type_str)
            .map_err(|e| GeniusError::MemoryError(format!("Deserialize object_type: {}", e)))?;

        Ok(MemoryObject {
            id: row.get("id"),
            short_name: row.get("short_name"),
            long_name: row.get("long_name"),
            description: row.get("description"),
            object_type,
            content: row.get("content"),
            embedding: None,
            metadata: row.get("metadata"),
            created_at: row.get::<i64, _>("created_at") as u64,
            updated_at: row.get::<i64, _>("updated_at") as u64,
            ttl: None,
        })
    }
}

/// Cosine similarity between two vectors.
#[cfg(not(feature = "vec0"))]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
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

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError> {
        let type_json = Self::type_json(&object.object_type);

        sqlx::query(
            r#"INSERT OR REPLACE INTO memory_objects
               (id, short_name, long_name, description, object_type, content, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        )
        .bind(&object.id)
        .bind(&object.short_name)
        .bind(&object.long_name)
        .bind(&object.description)
        .bind(&type_json)
        .bind(&object.content)
        .bind(&object.metadata)
        .bind(object.created_at as i64)
        .bind(object.updated_at as i64)
        .execute(&self.pool)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("INSERT memory_objects error: {}", e)))?;

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
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        // FTS5 text search
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

        // Vector search
        let vec_ids = self.vector_search(embedding, limit).await?;

        // Merge: vector results first, then FTS, deduplicated
        let type_filter = object_type.map(Self::type_json);
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
                if let Some(ref tf) = type_filter {
                    if &Self::type_json(&obj.object_type) != tf {
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
        let ids = self.vector_search(embedding, limit).await?;
        let type_filter = object_type.map(Self::type_json);
        let mut results = Vec::new();

        for id in &ids {
            if let Some(obj) = self.get(id).await? {
                if let Some(ref tf) = type_filter {
                    if &Self::type_json(&obj.object_type) != tf {
                        continue;
                    }
                }
                results.push(obj);
            }
        }

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        let row = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects WHERE id = ?"#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("SELECT error: {}", e)))?;

        match row {
            None => Ok(None),
            Some(row) => {
                let mut obj = Self::row_to_object(&row)?;
                obj.embedding = self.load_embedding(id).await;
                Ok(Some(obj))
            }
        }
    }

    async fn forget(&self, id: &str) -> Result<(), GeniusError> {
        self.delete_embedding(id).await?;

        sqlx::query("DELETE FROM memory_objects WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("DELETE memory_objects error: {}", e)))?;

        Ok(())
    }

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let type_json = Self::type_json(object_type);

        let rows = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects WHERE object_type = ?"#,
        )
        .bind(&type_json)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("SELECT by type error: {}", e)))?;

        rows.iter().map(Self::row_to_object).collect()
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError> {
        let rows = sqlx::query(
            r#"SELECT id, short_name, long_name, description, object_type,
                      content, metadata, created_at, updated_at
               FROM memory_objects"#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| GeniusError::MemoryError(format!("SELECT all error: {}", e)))?;

        rows.iter().map(Self::row_to_object).collect()
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        #[cfg(feature = "vec0")]
        sqlx::query("DELETE FROM memory_vec")
            .execute(&self.pool)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("DELETE memory_vec error: {}", e)))?;

        #[cfg(not(feature = "vec0"))]
        sqlx::query("DELETE FROM memory_embeddings")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                GeniusError::MemoryError(format!("DELETE memory_embeddings error: {}", e))
            })?;

        sqlx::query("DELETE FROM memory_objects")
            .execute(&self.pool)
            .await
            .map_err(|e| {
                GeniusError::MemoryError(format!("DELETE memory_objects error: {}", e))
            })?;

        Ok(())
    }
}
