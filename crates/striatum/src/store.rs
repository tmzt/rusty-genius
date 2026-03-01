use async_trait::async_trait;
use redis::AsyncCommands;
use rusty_genius_core::error::GeniusError;
use rusty_genius_core::memory::{MemoryObject, MemoryObjectType, MemoryStore};

use crate::bootstrap::{self, RedisCapabilities, LUA_COSINE_SEARCH};

/// Incrementally collect keys matching `pattern` using SCAN (non-blocking).
async fn scan_keys(
    conn: &mut redis::aio::MultiplexedConnection,
    pattern: &str,
) -> Result<Vec<String>, GeniusError> {
    let mut keys = Vec::new();
    let mut cursor: u64 = 0;
    loop {
        let (next_cursor, batch): (u64, Vec<String>) = redis::cmd("SCAN")
            .arg(cursor)
            .arg("MATCH")
            .arg(pattern)
            .arg("COUNT")
            .arg(100)
            .query_async(conn)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis SCAN error: {}", e)))?;
        keys.extend(batch);
        cursor = next_cursor;
        if cursor == 0 {
            break;
        }
    }
    Ok(keys)
}

pub struct RedisMemoryStore {
    connection: redis::aio::MultiplexedConnection,
    prefix: String,
    capabilities: RedisCapabilities,
    cosine_script: redis::Script,
}

impl RedisMemoryStore {
    pub async fn new(url: &str, prefix: Option<String>) -> Result<Self, GeniusError> {
        let client = redis::Client::open(url)
            .map_err(|e| GeniusError::MemoryError(format!("Redis client error: {}", e)))?;
        let mut connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis connection error: {}", e)))?;

        let capabilities = bootstrap::detect_capabilities(&mut connection).await;
        let prefix = prefix.unwrap_or_else(|| "pfc".to_string());

        if capabilities.has_redisearch {
            bootstrap::create_redisearch_index(&mut connection, &prefix).await?;
        }

        Ok(Self {
            connection,
            prefix,
            capabilities,
            cosine_script: redis::Script::new(LUA_COSINE_SEARCH),
        })
    }

    fn obj_key(&self, id: &str) -> String {
        format!("{}:obj:{}", self.prefix, id)
    }

    fn vec_key(&self, id: &str) -> String {
        format!("{}:vec:{}", self.prefix, id)
    }

    /// Serialize a MemoryObjectType to a tag string for RediSearch TAG fields.
    fn type_tag(object_type: &MemoryObjectType) -> String {
        serde_json::to_string(object_type).unwrap_or_else(|_| "unknown".to_string())
    }

    /// Text search using RediSearch FT.SEARCH when available.
    async fn text_search_redisearch(
        &self,
        query: &str,
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let idx_name = format!("{}:idx", self.prefix);

        let mut ft_query = query.to_string();
        if let Some(ot) = object_type {
            let tag = Self::type_tag(ot);
            ft_query = format!("{} @object_type:{{{}}}", ft_query, tag);
        }

        let result: Vec<redis::Value> = redis::cmd("FT.SEARCH")
            .arg(&idx_name)
            .arg(&ft_query)
            .arg("LIMIT")
            .arg(0)
            .arg(limit)
            .arg("NOCONTENT")
            .query_async(&mut conn)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("FT.SEARCH error: {}", e)))?;

        // FT.SEARCH NOCONTENT returns: [total_count, key1, key2, ...]
        let obj_prefix = format!("{}:obj:", self.prefix);
        let mut ids = Vec::new();
        for item in result.iter().skip(1) {
            if let redis::Value::BulkString(bytes) = item {
                let key = String::from_utf8_lossy(bytes).to_string();
                let id = key.strip_prefix(&obj_prefix).unwrap_or(&key);
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }

    /// Fallback text search: SCAN + HGET + substring matching.
    async fn text_search_fallback(
        &self,
        query: &str,
        limit: usize,
        object_type: Option<&MemoryObjectType>,
    ) -> Result<Vec<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let pattern = format!("{}:obj:*", self.prefix);
        let keys = scan_keys(&mut conn, &pattern).await?;

        let query_lower = query.to_lowercase();
        let type_filter = object_type.map(Self::type_tag);
        let mut matched_ids = Vec::new();

        for key in &keys {
            let fields: Vec<Option<String>> = redis::cmd("HMGET")
                .arg(key)
                .arg("short_name")
                .arg("long_name")
                .arg("description")
                .arg("content")
                .arg("object_type")
                .query_async(&mut conn)
                .await
                .unwrap_or_default();

            if fields.len() < 5 {
                continue;
            }

            // Type filter
            if let Some(ref tf) = type_filter {
                if fields[4].as_deref().unwrap_or_default() != tf.as_str() {
                    continue;
                }
            }

            // Text match against short_name, long_name, description, content
            let matches = fields[..4]
                .iter()
                .any(|f| f.as_deref().unwrap_or_default().to_lowercase().contains(&query_lower));

            if matches {
                let id = key
                    .strip_prefix(&format!("{}:obj:", self.prefix))
                    .unwrap_or(key);
                matched_ids.push(id.to_string());
                if matched_ids.len() >= limit {
                    break;
                }
            }
        }

        Ok(matched_ids)
    }

    /// Load a MemoryObject from a Redis hash key.
    async fn load_object(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        let mut conn = self.connection.clone();
        let key = self.obj_key(id);

        let exists: bool = conn
            .exists(&key)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis EXISTS error: {}", e)))?;

        if !exists {
            return Ok(None);
        }

        let fields: Vec<Option<String>> = redis::cmd("HMGET")
            .arg(&key)
            .arg("id")
            .arg("short_name")
            .arg("long_name")
            .arg("description")
            .arg("object_type")
            .arg("content")
            .arg("metadata")
            .arg("created_at")
            .arg("updated_at")
            .arg("ttl")
            .query_async(&mut conn)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis HMGET error: {}", e)))?;

        let get = |idx: usize| -> String { fields[idx].clone().unwrap_or_default() };

        let object_type: MemoryObjectType =
            serde_json::from_str(&get(4)).map_err(|e| {
                GeniusError::MemoryError(format!("Failed to deserialize object_type: {}", e))
            })?;

        // Load embedding vector if present
        let vec_key = self.vec_key(id);
        let embedding: Option<Vec<f32>> = conn
            .get::<_, Option<String>>(&vec_key)
            .await
            .ok()
            .flatten()
            .and_then(|s| serde_json::from_str(&s).ok());

        let ttl: Option<u64> = fields[9]
            .as_ref()
            .and_then(|s| s.parse().ok());

        Ok(Some(MemoryObject {
            id: get(0),
            short_name: get(1),
            long_name: get(2),
            description: get(3),
            object_type,
            content: get(5),
            embedding,
            metadata: fields[6].clone(),
            created_at: get(7).parse().unwrap_or(0),
            updated_at: get(8).parse().unwrap_or(0),
            ttl,
        }))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl MemoryStore for RedisMemoryStore {
    async fn store(&self, object: MemoryObject) -> Result<String, GeniusError> {
        let mut conn = self.connection.clone();
        let obj_key = self.obj_key(&object.id);

        let type_json = serde_json::to_string(&object.object_type)
            .map_err(|e| GeniusError::MemoryError(format!("Serialize error: {}", e)))?;

        let ttl_str = object.ttl.map(|t| t.to_string()).unwrap_or_default();

        redis::cmd("HSET")
            .arg(&obj_key)
            .arg("id")
            .arg(&object.id)
            .arg("short_name")
            .arg(&object.short_name)
            .arg("long_name")
            .arg(&object.long_name)
            .arg("description")
            .arg(&object.description)
            .arg("object_type")
            .arg(&type_json)
            .arg("content")
            .arg(&object.content)
            .arg("metadata")
            .arg(object.metadata.as_deref().unwrap_or(""))
            .arg("created_at")
            .arg(object.created_at.to_string())
            .arg("updated_at")
            .arg(object.updated_at.to_string())
            .arg("ttl")
            .arg(&ttl_str)
            .query_async::<redis::Value>(&mut conn)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis HSET error: {}", e)))?;

        // Store embedding vector as JSON string
        if let Some(ref embedding) = object.embedding {
            let vec_key = self.vec_key(&object.id);
            let vec_json = serde_json::to_string(embedding)
                .map_err(|e| GeniusError::MemoryError(format!("Serialize vec error: {}", e)))?;
            conn.set::<_, _, ()>(&vec_key, &vec_json)
                .await
                .map_err(|e| GeniusError::MemoryError(format!("Redis SET vec error: {}", e)))?;

            // Set TTL on vec key if applicable
            if let Some(ttl) = object.ttl {
                conn.expire::<_, ()>(&vec_key, ttl as i64)
                    .await
                    .map_err(|e| {
                        GeniusError::MemoryError(format!("Redis EXPIRE vec error: {}", e))
                    })?;
            }
        }

        // Set TTL on obj key if applicable
        if let Some(ttl) = object.ttl {
            conn.expire::<_, ()>(&obj_key, ttl as i64)
                .await
                .map_err(|e| GeniusError::MemoryError(format!("Redis EXPIRE error: {}", e)))?;
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
        // Combine text search + vector search results
        let text_ids = if self.capabilities.has_redisearch {
            self.text_search_redisearch(query, limit, object_type).await?
        } else {
            self.text_search_fallback(query, limit, object_type).await?
        };
        let vec_ids = self
            .recall_by_vector(embedding, limit, object_type)
            .await?
            .into_iter()
            .map(|o| o.id)
            .collect::<Vec<_>>();

        // Merge and deduplicate, vector results first
        let mut seen = std::collections::HashSet::new();
        let mut merged_ids = Vec::new();
        for id in vec_ids.iter().chain(text_ids.iter()) {
            if seen.insert(id.clone()) {
                merged_ids.push(id.clone());
            }
        }
        merged_ids.truncate(limit);

        let mut results = Vec::new();
        for id in &merged_ids {
            if let Some(obj) = self.load_object(id).await? {
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
        let mut conn = self.connection.clone();
        let query_json = serde_json::to_string(embedding)
            .map_err(|e| GeniusError::MemoryError(format!("Serialize query vec error: {}", e)))?;

        let result: Vec<String> = self
            .cosine_script
            .key(&self.prefix)
            .arg(&query_json)
            .arg(limit)
            .invoke_async(&mut conn)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Lua cosine search error: {}", e)))?;

        // Result is alternating [id, score, id, score, ...]
        let type_filter = object_type.map(Self::type_tag);
        let mut objects = Vec::new();
        let mut i = 0;
        while i + 1 < result.len() && objects.len() < limit {
            let id = &result[i];
            // skip score at result[i+1]
            if let Some(obj) = self.load_object(id).await? {
                if let Some(ref tf) = type_filter {
                    let obj_tag = Self::type_tag(&obj.object_type);
                    if &obj_tag != tf {
                        i += 2;
                        continue;
                    }
                }
                objects.push(obj);
            }
            i += 2;
        }

        Ok(objects)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryObject>, GeniusError> {
        self.load_object(id).await
    }

    async fn forget(&self, id: &str) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        let obj_key = self.obj_key(id);
        let vec_key = self.vec_key(id);

        conn.del::<_, ()>(&obj_key)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis DEL obj error: {}", e)))?;
        conn.del::<_, ()>(&vec_key)
            .await
            .map_err(|e| GeniusError::MemoryError(format!("Redis DEL vec error: {}", e)))?;

        Ok(())
    }

    async fn list_by_type(
        &self,
        object_type: &MemoryObjectType,
    ) -> Result<Vec<MemoryObject>, GeniusError> {
        let all = self.list_all().await?;
        let type_tag = Self::type_tag(object_type);
        Ok(all
            .into_iter()
            .filter(|o| Self::type_tag(&o.object_type) == type_tag)
            .collect())
    }

    async fn list_all(&self) -> Result<Vec<MemoryObject>, GeniusError> {
        let mut conn = self.connection.clone();
        let pattern = format!("{}:obj:*", self.prefix);
        let keys = scan_keys(&mut conn, &pattern).await?;

        let mut objects = Vec::new();
        for key in &keys {
            let id = key
                .strip_prefix(&format!("{}:obj:", self.prefix))
                .unwrap_or(key);
            if let Some(obj) = self.load_object(id).await? {
                objects.push(obj);
            }
        }

        Ok(objects)
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        let pattern = format!("{}:*", self.prefix);
        let keys = scan_keys(&mut conn, &pattern).await?;

        if !keys.is_empty() {
            redis::cmd("DEL")
                .arg(&keys)
                .exec_async(&mut conn)
                .await
                .map_err(|e| GeniusError::MemoryError(format!("Redis DEL error: {}", e)))?;
        }

        Ok(())
    }
}
