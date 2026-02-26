#![cfg(feature = "redis-context")]

use async_trait::async_trait;
use redis::AsyncCommands;
use rusty_genius_core::context::ContextStore;
use rusty_genius_core::error::GeniusError;

pub struct RedisContextStore {
    connection: redis::aio::MultiplexedConnection,
    prefix: String,
}

impl RedisContextStore {
    pub async fn new(url: &str, prefix: Option<String>) -> Result<Self, GeniusError> {
        let client = redis::Client::open(url)
            .map_err(|e| GeniusError::Unknown(format!("Redis client error: {}", e)))?;
        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis connection error: {}", e)))?;
        Ok(Self {
            connection,
            prefix: prefix.unwrap_or_else(|| "rusty-genius:context:".to_string()),
        })
    }

    fn prefixed_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key)
    }

    fn strip_prefix<'a>(&self, key: &'a str) -> &'a str {
        key.strip_prefix(&self.prefix).unwrap_or(key)
    }
}

#[async_trait]
impl ContextStore for RedisContextStore {
    async fn get(&self, key: &str) -> Result<Option<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let result: Option<String> = conn
            .get(self.prefixed_key(key))
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis GET error: {}", e)))?;
        Ok(result)
    }

    async fn set(&self, key: &str, value: &str) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        conn.set::<_, _, ()>(self.prefixed_key(key), value)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis SET error: {}", e)))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        conn.del::<_, ()>(self.prefixed_key(key))
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis DEL error: {}", e)))?;
        Ok(())
    }

    async fn list_keys(&self, pattern: &str) -> Result<Vec<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let prefixed_pattern = self.prefixed_key(pattern);
        let keys: Vec<String> = conn
            .keys(prefixed_pattern)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis KEYS error: {}", e)))?;
        Ok(keys.iter().map(|k| self.strip_prefix(k).to_string()).collect())
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        let pattern = format!("{}*", self.prefix);
        let keys: Vec<String> = conn
            .keys(&pattern)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis KEYS error: {}", e)))?;
        if !keys.is_empty() {
            redis::cmd("DEL")
                .arg(&keys)
                .exec_async(&mut conn)
                .await
                .map_err(|e| GeniusError::Unknown(format!("Redis DEL error: {}", e)))?;
        }
        Ok(())
    }
}
