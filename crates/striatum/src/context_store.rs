use async_trait::async_trait;
use redis::AsyncCommands;
use rusty_genius_core::context::ContextStore;
use rusty_genius_core::error::GeniusError;

/// A Redis-backed [`ContextStore`] that stores all entries as fields in a
/// single Redis hash keyed by `prefix`.
///
/// Using a single hash avoids keyspace pollution and makes listing cheap:
/// - `list_keys` → `HKEYS {prefix}` (no SCAN/KEYS needed)
/// - `flush_all` → `DEL {prefix}` (single atomic delete)
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
            prefix: prefix.unwrap_or_else(|| "rusty-genius:context".to_string()),
        })
    }
}

/// Simple glob-style pattern matching: `*` matches any sequence of characters.
fn glob_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if pos > text.len() {
            return false;
        }
        match text[pos..].find(part) {
            Some(idx) => {
                if i == 0 && idx != 0 {
                    return false;
                }
                pos += idx + part.len();
            }
            None => return false,
        }
    }
    if let Some(last) = parts.last() {
        if !last.is_empty() {
            return text.ends_with(last);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", ""));
        assert!(glob_match("user:*", "user:123"));
        assert!(!glob_match("user:*", "session:123"));
        assert!(glob_match("*:123", "user:123"));
        assert!(glob_match("u*r:*", "user:123"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exact2"));
        assert!(!glob_match("exact", ""));
        assert!(glob_match("*a*", "bar"));
        assert!(!glob_match("*a*", "bor"));
        assert!(glob_match("pre*suf", "prefixsuf"));
        assert!(!glob_match("pre*suf", "prefix"));
    }
}

#[async_trait]
impl ContextStore for RedisContextStore {
    async fn get(&self, key: &str) -> Result<Option<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let result: Option<String> = conn
            .hget(&self.prefix, key)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis HGET error: {}", e)))?;
        Ok(result)
    }

    async fn set(&self, key: &str, value: &str) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        conn.hset::<_, _, _, ()>(&self.prefix, key, value)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis HSET error: {}", e)))?;
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        conn.hdel::<_, _, ()>(&self.prefix, key)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis HDEL error: {}", e)))?;
        Ok(())
    }

    async fn list_keys(&self, pattern: &str) -> Result<Vec<String>, GeniusError> {
        let mut conn = self.connection.clone();
        let all_keys: Vec<String> = conn
            .hkeys(&self.prefix)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis HKEYS error: {}", e)))?;
        Ok(all_keys
            .into_iter()
            .filter(|k| glob_match(pattern, k))
            .collect())
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        let mut conn = self.connection.clone();
        conn.del::<_, ()>(&self.prefix)
            .await
            .map_err(|e| GeniusError::Unknown(format!("Redis DEL error: {}", e)))?;
        Ok(())
    }
}
