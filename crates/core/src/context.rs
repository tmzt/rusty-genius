use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::error::GeniusError;

#[async_trait]
pub trait ContextStore: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<String>, GeniusError>;
    async fn set(&self, key: &str, value: &str) -> Result<(), GeniusError>;
    async fn delete(&self, key: &str) -> Result<(), GeniusError>;
    async fn list_keys(&self, pattern: &str) -> Result<Vec<String>, GeniusError>;
    async fn flush_all(&self) -> Result<(), GeniusError>;
}

pub struct InMemoryContextStore {
    data: RwLock<HashMap<String, String>>,
}

impl InMemoryContextStore {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryContextStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple glob-style pattern matching supporting `*` as wildcard.
fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
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

#[async_trait]
impl ContextStore for InMemoryContextStore {
    async fn get(&self, key: &str) -> Result<Option<String>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::Unknown(e.to_string()))?;
        Ok(data.get(key).cloned())
    }

    async fn set(&self, key: &str, value: &str) -> Result<(), GeniusError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::Unknown(e.to_string()))?;
        data.insert(key.to_string(), value.to_string());
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<(), GeniusError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::Unknown(e.to_string()))?;
        data.remove(key);
        Ok(())
    }

    async fn list_keys(&self, pattern: &str) -> Result<Vec<String>, GeniusError> {
        let data = self
            .data
            .read()
            .map_err(|e| GeniusError::Unknown(e.to_string()))?;
        let keys = data
            .keys()
            .filter(|k| glob_match(pattern, k))
            .cloned()
            .collect();
        Ok(keys)
    }

    async fn flush_all(&self) -> Result<(), GeniusError> {
        let mut data = self
            .data
            .write()
            .map_err(|e| GeniusError::Unknown(e.to_string()))?;
        data.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[async_std::test]
    async fn test_set_and_get() {
        let store = InMemoryContextStore::new();
        store.set("key1", "value1").await.unwrap();
        assert_eq!(store.get("key1").await.unwrap(), Some("value1".to_string()));
    }

    #[async_std::test]
    async fn test_get_missing_key() {
        let store = InMemoryContextStore::new();
        assert_eq!(store.get("nonexistent").await.unwrap(), None);
    }

    #[async_std::test]
    async fn test_delete() {
        let store = InMemoryContextStore::new();
        store.set("key1", "value1").await.unwrap();
        store.delete("key1").await.unwrap();
        assert_eq!(store.get("key1").await.unwrap(), None);
    }

    #[async_std::test]
    async fn test_list_keys_wildcard() {
        let store = InMemoryContextStore::new();
        store.set("user:1", "alice").await.unwrap();
        store.set("user:2", "bob").await.unwrap();
        store.set("session:1", "data").await.unwrap();

        let mut keys = store.list_keys("user:*").await.unwrap();
        keys.sort();
        assert_eq!(keys, vec!["user:1", "user:2"]);
    }

    #[async_std::test]
    async fn test_list_keys_exact() {
        let store = InMemoryContextStore::new();
        store.set("exact", "val").await.unwrap();
        store.set("exact2", "val2").await.unwrap();

        let keys = store.list_keys("exact").await.unwrap();
        assert_eq!(keys, vec!["exact"]);
    }

    #[async_std::test]
    async fn test_flush_all() {
        let store = InMemoryContextStore::new();
        store.set("a", "1").await.unwrap();
        store.set("b", "2").await.unwrap();
        store.flush_all().await.unwrap();
        assert_eq!(store.list_keys("*").await.unwrap().len(), 0);
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("user:*", "user:123"));
        assert!(!glob_match("user:*", "session:123"));
        assert!(glob_match("*:123", "user:123"));
        assert!(glob_match("u*r:*", "user:123"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "exact2"));
    }
}
