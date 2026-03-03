use async_trait::async_trait;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rusty_genius_core::error::GeniusError;
use rusty_genius_core::memory::{EmbeddingProvider, MemoryObject, MemoryObjectType, MemoryStore};
use rusty_genius_core::protocol::{ToolCall, ToolDefinition, ToolResult};
use rusty_genius_core::tools::ToolExecutor;

/// Executes memory-related tool calls against a MemoryStore + EmbeddingProvider.
pub struct MemoryToolExecutor {
    store: Arc<dyn MemoryStore>,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl MemoryToolExecutor {
    pub fn new(store: Arc<dyn MemoryStore>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self { store, embedder }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    async fn handle_store(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        let args = &call.arguments;
        let short_name = args["short_name"]
            .as_str()
            .unwrap_or("unnamed")
            .to_string();
        let content = args["content"].as_str().unwrap_or("").to_string();
        let description = args["description"].as_str().unwrap_or("").to_string();
        let object_type = match args["object_type"].as_str().unwrap_or("Fact") {
            "Entity" => MemoryObjectType::Entity,
            "Skill" => MemoryObjectType::Skill,
            "Observation" => MemoryObjectType::Observation,
            "Preference" => MemoryObjectType::Preference,
            "Relationship" => MemoryObjectType::Relationship,
            other if other.starts_with("Custom:") => {
                MemoryObjectType::Custom(other[7..].to_string())
            }
            _ => MemoryObjectType::Fact,
        };

        let now = Self::now_secs();
        let id = format!("mem-{}", now);

        let embedding = self.embedder.embed(&content).await.ok();

        let obj = MemoryObject {
            id: id.clone(),
            short_name: short_name.clone(),
            long_name: short_name,
            description,
            object_type,
            content,
            embedding,
            metadata: None,
            created_at: now,
            updated_at: now,
            ttl: None,
        };

        match self.store.store(obj).await {
            Ok(stored_id) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Stored memory with ID: {}", stored_id),
                is_error: false,
            }),
            Err(e) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Failed to store memory: {}", e),
                is_error: true,
            }),
        }
    }

    async fn handle_recall(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        let args = &call.arguments;
        let query = args["query"].as_str().unwrap_or("").to_string();
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        let embedding = self
            .embedder
            .embed(&query)
            .await
            .unwrap_or_else(|_| vec![]);

        match self.store.recall(&query, &embedding, limit, None).await {
            Ok(results) => {
                if results.is_empty() {
                    return Ok(ToolResult {
                        call_id: call.id.clone(),
                        content: "No memories found matching the query.".to_string(),
                        is_error: false,
                    });
                }
                let mut output = String::new();
                for obj in &results {
                    output.push_str(&format!(
                        "- [{}] {} ({}): {}\n",
                        obj.id,
                        obj.short_name,
                        format!("{:?}", obj.object_type),
                        truncate(&obj.content, 120),
                    ));
                }
                Ok(ToolResult {
                    call_id: call.id.clone(),
                    content: output,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Recall failed: {}", e),
                is_error: true,
            }),
        }
    }

    async fn handle_forget(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        let id = call.arguments["id"]
            .as_str()
            .unwrap_or("")
            .to_string();

        match self.store.forget(&id).await {
            Ok(()) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Deleted memory: {}", id),
                is_error: false,
            }),
            Err(e) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Failed to delete memory: {}", e),
                is_error: true,
            }),
        }
    }

    async fn handle_list(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        match self.store.list_all().await {
            Ok(items) => {
                if items.is_empty() {
                    return Ok(ToolResult {
                        call_id: call.id.clone(),
                        content: "No memories stored.".to_string(),
                        is_error: false,
                    });
                }
                let mut output = String::new();
                for obj in &items {
                    output.push_str(&format!(
                        "- [{}] {} ({:?}): {}\n",
                        obj.id,
                        obj.short_name,
                        obj.object_type,
                        truncate(&obj.content, 80),
                    ));
                }
                Ok(ToolResult {
                    call_id: call.id.clone(),
                    content: output,
                    is_error: false,
                })
            }
            Err(e) => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Failed to list memories: {}", e),
                is_error: true,
            }),
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl ToolExecutor for MemoryToolExecutor {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "memory_store".to_string(),
                description: "Store a new memory".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "short_name": { "type": "string", "description": "Short identifier for the memory" },
                        "content": { "type": "string", "description": "The content to remember" },
                        "object_type": { "type": "string", "enum": ["Fact", "Entity", "Skill", "Observation", "Preference", "Relationship"], "description": "Type of memory object" },
                        "description": { "type": "string", "description": "Brief description of the memory" }
                    },
                    "required": ["short_name", "content"]
                }),
            },
            ToolDefinition {
                name: "memory_recall".to_string(),
                description: "Search memories by query and embedding similarity".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" },
                        "limit": { "type": "integer", "description": "Maximum number of results", "default": 5 }
                    },
                    "required": ["query"]
                }),
            },
            ToolDefinition {
                name: "memory_forget".to_string(),
                description: "Delete a memory by ID".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "The memory ID to delete" }
                    },
                    "required": ["id"]
                }),
            },
            ToolDefinition {
                name: "memory_list".to_string(),
                description: "List all stored memories".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        ]
    }

    async fn execute(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        match call.name.as_str() {
            "memory_store" => self.handle_store(call).await,
            "memory_recall" => self.handle_recall(call).await,
            "memory_forget" => self.handle_forget(call).await,
            "memory_list" => self.handle_list(call).await,
            _ => Ok(ToolResult {
                call_id: call.id.clone(),
                content: format!("Unknown memory tool: {}", call.name),
                is_error: true,
            }),
        }
    }
}
