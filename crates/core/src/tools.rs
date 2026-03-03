use async_trait::async_trait;

use crate::error::GeniusError;
use crate::protocol::{ToolCall, ToolDefinition, ToolResult};

/// Trait for executing tool calls from an AI model.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait ToolExecutor: Send + Sync {
    /// Return the tool definitions this executor can handle.
    fn tool_definitions(&self) -> Vec<ToolDefinition>;

    /// Execute a single tool call and return its result.
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult, GeniusError>;

    /// Check if this executor handles a tool by name.
    fn handles(&self, name: &str) -> bool {
        self.tool_definitions().iter().any(|d| d.name == name)
    }
}

/// Combines multiple ToolExecutors, dispatching calls to the first that handles each tool.
pub struct CompositeToolExecutor {
    executors: Vec<Box<dyn ToolExecutor>>,
}

impl CompositeToolExecutor {
    pub fn new() -> Self {
        Self {
            executors: Vec::new(),
        }
    }

    pub fn add(mut self, executor: Box<dyn ToolExecutor>) -> Self {
        self.executors.push(executor);
        self
    }
}

impl Default for CompositeToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl ToolExecutor for CompositeToolExecutor {
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.executors
            .iter()
            .flat_map(|e| e.tool_definitions())
            .collect()
    }

    async fn execute(&self, call: &ToolCall) -> Result<ToolResult, GeniusError> {
        for executor in &self.executors {
            if executor.handles(&call.name) {
                return executor.execute(call).await;
            }
        }
        Ok(ToolResult {
            call_id: call.id.clone(),
            content: format!("Unknown tool: {}", call.name),
            is_error: true,
        })
    }

    fn handles(&self, name: &str) -> bool {
        self.executors.iter().any(|e| e.handles(name))
    }
}
