#![cfg(not(feature = "real-engine"))]

use rusty_genius_core::engine::Engine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{
    ChatContent, ChatMessage, ChatRole, InferenceEvent, ThoughtEvent, ToolCall, ToolDefinition,
};
use std::time::Duration;

#[derive(Default)]
pub struct Pinky {
    model_loaded: bool,
}

impl Pinky {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Engine for Pinky {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        smol::Timer::after(Duration::from_millis(100)).await;
        self.model_loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model_loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model_loaded
    }

    fn default_model(&self) -> String {
        "tiny-model".to_string()
    }

    async fn infer(
        &mut self,
        prompt: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let prompt_owned = prompt.to_string();
        eprintln!("DEBUG: Pinky::infer prompt: {}", prompt_owned);
        smol::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            smol::Timer::after(Duration::from_millis(50)).await;

            // Emit a "thought"
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                .await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                    "Narf!".to_string(),
                ))))
                .await;
            smol::Timer::after(Duration::from_millis(50)).await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                .await;

            // Emit content (echo prompt mostly)
            let _ = tx
                .send(Ok(InferenceEvent::Content(format!(
                    "Pinky says: {}",
                    prompt_owned
                ))))
                .await;

            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        })
        .detach();

        Ok(rx)
    }

    fn supports_tool_use(&self) -> bool {
        true
    }

    async fn infer_with_tools(
        &mut self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);

        // Check if conversation contains tool results — produce final answer
        let has_tool_results = messages.iter().any(|m| {
            matches!(m.role, ChatRole::Tool)
                && matches!(m.content, ChatContent::ToolResult(_))
        });

        if has_tool_results {
            // Summarize tool results into a final answer
            let mut summary_parts = Vec::new();
            for m in messages {
                if let ChatContent::ToolResult(ref r) = m.content {
                    if !r.is_error {
                        summary_parts.push(r.content.clone());
                    }
                }
            }
            let summary = if summary_parts.is_empty() {
                "Narf! The tools didn't return anything useful.".to_string()
            } else {
                format!(
                    "Based on what the tools told me: {}",
                    summary_parts.join("; ")
                )
            };

            smol::spawn(async move {
                let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
                let _ = tx
                    .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                    .await;
                let _ = tx
                    .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                        "Poit! I got the tool results back!".to_string(),
                    ))))
                    .await;
                let _ = tx
                    .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                    .await;
                let _ = tx.send(Ok(InferenceEvent::Content(summary))).await;
                let _ = tx.send(Ok(InferenceEvent::Complete)).await;
            })
            .detach();

            return Ok(rx);
        }

        // Extract last user text
        let user_text = messages
            .iter()
            .rev()
            .find_map(|m| match (&m.role, &m.content) {
                (ChatRole::User, ChatContent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let user_lower = user_text.to_lowercase();

        // Check for memory-related keywords and emit tool calls
        let memory_keywords: &[(&str, &str)] = &[
            ("remember", "memory_store"),
            ("store", "memory_store"),
            ("save", "memory_store"),
            ("recall", "memory_recall"),
            ("search", "memory_recall"),
            ("find", "memory_recall"),
            ("what do you", "memory_recall"),
            ("forget", "memory_forget"),
            ("delete", "memory_forget"),
            ("remove", "memory_forget"),
            ("list", "memory_list"),
        ];

        let matched_tool = memory_keywords
            .iter()
            .find(|(kw, _)| user_lower.contains(kw))
            .map(|(_, tool)| *tool);

        if let Some(tool_name) = matched_tool {
            if tools.iter().any(|t| t.name == tool_name) {
                let call_id = format!("call-{}", std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis());

                let arguments = match tool_name {
                    "memory_store" => {
                        // Extract content after "remember" / "store" / "save"
                        let content = extract_after_keyword(&user_text, &["remember that ", "remember ", "store ", "save "]);
                        let short = content
                            .split_whitespace()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join("-")
                            .to_lowercase();
                        serde_json::json!({
                            "short_name": short,
                            "content": content,
                            "object_type": "Fact",
                            "description": format!("Stored from user request: {}", truncate_str(&user_text, 60))
                        })
                    }
                    "memory_recall" => {
                        let query = extract_after_keyword(&user_text, &["recall ", "search for ", "search ", "find ", "what do you remember about ", "what do you remember", "what do you know about "]);
                        serde_json::json!({ "query": query, "limit": 5 })
                    }
                    "memory_forget" => {
                        let id = extract_after_keyword(&user_text, &["forget ", "delete ", "remove "]);
                        serde_json::json!({ "id": id.trim() })
                    }
                    "memory_list" => {
                        serde_json::json!({})
                    }
                    _ => serde_json::json!({}),
                };

                let tool_call = ToolCall {
                    id: call_id,
                    name: tool_name.to_string(),
                    arguments,
                };

                smol::spawn(async move {
                    let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
                    let _ = tx
                        .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                        .await;
                    let _ = tx
                        .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                            "Narf! I should use a tool for this!".to_string(),
                        ))))
                        .await;
                    let _ = tx
                        .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                        .await;
                    let _ = tx
                        .send(Ok(InferenceEvent::ToolUse(vec![tool_call])))
                        .await;
                    let _ = tx.send(Ok(InferenceEvent::Complete)).await;
                })
                .detach();

                return Ok(rx);
            }
        }

        // No tool match — fall back to echo behavior
        let prompt_owned = user_text;
        smol::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            smol::Timer::after(Duration::from_millis(50)).await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                .await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(
                    "Narf!".to_string(),
                ))))
                .await;
            smol::Timer::after(Duration::from_millis(50)).await;
            let _ = tx
                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                .await;
            let _ = tx
                .send(Ok(InferenceEvent::Content(format!(
                    "Pinky says: {}",
                    prompt_owned
                ))))
                .await;
            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        })
        .detach();

        Ok(rx)
    }

    async fn embed(
        &mut self,
        input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.model_loaded {
            return Err(anyhow!("Pinky Error: No model loaded!"));
        }

        let (mut tx, rx) = mpsc::channel(100);
        let input_owned = input.to_string();
        eprintln!("DEBUG: Pinky::embed input: {}", input_owned);
        smol::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            smol::Timer::after(Duration::from_millis(50)).await;

            // Generate a simple mock embedding (384 dimensions with random-ish values)
            let mock_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();

            let _ = tx.send(Ok(InferenceEvent::Embedding(mock_embedding))).await;
            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        })
        .detach();

        Ok(rx)
    }
}

/// Extract text after the first matching keyword prefix, or return the full input.
fn extract_after_keyword(text: &str, prefixes: &[&str]) -> String {
    let lower = text.to_lowercase();
    for prefix in prefixes {
        if let Some(pos) = lower.find(prefix) {
            let start = pos + prefix.len();
            let result = text[start..].trim().to_string();
            if !result.is_empty() {
                return result;
            }
        }
    }
    text.to_string()
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
