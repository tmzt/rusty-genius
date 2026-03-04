#![cfg(feature = "real-engine")]

use rusty_genius_core::engine::Engine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{
    ChatContent, ChatMessage, ChatRole, InferenceEvent, ThoughtEvent, ToolCall, ToolDefinition,
};
use futures::StreamExt;
use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn get_llama_backend() -> Arc<LlamaBackend> {
    LLAMA_BACKEND
        .get_or_init(|| Arc::new(LlamaBackend::init().expect("Failed to init llama backend")))
        .clone()
}

pub struct Brain {
    model: Option<Arc<LlamaModel>>,
    backend: Arc<LlamaBackend>,
    model_loaded: bool,
}

impl Brain {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self {
            model: None,
            backend: get_llama_backend(),
            model_loaded: false,
        }
    }
}

#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Load model
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| anyhow!("Failed to load model from {}: {}", model_path, e))?;
        self.model = Some(Arc::new(model));
        self.model_loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model_loaded = false;
        self.model = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn default_model(&self) -> String {
        "qwen-2.5-7b-instruct".to_string()
    }

    async fn infer(
        &mut self,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .clone();

        // Share the backend reference
        let backend = self.backend.clone();

        let prompt_str = prompt.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            // Send ProcessStart
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            // Use the shared backend (no re-init)
            let backend_ref = &backend;

            // Create context
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(config.context_size.and_then(|s| NonZeroU32::new(s)));

            let mut ctx = match model.new_context(backend_ref, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Context creation failed: {}", e))),
                    );
                    return;
                }
            };

            // Tokenize
            let tokens_list = match model.str_to_token(&prompt_str, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Tokenize failed: {}", e))),
                    );
                    return;
                }
            };

            // Prepare Batch for Prompt
            let n_tokens = tokens_list.len();
            let mut batch = LlamaBatch::new(2048, 1); // Ensure batch size can handle context

            // Load prompt into batch
            let last_index = n_tokens as i32 - 1;
            for (i, token) in tokens_list.iter().enumerate() {
                // add(token, pos, &[seq_id], logits)
                // We only need logits for the very last token to predict the next one
                let _ = batch.add(*token, i as i32, &[0], i as i32 == last_index);
            }

            // Decode Prompt
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(
                    tx.send(Err(anyhow!("Decode prompt failed: {}", e))),
                );
                return;
            }

            // Generation Loop
            let mut n_cur = n_tokens as i32;
            let n_decode = 0; // generated tokens count
            let max_tokens = 512; // Hard limit for safety

            let mut in_think_block = false;
            let mut token_str_buffer = String::new();

            loop {
                // Sample next token
                let mut sampler = LlamaSampler::greedy();
                let next_token = sampler.sample(&ctx, batch.n_tokens() - 1);

                // Decode token to string
                let token_str = match model.token_to_str(next_token, Special::Plaintext) {
                    Ok(s) => s.to_string(),
                    Err(_) => "??".to_string(),
                };

                // Check for EOS
                if next_token == model.token_eos() || n_decode >= max_tokens {
                    break;
                }

                // Parse Logic for <think> tags
                // Simple stream parsing
                token_str_buffer.push_str(&token_str);

                // If we are NOT in a think block, check if one is starting
                if !in_think_block && config.show_thinking {
                    if token_str_buffer.contains("<think>") {
                        in_think_block = true;
                        // Emit Start Thought event
                        let _ = futures::executor::block_on(
                            tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Start))),
                        );

                        // Remove <think> from buffer to find remainder
                        token_str_buffer = token_str_buffer.replace("<think>", "");
                    }
                }

                // If we ARE in a think block
                if in_think_block {
                    if token_str_buffer.contains("</think>") {
                        in_think_block = false;
                        // Emit Stop Thought event
                        let parts: Vec<&str> = token_str_buffer.split("</think>").collect();
                        if let Some(think_content) = parts.first() {
                            if !think_content.is_empty() {
                                let _ = futures::executor::block_on(tx.send(Ok(
                                    InferenceEvent::Thought(ThoughtEvent::Delta(
                                        think_content.to_string(),
                                    )),
                                )));
                            }
                        }

                        let _ = futures::executor::block_on(
                            tx.send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop))),
                        );

                        // Remainder after </think> should be content?
                        if parts.len() > 1 {
                            token_str_buffer = parts[1].to_string();
                            // Fallthrough to emit content
                        } else {
                            token_str_buffer.clear();
                        }
                    } else {
                        // Stream delta
                        if !token_str_buffer.is_empty() {
                            let _ =
                                futures::executor::block_on(tx.send(Ok(InferenceEvent::Thought(
                                    ThoughtEvent::Delta(token_str_buffer.clone()),
                                ))));
                            token_str_buffer.clear();
                        }
                    }
                }

                // If NOT in think block (anymore), emit as content
                if !in_think_block && !token_str_buffer.is_empty() {
                    let _ = futures::executor::block_on(
                        tx.send(Ok(InferenceEvent::Content(token_str_buffer.clone()))),
                    );
                    token_str_buffer.clear();
                }

                // Prepare next batch
                batch.clear();
                let _ = batch.add(next_token, n_cur, &[0], true);
                n_cur += 1;

                if let Err(e) = ctx.decode(&mut batch) {
                    let _ =
                        futures::executor::block_on(tx.send(Err(anyhow!("Decode failed: {}", e))));
                    break;
                }
            }

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        }))
        .detach();

        Ok(rx)
    }

    async fn embed(
        &mut self,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("No model loaded"))?
            .clone();

        let backend = self.backend.clone();
        let input_str = input.to_string();
        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(smol::unblock(move || {
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::ProcessStart)));

            let backend_ref = &backend;

            // Create context for embeddings
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(config.context_size.and_then(|s| NonZeroU32::new(s)))
                .with_embeddings(true); // Enable embedding mode

            let mut ctx = match model.new_context(backend_ref, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Context creation failed: {}", e))),
                    );
                    return;
                }
            };

            // Tokenize input
            let tokens_list = match model.str_to_token(&input_str, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Tokenize failed: {}", e))),
                    );
                    return;
                }
            };

            // Prepare batch
            let mut batch = LlamaBatch::new(2048, 1);

            // Add all tokens to batch (no need for logits in embedding mode)
            for (i, token) in tokens_list.iter().enumerate() {
                let _ = batch.add(*token, i as i32, &[0], false);
            }

            // Decode to get embeddings
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = futures::executor::block_on(tx.send(Err(anyhow!("Decode failed: {}", e))));
                return;
            }

            // Extract embeddings from the context
            let embeddings = match ctx.embeddings_seq_ith(0) {
                Ok(e) => e.to_vec(),
                Err(e) => {
                    let _ = futures::executor::block_on(
                        tx.send(Err(anyhow!("Failed to get embeddings from context: {}", e))),
                    );
                    return;
                }
            };

            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Embedding(embeddings))));
            let _ = futures::executor::block_on(tx.send(Ok(InferenceEvent::Complete)));
        }))
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
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let prompt = format_qwen_chat(messages, tools);
        let mut inner_rx = self.infer(&prompt, config).await?;

        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(async move {
            let mut content_buf = String::new();

            while let Some(event) = inner_rx.next().await {
                match event {
                    Ok(InferenceEvent::Content(text)) => {
                        content_buf.push_str(&text);
                        // Stream content through for responsiveness
                        let _ = tx.send(Ok(InferenceEvent::Content(text))).await;
                    }
                    Ok(InferenceEvent::Complete) => {
                        // Check accumulated content for tool calls
                        if content_buf.contains("<tool_call>") {
                            let calls = parse_tool_calls(&content_buf);
                            if !calls.is_empty() {
                                let _ =
                                    tx.send(Ok(InferenceEvent::ToolUse(calls))).await;
                            }
                        }
                        let _ = tx.send(Ok(InferenceEvent::Complete)).await;
                    }
                    other => {
                        let _ = tx.send(other).await;
                    }
                }
            }
        })
        .detach();

        Ok(rx)
    }
}

/// Format a conversation into the Qwen chat template with tool definitions.
fn format_qwen_chat(messages: &[ChatMessage], tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    // System message
    prompt.push_str("<|im_start|>system\n");

    let has_system = messages.iter().any(|m| matches!(m.role, ChatRole::System));
    if has_system {
        for msg in messages {
            if matches!(msg.role, ChatRole::System) {
                if let ChatContent::Text(t) = &msg.content {
                    prompt.push_str(t);
                    prompt.push('\n');
                }
            }
        }
    } else {
        prompt.push_str("You are a helpful assistant.\n");
    }

    // Tool definitions
    if !tools.is_empty() {
        prompt.push_str("\n# Tools\n\n");
        prompt.push_str(
            "You may call one or more functions to assist with the user query.\n\n",
        );
        prompt.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for tool in tools {
            let tool_json = serde_json::json!({
                "type": "function",
                "function": {
                    "name": &tool.name,
                    "description": &tool.description,
                    "parameters": &tool.parameters,
                }
            });
            prompt.push_str(
                &serde_json::to_string(&tool_json).unwrap_or_default(),
            );
            prompt.push('\n');
        }
        prompt.push_str("</tools>\n\n");
        prompt.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
        prompt.push_str("<tool_call>\n{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n</tool_call>\n");
    }

    prompt.push_str("<|im_end|>\n");

    // Conversation messages (skip system, already handled)
    for msg in messages {
        match msg.role {
            ChatRole::System => continue,
            ChatRole::User => {
                prompt.push_str("<|im_start|>user\n");
                if let ChatContent::Text(t) = &msg.content {
                    prompt.push_str(t);
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Assistant => {
                prompt.push_str("<|im_start|>assistant\n");
                match &msg.content {
                    ChatContent::Text(t) => prompt.push_str(t),
                    ChatContent::ToolCalls(calls) => {
                        for call in calls {
                            prompt.push_str("<tool_call>\n");
                            let call_json = serde_json::json!({
                                "name": &call.name,
                                "arguments": &call.arguments,
                            });
                            prompt.push_str(
                                &serde_json::to_string(&call_json).unwrap_or_default(),
                            );
                            prompt.push_str("\n</tool_call>\n");
                        }
                    }
                    _ => {}
                }
                prompt.push_str("<|im_end|>\n");
            }
            ChatRole::Tool => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                if let ChatContent::ToolResult(result) = &msg.content {
                    prompt.push_str(&result.content);
                }
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
        }
    }

    // Start the assistant turn
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Parse `<tool_call>...</tool_call>` blocks from model output into `ToolCall`s.
fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start_idx) = remaining.find("<tool_call>") {
        let after_tag = &remaining[start_idx + "<tool_call>".len()..];
        if let Some(end_idx) = after_tag.find("</tool_call>") {
            let inner = after_tag[..end_idx].trim();

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(inner) {
                let name = parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let arguments = parsed
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                calls.push(ToolCall {
                    id: format!("call-{}", calls.len()),
                    name,
                    arguments,
                });
            }

            remaining = &after_tag[end_idx + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    calls
}
