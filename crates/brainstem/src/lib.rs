pub mod context_worker;
pub mod embedder;
pub mod memory_tools;
// Re-exported from striatum for backward compatibility; Redis access patterns
// live in rusty-genius-striatum.
#[cfg(feature = "redis-context")]
pub use rusty_genius_striatum::RedisContextStore;
#[cfg(feature = "wllama")]
pub mod engine_wllama;

pub use context_worker::ContextWorker;
pub use embedder::BrainstemEmbedder;
pub use memory_tools::MemoryToolExecutor;
#[cfg(feature = "wllama")]
pub use engine_wllama::WllamaEngine;

use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, ChatContent, ChatMessage,
    ChatRole, InferenceEvent, ModelDescriptor, ToolCall,
};
use rusty_genius_core::tools::ToolExecutor;
use std::time::{Duration, Instant};

#[cfg(feature = "cortex-engine")]
use facecrab::AssetAuthority;
#[cfg(feature = "cortex-engine")]
use rusty_genius_core::protocol::AssetEvent;

#[cfg(not(any(feature = "cortex-engine", feature = "wllama")))]
compile_error!(
    "rusty-genius-stem requires at least one engine feature: `cortex-engine` or `wllama`"
);

#[derive(Debug, Clone)]
pub enum CortexStrategy {
    Immediate,
    HibernateAfter(Duration),
    KeepAlive,
}

pub struct Orchestrator {
    engine: Box<dyn Engine>,
    #[cfg(feature = "cortex-engine")]
    asset_authority: AssetAuthority,
    strategy: CortexStrategy,
    last_activity: Instant,
    last_model_name: Option<String>,
    tool_executor: Option<Box<dyn ToolExecutor>>,
}

impl Orchestrator {
    #[cfg(feature = "cortex-engine")]
    pub async fn new() -> Result<Self> {
        let engine = rusty_genius_cortex::create_engine().await;
        let asset_authority = AssetAuthority::new()?;
        Ok(Self {
            engine,
            asset_authority,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            last_model_name: None,
            tool_executor: None,
        })
    }

    #[cfg(all(feature = "wllama", not(feature = "cortex-engine")))]
    pub async fn new() -> Result<Self> {
        Err(anyhow::anyhow!(
            "Use Orchestrator::with_engine(WllamaEngine::from_wasm_bytes(...)) to create a wllama-backed orchestrator"
        ))
    }

    /// Create an Orchestrator with a pre-built engine (useful for testing).
    pub fn with_engine(engine: Box<dyn Engine>) -> Self {
        Self {
            engine,
            #[cfg(feature = "cortex-engine")]
            asset_authority: AssetAuthority::new().expect("failed to create asset authority"),
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            last_model_name: None,
            tool_executor: None,
        }
    }

    /// Attach a tool executor for handling tool calls during inference.
    pub fn with_tool_executor(mut self, executor: Box<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    pub fn set_strategy(&mut self, strategy: CortexStrategy) {
        self.strategy = strategy;
    }

    pub async fn run(
        &mut self,
        mut input_rx: mpsc::Receiver<BrainstemInput>,
        mut output_tx: mpsc::Sender<BrainstemOutput>,
    ) -> Result<()> {
        loop {
            let timeout_duration = match self.strategy {
                CortexStrategy::HibernateAfter(duration) => Some(duration),
                CortexStrategy::Immediate => Some(Duration::ZERO),
                CortexStrategy::KeepAlive => None,
            };

            let next_activity = if let Some(d) = timeout_duration {
                let elapsed = self.last_activity.elapsed();
                if elapsed >= d {
                    if let Err(e) = self.engine.unload_model().await {
                        eprintln!("Failed to hibernate engine: {}", e);
                    }
                    None
                } else {
                    Some(d - elapsed)
                }
            } else {
                None
            };

            let msg_option = if let Some(wait_time) = next_activity {
                use futures::future::{self, Either};
                use futures_timer::Delay;

                let delay = Delay::new(wait_time);
                futures::pin_mut!(delay);
                let next = input_rx.next();
                futures::pin_mut!(next);
                match future::select(next, delay).await {
                    Either::Left((msg, _)) => msg,
                    Either::Right((_, _)) => {
                        continue;
                    }
                }
            } else {
                input_rx.next().await
            };

            match msg_option {
                Some(msg) => {
                    self.last_activity = Instant::now();
                    let request_id = msg.id.clone().unwrap_or_else(|| "anon".to_string());
                    eprintln!("DEBUG: [orchestrator] command: {:?}", msg.command);
                    eprintln!(
                        "DEBUG: [orchestrator] received command for [{}]: {:?}",
                        request_id, msg.command
                    );

                    match msg.command {
                        BrainstemCommand::LoadModel(name_or_path) => {
                            self.handle_load_model(name_or_path, &request_id, &mut output_tx)
                                .await;
                        }
                        BrainstemCommand::Infer {
                            model,
                            prompt,
                            config,
                        } => {
                            self.handle_infer(model, prompt, config, &request_id, &mut output_tx)
                                .await;
                        }
                        BrainstemCommand::Embed {
                            model,
                            input,
                            config,
                        } => {
                            self.handle_embed(model, input, config, &request_id, &mut output_tx)
                                .await;
                        }
                        BrainstemCommand::InferWithTools {
                            model,
                            messages,
                            tools,
                            config,
                        } => {
                            self.handle_infer_with_tools(
                                model,
                                messages,
                                tools,
                                config,
                                &request_id,
                                &mut output_tx,
                            )
                            .await;
                        }
                        BrainstemCommand::ListModels => {
                            self.handle_list_models(&request_id, &mut output_tx).await;
                        }
                        BrainstemCommand::Reset => {
                            if let Err(e) = self.engine.unload_model().await {
                                let _ = output_tx
                                    .send(BrainstemOutput {
                                        id: Some(request_id),
                                        body: BrainstemBody::Error(e.to_string()),
                                    })
                                    .await;
                            } else {
                                self.last_model_name = None;
                                let _ = output_tx
                                    .send(BrainstemOutput {
                                        id: Some(request_id),
                                        body: BrainstemBody::Event(
                                            rusty_genius_core::protocol::InferenceEvent::Complete,
                                        ),
                                    })
                                    .await;
                            }
                        }
                        BrainstemCommand::Stop => {
                            break;
                        }
                    }
                }
                None => {
                    break;
                }
            }
        }
        Ok(())
    }

    // ── LoadModel ──

    #[cfg(feature = "cortex-engine")]
    async fn handle_load_model(
        &mut self,
        name_or_path: String,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        let mut events = self.asset_authority.ensure_model_stream(&name_or_path);
        let mut path_to_load = name_or_path.clone();

        while let Some(event) = events.next().await {
            if let AssetEvent::Complete(path) = &event {
                path_to_load = path.clone();
            }
            if output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Asset(event),
                })
                .await
                .is_err()
            {
                break;
            }
        }

        if let Err(e) = self.engine.load_model(&path_to_load).await {
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Error(e.to_string()),
                })
                .await;
        } else {
            self.last_model_name = Some(name_or_path);
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Event(InferenceEvent::Complete),
                })
                .await;
        }
    }

    #[cfg(not(feature = "cortex-engine"))]
    async fn handle_load_model(
        &mut self,
        name_or_path: String,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        if let Err(e) = self.engine.load_model(&name_or_path).await {
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Error(e.to_string()),
                })
                .await;
        } else {
            self.last_model_name = Some(name_or_path);
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Event(InferenceEvent::Complete),
                })
                .await;
        }
    }

    // ── Ensure model loaded (cold reload) ──

    #[cfg(feature = "cortex-engine")]
    async fn ensure_model_loaded(
        &mut self,
        model: Option<String>,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> bool {
        if self.engine.is_loaded() {
            return true;
        }
        let model_to_load = model
            .or_else(|| self.last_model_name.clone())
            .unwrap_or_else(|| self.engine.default_model());

        let start = Instant::now();
        match self.asset_authority.ensure_model(&model_to_load).await {
            Ok(path) => {
                if let Err(e) = self.engine.load_model(path.to_str().unwrap()).await {
                    let _ = output_tx
                        .send(BrainstemOutput {
                            id: Some(request_id.to_string()),
                            body: BrainstemBody::Error(format!("Cold reload failed: {}", e)),
                        })
                        .await;
                    return false;
                }
                self.last_model_name = Some(model_to_load);
                eprintln!("NOTICE: Model reload took {:?}.", start.elapsed());
                true
            }
            Err(e) => {
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::Error(format!("Cold reload asset fail: {}", e)),
                    })
                    .await;
                false
            }
        }
    }

    #[cfg(not(feature = "cortex-engine"))]
    async fn ensure_model_loaded(
        &mut self,
        model: Option<String>,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> bool {
        if self.engine.is_loaded() {
            return true;
        }
        let model_to_load = model
            .or_else(|| self.last_model_name.clone())
            .unwrap_or_else(|| self.engine.default_model());

        if let Err(e) = self.engine.load_model(&model_to_load).await {
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Error(format!("Cold reload failed: {}", e)),
                })
                .await;
            return false;
        }
        self.last_model_name = Some(model_to_load);
        true
    }

    // ── Infer ──

    async fn handle_infer(
        &mut self,
        model: Option<String>,
        prompt: String,
        config: rusty_genius_core::manifest::InferenceConfig,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        if !self
            .ensure_model_loaded(model, request_id, output_tx)
            .await
        {
            return;
        }

        match self.engine.infer(&prompt, config).await {
            Ok(mut event_rx) => {
                while let Some(event_res) = event_rx.next().await {
                    match event_res {
                        Ok(event) => {
                            if output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Event(event),
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Error(e.to_string()),
                                })
                                .await;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::Error(e.to_string()),
                    })
                    .await;
            }
        }
    }

    // ── Embed ──

    async fn handle_embed(
        &mut self,
        model: Option<String>,
        input: String,
        config: rusty_genius_core::manifest::InferenceConfig,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        if !self
            .ensure_model_loaded(model, request_id, output_tx)
            .await
        {
            return;
        }

        match self.engine.embed(&input, config).await {
            Ok(mut event_rx) => {
                while let Some(event_res) = event_rx.next().await {
                    match event_res {
                        Ok(event) => {
                            if output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Event(event),
                                })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Error(e.to_string()),
                                })
                                .await;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::Error(e.to_string()),
                    })
                    .await;
            }
        }
    }

    // ── InferWithTools ──

    async fn handle_infer_with_tools(
        &mut self,
        model: Option<String>,
        mut messages: Vec<ChatMessage>,
        tools: Vec<rusty_genius_core::protocol::ToolDefinition>,
        config: rusty_genius_core::manifest::InferenceConfig,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        if !self
            .ensure_model_loaded(model, request_id, output_tx)
            .await
        {
            return;
        }

        let max_rounds = config.max_tool_rounds;

        for _round in 0..max_rounds {
            let event_rx_result = self
                .engine
                .infer_with_tools(&messages, &tools, config.clone())
                .await;

            let mut event_rx = match event_rx_result {
                Ok(rx) => rx,
                Err(e) => {
                    let _ = output_tx
                        .send(BrainstemOutput {
                            id: Some(request_id.to_string()),
                            body: BrainstemBody::Error(e.to_string()),
                        })
                        .await;
                    return;
                }
            };

            let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
            let mut got_tool_use = false;

            while let Some(event_res) = event_rx.next().await {
                match event_res {
                    Ok(InferenceEvent::ToolUse(calls)) => {
                        got_tool_use = true;
                        pending_tool_calls = calls.clone();
                        // Forward the ToolUse event to the caller
                        let _ = output_tx
                            .send(BrainstemOutput {
                                id: Some(request_id.to_string()),
                                body: BrainstemBody::Event(InferenceEvent::ToolUse(calls)),
                            })
                            .await;
                    }
                    Ok(event) => {
                        let _ = output_tx
                            .send(BrainstemOutput {
                                id: Some(request_id.to_string()),
                                body: BrainstemBody::Event(event),
                            })
                            .await;
                    }
                    Err(e) => {
                        let _ = output_tx
                            .send(BrainstemOutput {
                                id: Some(request_id.to_string()),
                                body: BrainstemBody::Error(e.to_string()),
                            })
                            .await;
                        return;
                    }
                }
            }

            if !got_tool_use || pending_tool_calls.is_empty() {
                // No tool calls — inference is complete
                return;
            }

            // Execute tool calls if we have an executor
            if let Some(ref executor) = self.tool_executor {
                // Append assistant message with tool calls to conversation
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: ChatContent::ToolCalls(pending_tool_calls.clone()),
                });

                // Execute each tool call and append results
                for call in &pending_tool_calls {
                    let result = match executor.execute(call).await {
                        Ok(r) => r,
                        Err(e) => rusty_genius_core::protocol::ToolResult {
                            call_id: call.id.clone(),
                            content: format!("Tool execution error: {}", e),
                            is_error: true,
                        },
                    };

                    // Forward tool result as an event
                    let _ = output_tx
                        .send(BrainstemOutput {
                            id: Some(request_id.to_string()),
                            body: BrainstemBody::Event(InferenceEvent::Content(format!(
                                "[tool_result:{}] {}",
                                result.call_id, result.content
                            ))),
                        })
                        .await;

                    messages.push(ChatMessage {
                        role: ChatRole::Tool,
                        content: ChatContent::ToolResult(result),
                    });
                }
                // Loop back to re-infer with tool results
            } else {
                // No executor — emit ToolCallRequest for external handling
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::ToolCallRequest {
                            session_id: request_id.to_string(),
                            calls: pending_tool_calls,
                        },
                    })
                    .await;
                return;
            }
        }

        // Exceeded max tool rounds
        let _ = output_tx
            .send(BrainstemOutput {
                id: Some(request_id.to_string()),
                body: BrainstemBody::Error(format!(
                    "Exceeded maximum tool call rounds ({})",
                    max_rounds
                )),
            })
            .await;
    }

    // ── ListModels ──

    #[cfg(feature = "cortex-engine")]
    async fn handle_list_models(
        &self,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        let models = self
            .asset_authority
            .list_models()
            .into_iter()
            .map(|m| ModelDescriptor {
                id: m.name,
                purpose: format!("{:?}", m.purpose),
            })
            .collect();
        let _ = output_tx
            .send(BrainstemOutput {
                id: Some(request_id.to_string()),
                body: BrainstemBody::ModelList(models),
            })
            .await;
    }

    #[cfg(not(feature = "cortex-engine"))]
    async fn handle_list_models(
        &self,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        let _ = output_tx
            .send(BrainstemOutput {
                id: Some(request_id.to_string()),
                body: BrainstemBody::ModelList(vec![]),
            })
            .await;
    }
}
