use anyhow::Result;
use facecrab::AssetAuthority;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{BrainstemCommand, BrainstemInput, BrainstemOutput};
use rusty_genius_cortex::{create_engine, Engine};
use rusty_genius_thinkerv1::Response;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const DEFAULT_TTL_SECONDS: u64 = 300; // 5 minutes

pub struct Orchestrator {
    engine: Box<dyn Engine>,
    asset_authority: AssetAuthority,
    // Maps model name to its loaded path and last activity time
    loaded_models: HashMap<String, (String, Instant)>,
    // Default TTL for models, can be overridden by client requests
    default_unload_after: Duration,
    // Model-specific TTL overrides (from client requests), -1 for infinite
    model_ttl_overrides: HashMap<String, i64>,
    last_activity: Instant, // Last activity across all models
    current_active_model: Option<String>,
}

impl Orchestrator {
    pub async fn new(default_unload_after: Option<u64>) -> Result<Self> {
        let engine = create_engine().await;
        let asset_authority = AssetAuthority::new()?;
        Ok(Self {
            engine,
            asset_authority,
            loaded_models: HashMap::new(),
            default_unload_after: Duration::from_secs(default_unload_after.unwrap_or(DEFAULT_TTL_SECONDS)),
            model_ttl_overrides: HashMap::new(),
            last_activity: Instant::now(),
            current_active_model: None,
        })
    }

    pub async fn run(
        &mut self,
        mut input_rx: mpsc::Receiver<BrainstemInput>,
        mut output_tx: mpsc::Sender<BrainstemOutput>,
    ) -> Result<()> {
        loop {
            // Determine effective TTL for the current active model
            let effective_ttl = if let Some(model_name) = &self.current_active_model {
                match self.model_ttl_overrides.get(model_name) {
                    Some(-1) => None, // Infinite TTL
                    Some(ttl) if *ttl >= 0 => Some(Duration::from_secs(*ttl as u64)),
                    _ => Some(self.default_unload_after), // Use default
                }
            } else {
                Some(self.default_unload_after) // No model active, use default for any future model
            };

            let next_timeout = if let Some(ttl) = effective_ttl {
                let elapsed = self.last_activity.elapsed();
                if elapsed >= ttl {
                    // Time to unload
                    if self.engine.is_loaded() {
                        eprintln!("DEBUG: Unloading model due to inactivity.");
                        if let Err(e) = self.engine.unload_model().await {
                            eprintln!("Failed to unload engine: {}", e);
                        }
                        self.current_active_model = None;
                        self.loaded_models.clear(); // Clear info about unloaded model
                    }
                    Some(Duration::MAX) // Wait indefinitely for next message
                } else {
                    Some(ttl - elapsed) // Wait remaining time
                }
            } else {
                None // Infinite TTL for active model, wait indefinitely
            };

            let msg_option = if let Some(wait_time) = next_timeout {
                if wait_time == Duration::MAX {
                    input_rx.next().await // Wait without timeout
                } else {
                    match async_std::future::timeout(wait_time, input_rx.next()).await {
                        Ok(msg) => msg,
                        Err(_) => {
                            // Timeout occurred, loop to re-evaluate TTL and unload if necessary
                            continue;
                        }
                    }
                }
            } else {
                input_rx.next().await // No timeout (infinite TTL)
            };

            match msg_option {
                Some(input) => {
                    self.last_activity = Instant::now();
                    let request_id = input.id.clone();
                    eprintln!("DEBUG: [orchestrator] received command for [{}]: {:?}", request_id, input.command);

                    match input.command {
                        BrainstemCommand::EnsureModel(ensure_req) => {
                            // Store TTL override from client request
                            if let Some(mc) = &ensure_req.model_config {
                                if let Some(ttl) = mc.ttl_seconds {
                                    self.model_ttl_overrides.insert(ensure_req.model.clone(), ttl);
                                }
                            }

                            // facecrab's stream now returns thinkerv1::Response directly.
                            let mut asset_stream = self.asset_authority.ensure_model_stream(
                                request_id.clone(),
                                &ensure_req.model,
                                ensure_req.model_config.clone(),
                            );

                            let mut final_path: Option<String> = None;

                            while let Some(response) = asset_stream.next().await {
                                if let Response::Status(status) = &response {
                                    if status.status == "ready" {
                                        final_path = status.message.clone();
                                    }
                                }
                                // Forward the response from facecrab
                                if output_tx.send(BrainstemOutput::new_thinker(request_id.clone(), response)).await.is_err() {
                                    break; // Output channel closed
                                }
                            }

                            if let Some(path) = final_path {
                                // Load model into engine if not already loaded or if it's a different model
                                if self.current_active_model.as_ref() != Some(&ensure_req.model) || !self.engine.is_loaded() {
                                    if self.engine.is_loaded() {
                                        if let Err(e) = self.engine.unload_model().await {
                                            eprintln!("Error unloading previous model: {}", e);
                                        }
                                    }
                                    match self.engine.load_model(&path).await {
                                        Ok(_) => {
                                            self.current_active_model = Some(ensure_req.model.clone());
                                            self.loaded_models.insert(ensure_req.model.clone(), (path, Instant::now()));
                                            // The "ready" status was already sent by facecrab stream, no need to send another one.
                                        }
                                        Err(e) => {
                                            let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), format!("Failed to load model into engine: {}", e))).await;
                                            self.model_ttl_overrides.remove(&ensure_req.model);
                                        }
                                    }
                                }
                            } else {
                                // Error should have been streamed from facecrab, but as a fallback:
                                let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), "Failed to ensure model: no ready signal received.".to_string())).await;
                                self.model_ttl_overrides.remove(&ensure_req.model);
                            }
                        }
                        BrainstemCommand::Inference(inference_req) => {
                            // Ensure model is loaded first (cold start logic)
                            let model_name_to_use = self.current_active_model.clone().unwrap_or_else(|| self.engine.default_model());
                            // let mut model_path_to_use = String::new(); // Unused

                            if self.current_active_model.is_none() || self.current_active_model.as_ref() != Some(&model_name_to_use) || !self.engine.is_loaded() {
                                let start = Instant::now();
                                let internal_id = format!("internal-cold-start-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros());
                                match self.asset_authority.ensure_model(internal_id, &model_name_to_use, None).await { // No model_config for cold start if not explicitly provided
                                    Ok(path) => {
                                        if self.engine.is_loaded() {
                                            if let Err(e) = self.engine.unload_model().await {
                                                eprintln!("Error unloading previous model during cold start: {}", e);
                                            }
                                        }
                                        match self.engine.load_model(path.to_str().unwrap()).await {
                                            Ok(_) => {
                                                self.current_active_model = Some(model_name_to_use.clone());
                                                self.loaded_models.insert(model_name_to_use.clone(), (path.to_str().unwrap().to_string(), Instant::now()));
                                                eprintln!("NOTICE: Model reload took {:?}.", start.elapsed());
                                                // model_path_to_use = path.to_str().unwrap().to_string();
                                            }
                                            Err(e) => {
                                                let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), format!("Cold reload failed: {}", e))).await;
                                                continue;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), format!("Cold reload asset fail: {}", e))).await;
                                        continue;
                                    }
                                }
                            }

                            // Pass config from request, merging with any default engine config
                            let mut engine_config = rusty_genius_core::manifest::EngineConfig::default();
                            if let Some(inf_config) = inference_req.inference_config {
                                engine_config.show_thinking = inf_config.show_thinking;
                                engine_config.temperature = inf_config.temperature;
                                engine_config.top_p = inf_config.top_p;
                                engine_config.top_k = inf_config.top_k;
                                engine_config.repetition_penalty = inf_config.repetition_penalty;
                                engine_config.max_tokens = inf_config.max_tokens;
                            }
                            // Also merge ModelConfig if available from `loaded_models` or `model_ttl_overrides` etc.
                            // For simplicity, this initial implementation assumes the engine config is mostly for inference.
                            // More complex merging will be needed later if model config deeply affects inference.


                            match self.engine.infer(request_id.clone(), &inference_req.prompt, engine_config).await {
                                Ok(mut event_rx) => {
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx.send(BrainstemOutput::new_thinker(request_id.clone(), event)).await.is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), e.to_string())).await;
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), e.to_string())).await;
                                }
                            }
                        }
                        BrainstemCommand::Embed(embed_req) => {
                            // Ensure model is loaded first (cold start logic) - similar to inference
                            let model_name_to_use = self.current_active_model.clone().unwrap_or_else(|| self.engine.default_model());
                            // let mut model_path_to_use = String::new(); // Unused

                            if self.current_active_model.is_none() || self.current_active_model.as_ref() != Some(&model_name_to_use) || !self.engine.is_loaded() {
                                let start = Instant::now();
                                let internal_id = format!("internal-cold-start-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros());
                                match self.asset_authority.ensure_model(internal_id, &model_name_to_use, None).await {
                                    Ok(path) => {
                                        if self.engine.is_loaded() {
                                            if let Err(e) = self.engine.unload_model().await {
                                                eprintln!("Error unloading previous model during cold start: {}", e);
                                            }
                                        }
                                        match self.engine.load_model(path.to_str().unwrap()).await {
                                            Ok(_) => {
                                                self.current_active_model = Some(model_name_to_use.clone());
                                                self.loaded_models.insert(model_name_to_use.clone(), (path.to_str().unwrap().to_string(), Instant::now()));
                                                eprintln!("NOTICE: Model reload took {:?}.", start.elapsed());
                                                // model_path_to_use = path.to_str().unwrap().to_string();
                                            }
                                            Err(e) => {
                                                let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), format!("Cold reload failed: {}", e))).await;
                                                continue;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), format!("Cold reload asset fail: {}", e))).await;
                                        continue;
                                    }
                                }
                            }

                            // Pass config from request, merging with any default engine config
                            let engine_config = rusty_genius_core::manifest::EngineConfig::default(); // Embed doesn't have specific config yet

                            match self.engine.embed(request_id.clone(), &embed_req.text, engine_config).await {
                                Ok(mut event_rx) => {
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx.send(BrainstemOutput::new_thinker(request_id.clone(), event)).await.is_err() {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), e.to_string())).await;
                                                break;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), e.to_string())).await;
                                }
                            }
                        }
                        BrainstemCommand::Reset => {
                            if self.engine.is_loaded() {
                                if let Err(e) = self.engine.unload_model().await {
                                    let _ = output_tx.send(BrainstemOutput::new_error(request_id.clone(), e.to_string())).await;
                                } else {
                                    self.current_active_model = None;
                                    self.loaded_models.clear();
                                    self.model_ttl_overrides.clear();
                                    let _ = output_tx.send(BrainstemOutput::new_done(request_id)).await;
                                }
                            } else {
                                self.current_active_model = None;
                                self.loaded_models.clear();
                                self.model_ttl_overrides.clear();
                                let _ = output_tx.send(BrainstemOutput::new_done(request_id)).await;
                            }
                        }
                        BrainstemCommand::Shutdown => {
                            let _ = output_tx.send(BrainstemOutput::new_done(request_id)).await;
                            break;
                        }
                    }
                }
                None => {
                    // Input channel closed, terminate orchestrator
                    break;
                }
            }
        }
        Ok(())
    }
}
