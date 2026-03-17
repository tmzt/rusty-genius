pub mod context_worker;
pub mod embedder;
#[cfg(feature = "redis-context")]
pub mod redis_store;
#[cfg(feature = "wllama")]
pub mod engine_wllama;

pub use context_worker::ContextWorker;
pub use embedder::BrainstemEmbedder;
#[cfg(feature = "redis-context")]
pub use redis_store::RedisContextStore;
#[cfg(feature = "wllama")]
pub use engine_wllama::WllamaEngine;

use anyhow::Result;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, ModelDescriptor,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "cortex-engine")]
use facecrab::AssetAuthority;
#[cfg(feature = "cortex-engine")]
use rusty_genius_core::protocol::AssetEvent;

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
    /// Maps model registry names to resolved file paths.
    model_paths: HashMap<String, String>,
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
            model_paths: HashMap::new(),
        })
    }

    #[cfg(all(feature = "wllama", not(feature = "cortex-engine")))]
    pub async fn new() -> Result<Self> {
        let engine = Box::new(WllamaEngine::new());
        Ok(Self {
            engine,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            last_model_name: None,
            model_paths: HashMap::new(),
        })
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
            model_paths: HashMap::new(),
        }
    }

    /// Resolve a model name to its file path. If already resolved, returns
    /// the cached path. Otherwise returns the name as-is (may be a path).
    fn resolve_model_path(&self, name: &str) -> String {
        self.model_paths.get(name).cloned().unwrap_or_else(|| name.to_string())
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
            self.model_paths.insert(name_or_path.clone(), path_to_load);
            self.last_model_name = Some(name_or_path);
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
        }
    }

    // ── Ensure model is in local file cache ──

    /// Resolve the model name to a local file path, ensuring it exists in
    /// the file cache. Does NOT load the model into the engine — the engine
    /// lazy-loads on first infer/embed call. Returns false if the model
    /// cannot be found or resolved.
    #[cfg(feature = "cortex-engine")]
    async fn ensure_model_cached(
        &mut self,
        model: Option<String>,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> bool {
        let model_name = model
            .or_else(|| self.last_model_name.clone())
            .unwrap_or_else(|| self.engine.default_model());

        // Already resolved — file path is known.
        if self.model_paths.contains_key(&model_name) {
            self.last_model_name = Some(model_name);
            return true;
        }

        // Resolve via asset authority (checks local cache, does NOT download).
        match self.asset_authority.ensure_model(&model_name).await {
            Ok(path) => {
                let path_str = path.to_str().unwrap().to_string();
                self.model_paths.insert(model_name.clone(), path_str);
                self.last_model_name = Some(model_name);
                true
            }
            Err(e) => {
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::Error(format!(
                            "Model '{}' not found in cache: {}",
                            model_name, e
                        )),
                    })
                    .await;
                false
            }
        }
    }

    #[cfg(not(feature = "cortex-engine"))]
    async fn ensure_model_cached(
        &mut self,
        model: Option<String>,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> bool {
        let model_name = model
            .or_else(|| self.last_model_name.clone())
            .unwrap_or_else(|| self.engine.default_model());

        if self.model_paths.contains_key(&model_name) {
            self.last_model_name = Some(model_name);
            return true;
        }

        // Without cortex-engine, assume the name is a valid path.
        self.model_paths.insert(model_name.clone(), model_name.clone());
        self.last_model_name = Some(model_name);
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
            .ensure_model_cached(model, request_id, output_tx)
            .await
        {
            return;
        }

        let resolved = self.last_model_name.as_ref().map(|n| self.resolve_model_path(n));
        match self.engine.infer(resolved.as_deref(), &prompt, config).await {
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
            .ensure_model_cached(model, request_id, output_tx)
            .await
        {
            return;
        }

        let resolved = self.last_model_name.as_ref().map(|n| self.resolve_model_path(n));
        match self.engine.embed(resolved.as_deref(), &input, config).await {
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
