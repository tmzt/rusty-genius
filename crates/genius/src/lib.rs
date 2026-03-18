use anyhow::Result;
use async_std::sync::Mutex;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, ContextInput, ContextOutput,
    InferenceEvent,
};
use rusty_genius_core::InMemoryContextStore;
use rusty_genius_stem::{ContextWorker, Orchestrator};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ── Resident context tracking ──

/// Purpose a model was loaded for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextPurpose {
    Infer,
    Embed,
}

impl ContextPurpose {
    fn as_str(&self) -> &'static str {
        match self {
            ContextPurpose::Infer => "infer",
            ContextPurpose::Embed => "embed",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "embed" => ContextPurpose::Embed,
            _ => ContextPurpose::Infer,
        }
    }
}

/// Composite key for the resident context map.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResidentKey {
    model: String,
    purpose: ContextPurpose,
}

/// Metadata about a resident (preloaded / kept-alive) context.
#[derive(Debug, Clone)]
pub struct ResidentContext {
    pub model: String,
    pub purpose: ContextPurpose,
    /// `None` = indefinite, `Some(n)` = n seconds from `loaded_at`.
    pub duration_secs: Option<u64>,
    pub loaded_at: Instant,
}

impl ResidentContext {
    /// Whether this entry has expired (only meaningful for timed residency).
    pub fn is_expired(&self) -> bool {
        match self.duration_secs {
            None => false,
            Some(d) => self.loaded_at.elapsed().as_secs() >= d,
        }
    }
}

// ── Genius facade ──

pub struct Genius {
    input_tx: mpsc::Sender<BrainstemInput>,
    output_rx: Arc<Mutex<mpsc::Receiver<BrainstemOutput>>>,
    context_tx: mpsc::Sender<ContextInput>,
    context_rx: Arc<Mutex<mpsc::Receiver<ContextOutput>>>,
    /// Tracks which model contexts are currently resident in the engine.
    resident: HashMap<ResidentKey, ResidentContext>,
}

impl Genius {
    pub async fn new() -> Result<Self> {
        Self::with_engine_name("default").await
    }

    /// Create a Genius instance with a specific engine backend.
    /// Supported names: "default"/"llama", "mlx" (if compiled with mlx feature).
    pub async fn with_engine_name(engine_name: &str) -> Result<Self> {
        let (input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        let mut orchestrator = if engine_name == "default" || engine_name.is_empty() {
            Orchestrator::new().await?
        } else {
            Orchestrator::with_engine_name(engine_name).await?
        };

        // Spawn the brainstem orchestrator
        async_std::task::spawn(async move {
            if let Err(e) = orchestrator.run(input_rx, output_tx).await {
                log::error!("orchestrator error: {}", e);
            }
        });

        // Set up context worker
        let (context_tx, context_input_rx) = mpsc::channel(100);
        let (context_output_tx, context_rx) = mpsc::channel(100);

        let store: Box<dyn rusty_genius_core::context::ContextStore> = Self::create_store().await?;
        let worker = ContextWorker::new(store);

        async_std::task::spawn(async move {
            worker.run(context_input_rx, context_output_tx).await;
        });

        Ok(Self {
            input_tx,
            output_rx: Arc::new(Mutex::new(output_rx)),
            context_tx,
            context_rx: Arc::new(Mutex::new(context_rx)),
            resident: HashMap::new(),
        })
    }

    #[cfg(feature = "redis-context")]
    async fn create_store() -> Result<Box<dyn rusty_genius_core::context::ContextStore>> {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let prefix = std::env::var("REDIS_CONTEXT_PREFIX").ok();
        match rusty_genius_stem::RedisContextStore::new(&url, prefix).await {
            Ok(store) => Ok(Box::new(store)),
            Err(e) => {
                log::warn!("failed to connect to Redis ({}), falling back to in-memory store", e);
                Ok(Box::new(InMemoryContextStore::new()))
            }
        }
    }

    #[cfg(not(feature = "redis-context"))]
    async fn create_store() -> Result<Box<dyn rusty_genius_core::context::ContextStore>> {
        Ok(Box::new(InMemoryContextStore::new()))
    }

    // ── Resident context tracking ──

    /// Check if a model is resident for the given purpose.
    pub fn is_resident(&self, model: &str, purpose: ContextPurpose) -> bool {
        let key = ResidentKey { model: model.to_string(), purpose };
        match self.resident.get(&key) {
            Some(ctx) => !ctx.is_expired(),
            None => false,
        }
    }

    /// Get metadata for a resident context, if it exists and hasn't expired.
    pub fn get_resident(&self, model: &str, purpose: ContextPurpose) -> Option<&ResidentContext> {
        let key = ResidentKey { model: model.to_string(), purpose };
        self.resident.get(&key).filter(|ctx| !ctx.is_expired())
    }

    /// Return all currently resident (non-expired) contexts.
    pub fn list_resident(&self) -> Vec<&ResidentContext> {
        self.resident.values().filter(|ctx| !ctx.is_expired()).collect()
    }

    /// Remove expired entries from the resident map.
    pub fn evict_expired(&mut self) {
        self.resident.retain(|_, ctx| !ctx.is_expired());
    }

    fn register_resident(
        &mut self,
        model: &str,
        purpose: ContextPurpose,
        duration_secs: Option<u64>,
    ) {
        let key = ResidentKey { model: model.to_string(), purpose };
        log::info!(
            "registering resident context: model={} purpose={} duration={:?}",
            model, purpose.as_str(), duration_secs,
        );
        self.resident.insert(key, ResidentContext {
            model: model.to_string(),
            purpose,
            duration_secs,
            loaded_at: Instant::now(),
        });
    }

    // ── Context store ──

    pub fn context_sender(&self) -> mpsc::Sender<ContextInput> {
        self.context_tx.clone()
    }

    pub fn context_receiver(&self) -> Arc<Mutex<mpsc::Receiver<ContextOutput>>> {
        self.context_rx.clone()
    }

    pub async fn context_send(&mut self, input: ContextInput) -> Result<()> {
        self.context_tx.send(input).await?;
        Ok(())
    }

    // ── Request ID helper ──

    fn make_request_id(tag: &str) -> String {
        format!(
            "facade-{}-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        )
    }

    /// Wait for a Complete or Error response matching `request_id`.
    async fn await_completion(&self, request_id: &str) -> Result<()> {
        let output_rx = self.output_rx.clone();
        let mut output_rx = output_rx.lock().await;
        let rid = request_id.to_string();
        while let Some(output) = output_rx.next().await {
            if output.id.as_deref() != Some(&rid) {
                continue;
            }
            match output.body {
                BrainstemBody::Event(InferenceEvent::Complete) => return Ok(()),
                BrainstemBody::Error(e) => return Err(anyhow::anyhow!("{}", e)),
                _ => {}
            }
        }
        Ok(())
    }

    // ── Preload / KeepResident ──

    /// Preload a model into engine memory so it's ready for immediate use.
    /// Registers the context as resident (indefinite, since preload has no
    /// built-in timeout — the orchestrator's strategy governs eviction).
    pub async fn preload(
        &mut self,
        model: String,
        purpose: String,
    ) -> Result<()> {
        let request_id = Self::make_request_id("preload");

        self.input_tx
            .send(BrainstemInput {
                id: Some(request_id.clone()),
                command: BrainstemCommand::PreloadModel {
                    model: model.clone(),
                    purpose: purpose.clone(),
                },
            })
            .await?;

        self.await_completion(&request_id).await?;

        // Track as resident — preloaded contexts stay until the orchestrator
        // evicts them (HibernateAfter timeout), so we don't set a duration.
        self.register_resident(&model, ContextPurpose::from_str(&purpose), None);

        Ok(())
    }

    /// Keep a model resident in engine memory.
    /// `duration_secs: None` keeps it alive forever; `Some(n)` keeps it
    /// resident for `n` seconds before reverting to hibernate.
    pub async fn keep_resident(
        &mut self,
        model: String,
        purpose: String,
        duration_secs: Option<u64>,
    ) -> Result<()> {
        let request_id = Self::make_request_id("keepresident");

        self.input_tx
            .send(BrainstemInput {
                id: Some(request_id.clone()),
                command: BrainstemCommand::KeepResident {
                    model: model.clone(),
                    purpose: purpose.clone(),
                    duration_secs,
                },
            })
            .await?;

        self.await_completion(&request_id).await?;

        self.register_resident(
            &model,
            ContextPurpose::from_str(&purpose),
            duration_secs,
        );

        Ok(())
    }

    // ── Infer / Embed ──

    pub async fn infer(
        &mut self,
        model: Option<String>,
        prompt: String,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<InferenceEvent>> {
        if let Some(ref m) = model {
            if !self.is_resident(m, ContextPurpose::Infer) {
                log::debug!("infer: model {} not resident, will be loaded on demand", m);
            }
        }

        let request_id = Self::make_request_id("chat");

        self.input_tx
            .send(BrainstemInput {
                id: Some(request_id.clone()),
                command: BrainstemCommand::Infer {
                    model,
                    prompt,
                    config,
                },
            })
            .await?;

        let (mut tx, rx) = mpsc::channel(100);
        let output_rx = self.output_rx.clone();

        async_std::task::spawn(async move {
            let mut output_rx = output_rx.lock().await;
            while let Some(output) = output_rx.next().await {
                if output.id != Some(request_id.clone()) {
                    continue;
                }

                match output.body {
                    BrainstemBody::Event(event) => {
                        if let InferenceEvent::Complete = event {
                            let _ = tx.send(event).await;
                            break;
                        }
                        let _ = tx.send(event).await;
                    }
                    BrainstemBody::Error(e) => {
                        log::error!("stream error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(rx)
    }

    pub async fn embed(
        &mut self,
        model: Option<String>,
        input: String,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<InferenceEvent>> {
        if let Some(ref m) = model {
            if !self.is_resident(m, ContextPurpose::Embed) {
                log::debug!("embed: model {} not resident, will be loaded on demand", m);
            }
        }

        let request_id = Self::make_request_id("embed");

        self.input_tx
            .send(BrainstemInput {
                id: Some(request_id.clone()),
                command: BrainstemCommand::Embed {
                    model,
                    input,
                    config,
                },
            })
            .await?;

        let (mut tx, rx) = mpsc::channel(100);
        let output_rx = self.output_rx.clone();

        async_std::task::spawn(async move {
            let mut output_rx = output_rx.lock().await;
            while let Some(output) = output_rx.next().await {
                if output.id != Some(request_id.clone()) {
                    continue;
                }

                match output.body {
                    BrainstemBody::Event(event) => {
                        if let InferenceEvent::Complete = event {
                            let _ = tx.send(event).await;
                            break;
                        }
                        let _ = tx.send(event).await;
                    }
                    BrainstemBody::Error(e) => {
                        log::error!("stream error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(rx)
    }
}
