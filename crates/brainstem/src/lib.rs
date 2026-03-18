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
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceEvent,
    ModelDescriptor,
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

/// A resolved model entry in the local in-memory cache.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// Short registry name (e.g. "nomic-embed-text").
    pub short_name: String,
    /// GGUF filename on disk (e.g. "nomic-embed-text-v1.5.Q4_K_M.gguf").
    pub filename: String,
    /// Full resolved file path.
    pub path: String,
}

/// In-memory model cache. Allows lookup by short name, filename, or full
/// repo:filename spec. Once a model is resolved here, `ensure_model_cached`
/// is a noop. Persisted to `cache.toml` in the model cache directory.
#[derive(Debug)]
pub struct ModelCache {
    /// All cached entries.
    entries: Vec<CachedModel>,
    /// Index: any known key (short name, filename, repo:filename) → entry index.
    index: HashMap<String, usize>,
    /// Path to cache.toml for persistence.
    cache_file: Option<std::path::PathBuf>,
}

impl Default for ModelCache {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
            cache_file: None,
        }
    }
}

impl ModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a ModelCache backed by a cache.toml file. Loads existing
    /// entries from disk if the file exists.
    pub fn with_cache_dir(cache_dir: &std::path::Path) -> Self {
        let cache_file = cache_dir.join("cache.toml");
        let mut cache = Self {
            entries: Vec::new(),
            index: HashMap::new(),
            cache_file: Some(cache_file.clone()),
        };
        cache.load_from_disk();
        cache
    }

    /// Load cached entries from cache.toml.
    fn load_from_disk(&mut self) {
        let path = match &self.cache_file {
            Some(p) => p,
            None => return,
        };
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return,
        };
        // Simple TOML parsing: each [[models]] section has
        // short_name, filename, path.
        for section in content.split("[[models]]").skip(1) {
            let mut short_name = None;
            let mut filename = None;
            let mut file_path = None;
            for line in section.lines() {
                let line = line.trim();
                if let Some(val) = line.strip_prefix("short_name = \"") {
                    short_name = val.strip_suffix('"').map(|s| s.to_string());
                } else if let Some(val) = line.strip_prefix("filename = \"") {
                    filename = val.strip_suffix('"').map(|s| s.to_string());
                } else if let Some(val) = line.strip_prefix("path = \"") {
                    file_path = val.strip_suffix('"').map(|s| s.to_string());
                }
            }
            if let (Some(sn), Some(fn_), Some(fp)) = (short_name, filename, file_path) {
                // Only add if the file still exists on disk.
                if std::path::Path::new(&fp).exists() {
                    self.insert_no_persist(sn, fn_, fp);
                }
            }
        }
    }

    /// Write all entries to cache.toml.
    fn save_to_disk(&self) {
        let path = match &self.cache_file {
            Some(p) => p,
            None => return,
        };
        let mut out = String::new();
        for entry in &self.entries {
            out.push_str("[[models]]\n");
            out.push_str(&format!("short_name = \"{}\"\n", entry.short_name));
            out.push_str(&format!("filename = \"{}\"\n", entry.filename));
            out.push_str(&format!("path = \"{}\"\n\n", entry.path));
        }
        let _ = std::fs::write(path, out);
    }

    /// Insert without persisting (used during load_from_disk).
    fn insert_no_persist(&mut self, short_name: String, filename: String, path: String) {
        let idx = self.entries.len();
        self.index.insert(short_name.clone(), idx);
        self.index.insert(filename.clone(), idx);
        self.index.insert(path.clone(), idx);
        self.entries.push(CachedModel {
            short_name,
            filename,
            path,
        });
    }

    /// Insert a resolved model. Indexes by short_name, filename, and path.
    /// Persists to cache.toml.
    pub fn insert(&mut self, short_name: String, filename: String, path: String) {
        // Don't duplicate.
        if self.index.contains_key(&short_name) {
            return;
        }
        self.insert_no_persist(short_name, filename, path);
        self.save_to_disk();
    }

    /// Look up a model by any known key (short name, filename, or path).
    pub fn get(&self, key: &str) -> Option<&CachedModel> {
        self.index.get(key).map(|&idx| &self.entries[idx])
    }

    /// Get the resolved file path for a model key, if cached.
    pub fn resolve_path(&self, key: &str) -> Option<&str> {
        self.get(key).map(|e| e.path.as_str())
    }
}

pub struct Orchestrator {
    engine: Box<dyn Engine>,
    #[cfg(feature = "cortex-engine")]
    asset_authority: AssetAuthority,
    strategy: CortexStrategy,
    last_activity: Instant,
    /// Local in-memory model cache (name → file path).
    model_cache: ModelCache,
}

impl Orchestrator {
    #[cfg(feature = "cortex-engine")]
    pub async fn new() -> Result<Self> {
        let engine = rusty_genius_cortex::create_engine().await;
        let asset_authority = AssetAuthority::new()?;
        let cache_dir = asset_authority.registry().get_cache_dir();
        Ok(Self {
            engine,
            asset_authority,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            model_cache: ModelCache::with_cache_dir(&cache_dir),
        })
    }

    #[cfg(all(feature = "wllama", not(feature = "cortex-engine")))]
    pub async fn new() -> Result<Self> {
        let engine = Box::new(WllamaEngine::new());
        Ok(Self {
            engine,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            model_cache: ModelCache::new(),
        })
    }

    /// Create an Orchestrator with a named engine backend (e.g. "mlx", "llama").
    #[cfg(feature = "cortex-engine")]
    pub async fn with_engine_name(engine_name: &str) -> Result<Self> {
        let engine = rusty_genius_cortex::create_engine_by_name(engine_name)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let asset_authority = AssetAuthority::new()?;
        let cache_dir = asset_authority.registry().get_cache_dir();
        Ok(Self {
            engine,
            asset_authority,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            model_cache: ModelCache::with_cache_dir(&cache_dir),
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
            model_cache: ModelCache::new(),
        }
    }

    /// Resolve a model name to its file path via the in-memory cache.
    /// Falls back to the name as-is if not cached (may already be a path).
    fn resolve_model_path(&self, name: &str) -> String {
        self.model_cache
            .resolve_path(name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| name.to_string())
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
                        log::warn!("failed to hibernate engine: {}", e);
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
                    log::info!("command [{}]: {:?}", request_id, msg.command);

                    match msg.command {
                        BrainstemCommand::LoadModel(name_or_path) => {
                            self.handle_load_model(name_or_path, &request_id, &mut output_tx)
                                .await;
                        }
                        BrainstemCommand::PreloadModel { model, purpose } => {
                            self.handle_preload_model(model, purpose, &request_id, &mut output_tx)
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
                        BrainstemCommand::KeepResident {
                            model,
                            purpose,
                            duration_secs,
                        } => {
                            self.handle_keep_resident(
                                model,
                                purpose,
                                duration_secs,
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
            let filename = std::path::Path::new(&path_to_load)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| path_to_load.clone());
            self.model_cache.insert(name_or_path.clone(), filename, path_to_load);
            
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
            
        }
    }

    // ── Preload model into engine memory ──

    async fn handle_preload_model(
        &mut self,
        model: String,
        purpose: String,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        // Ensure model files are downloaded.
        if self.ensure_model_cached(&model, request_id, output_tx).await.is_none() {
            return;
        }

        if let Err(e) = self.engine.preload_model(&model, &purpose).await {
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Error(format!("Preload failed: {}", e)),
                })
                .await;
        } else {
            log::info!("preload complete: {} for {}", model, purpose);
            let _ = output_tx
                .send(BrainstemOutput {
                    id: Some(request_id.to_string()),
                    body: BrainstemBody::Event(InferenceEvent::Complete),
                })
                .await;
        }
    }

    // ── KeepResident: preload + pin strategy ──

    async fn handle_keep_resident(
        &mut self,
        model: String,
        purpose: String,
        duration_secs: Option<u64>,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) {
        // First, preload the model into the engine.
        self.handle_preload_model(model.clone(), purpose, request_id, output_tx)
            .await;

        // Switch strategy so the engine stays resident.
        match duration_secs {
            None => {
                self.strategy = CortexStrategy::KeepAlive;
                log::info!("keep-resident: {} pinned indefinitely", model);
            }
            Some(secs) => {
                self.strategy = CortexStrategy::HibernateAfter(Duration::from_secs(secs));
                self.last_activity = Instant::now();
                log::info!("keep-resident: {} pinned for {}s", model, secs);
            }
        }
    }

    // ── Ensure model is in local file cache ──

    /// Resolve the model name to a local file path, ensuring it exists in
    /// the file cache. Does NOT load the model into the engine — the engine
    /// lazy-loads on first infer/embed call. Returns false if the model
    /// cannot be found or resolved.
    /// Ensure a model is downloaded and cached. Returns the resolved file path
    /// for the model key, or sends an error event and returns None.
    #[cfg(feature = "cortex-engine")]
    async fn ensure_model_cached(
        &mut self,
        model_key: &str,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> Option<String> {
        // Already in the in-memory cache.
        if let Some(path) = self.model_cache.resolve_path(model_key) {
            self.engine.set_model_path(model_key, path);
            return Some(path.to_string());
        }

        // Resolve via asset authority (download if needed).
        match self.asset_authority.ensure_model(model_key).await {
            Ok(path) => {
                let path_str = path.to_str().unwrap().to_string();
                let filename = path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| path_str.clone());
                self.model_cache
                    .insert(model_key.to_string(), filename, path_str.clone());
                // Tell the engine about the resolved path so it can
                // map model IDs to file paths for backends that need it.
                self.engine.set_model_path(model_key, &path_str);
                Some(path_str)
            }
            Err(e) => {
                let _ = output_tx
                    .send(BrainstemOutput {
                        id: Some(request_id.to_string()),
                        body: BrainstemBody::Error(format!(
                            "Model '{}' not found: {}",
                            model_key, e
                        )),
                    })
                    .await;
                None
            }
        }
    }

    /// Non-cortex fallback: model key is used as-is (assumed to be a path).
    #[cfg(not(feature = "cortex-engine"))]
    async fn ensure_model_cached(
        &mut self,
        model_key: &str,
        request_id: &str,
        output_tx: &mut mpsc::Sender<BrainstemOutput>,
    ) -> Option<String> {
        if let Some(path) = self.model_cache.resolve_path(model_key) {
            return Some(path.to_string());
        }
        self.model_cache.insert(
            model_key.to_string(),
            model_key.to_string(),
            model_key.to_string(),
        );
        Some(model_key.to_string())
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
        let model_key = model.unwrap_or_else(|| self.engine.default_model());
        // Ensure model files are downloaded (facecrab caches them).
        // The engine receives the model key, not the resolved path —
        // each engine slot resolves paths internally.
        if self.ensure_model_cached(&model_key, request_id, output_tx).await.is_none() {
            return;
        }
        log::info!(
            "infer [{}] model={} prompt_len={}",
            request_id, model_key, prompt.len()
        );
        log::debug!("infer [{}] prompt: {}", request_id, prompt);
        match self.engine.infer(Some(&model_key), &prompt, config).await {
            Ok(mut event_rx) => {
                let mut response_len: usize = 0;
                while let Some(event_res) = event_rx.next().await {
                    match event_res {
                        Ok(event) => {
                            if let BrainstemBody::Event(InferenceEvent::Content(ref tok)) =
                                (BrainstemBody::Event(event.clone()))
                            {
                                response_len += tok.len();
                            }
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
                            log::error!("infer [{}] stream error: {}", request_id, e);
                            let _ = output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Error(e.to_string()),
                                })
                                .await;
                        }
                    }
                }
                log::info!("infer [{}] complete, response_len={}", request_id, response_len);
            }
            Err(e) => {
                log::error!("infer [{}] error: {}", request_id, e);
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
        let model_key = model.unwrap_or_else(|| self.engine.default_model());
        if self.ensure_model_cached(&model_key, request_id, output_tx).await.is_none() {
            return;
        }
        log::info!(
            "embed [{}] model={} input_len={}",
            request_id, model_key, input.len()
        );
        log::debug!("embed [{}] input: {}", request_id, input);
        match self.engine.embed(Some(&model_key), &input, config).await {
            Ok(mut event_rx) => {
                let mut got_embedding = false;
                while let Some(event_res) = event_rx.next().await {
                    match event_res {
                        Ok(event) => {
                            if let InferenceEvent::Embedding(_) = &event {
                                got_embedding = true;
                            }
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
                            log::error!("embed [{}] stream error: {}", request_id, e);
                            let _ = output_tx
                                .send(BrainstemOutput {
                                    id: Some(request_id.to_string()),
                                    body: BrainstemBody::Error(e.to_string()),
                                })
                                .await;
                        }
                    }
                }
                log::info!("embed [{}] complete, got_embedding={}", request_id, got_embedding);
            }
            Err(e) => {
                log::error!("embed [{}] error: {}", request_id, e);
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
