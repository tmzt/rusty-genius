mod engine_llamacpp;
mod engine_stub;

#[cfg(feature = "genai")]
mod engine_genai;

#[cfg(feature = "mlx")]
mod qwen35;
#[cfg(feature = "mlx")]
mod engine_mlx;

pub use rusty_genius_core::engine::Engine;

#[cfg(feature = "real-engine")]
pub use engine_llamacpp::LlamaCppEngine;

#[cfg(not(feature = "real-engine"))]
pub use engine_stub::Pinky;

#[cfg(feature = "genai")]
pub use engine_genai::{GeminiApiConfig, GeminiEngine};

#[cfg(feature = "genai")]
pub use engine_genai::{build_embed_body, build_infer_body, embed_url, infer_url, parse_sse_line};

#[cfg(feature = "mlx")]
pub use engine_mlx::MlxEngine;

// ── DispatchEngine ──

use anyhow::Result;
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;
use std::collections::HashMap;

/// A registered engine context, keyed by `{purpose}-{use_case}`.
///
/// Examples:
///   - `embed-embedding` → llama.cpp, nomic-embed-text
///   - `infer-router`    → llama.cpp, function-gemma
///   - `infer-thinker`   → MLX, qwen3.5-9b-mlx-4bit
struct EngineContext {
    /// Which engine slot owns this context.
    slot: String,
    /// Model ID (registry name, e.g. "nomic-embed-text").
    model: String,
    /// Whether this slot needs file path resolution (true for llama.cpp, false for MLX).
    needs_path_resolution: bool,
}

/// Composite engine that owns multiple backends and routes each operation
/// to the correct engine based on a `{purpose}-{use_case}` context key.
///
/// Callers pass model names through `infer(model, ...)` / `embed(model, ...)`.
/// The dispatch resolves the model name to a registered context, which
/// determines which engine slot handles the call.
pub struct DispatchEngine {
    /// Named engine backends (e.g. "llama", "mlx").
    slots: HashMap<String, Box<dyn Engine>>,
    /// Context key → engine context mapping.
    contexts: HashMap<String, EngineContext>,
    /// Reverse lookup: model ID → context key.
    model_index: HashMap<String, String>,
    /// Model ID → resolved file path (populated by the Orchestrator via set_model_path).
    path_cache: HashMap<String, String>,
    /// Slot name for models with no registered context.
    default_slot: String,
}

impl DispatchEngine {
    pub fn new(default_slot: &str) -> Self {
        Self {
            slots: HashMap::new(),
            contexts: HashMap::new(),
            model_index: HashMap::new(),
            path_cache: HashMap::new(),
            default_slot: default_slot.to_string(),
        }
    }

    /// Add a named engine backend.
    pub fn add_slot(&mut self, name: &str, engine: Box<dyn Engine>) {
        self.slots.insert(name.to_string(), engine);
    }

    /// Register a context: bind a use case to a specific engine and model.
    ///
    /// `key` is `{purpose}-{use_case}` (e.g. `"infer-thinker"`).
    /// `slot` is the engine backend name (e.g. `"mlx"` or `"llama"`).
    /// `model` is the model ID (e.g. `"nomic-embed-text"`).
    /// `needs_path` — if true, the engine needs a resolved file path (llama.cpp);
    ///   if false, the engine handles its own resolution (MLX).
    pub fn register(&mut self, key: &str, slot: &str, model: &str, needs_path: bool) {
        log::info!("dispatch: {} → slot={}, model={}, needs_path={}", key, slot, model, needs_path);
        self.model_index.insert(model.to_string(), key.to_string());
        self.contexts.insert(
            key.to_string(),
            EngineContext {
                slot: slot.to_string(),
                model: model.to_string(),
                needs_path_resolution: needs_path,
            },
        );
    }

    /// Record the resolved file path for a model ID (called by the Orchestrator
    /// after ensure_model_cached downloads/resolves the model).
    pub fn set_model_path(&mut self, model_id: &str, path: &str) {
        self.path_cache.insert(model_id.to_string(), path.to_string());
    }

    /// Resolve model ID → (engine slot, model string to pass to engine).
    /// For slots that need path resolution, returns the cached file path.
    /// For slots that handle their own resolution, returns the model ID.
    fn resolve_with_model<'a>(&'a mut self, model: Option<&str>) -> Result<(&'a mut Box<dyn Engine>, String)> {
        let model_id = model.unwrap_or(&self.default_slot);

        let (slot_name, needs_path) = self.model_index.get(model_id)
            .and_then(|key| self.contexts.get(key))
            .map(|ctx| (ctx.slot.as_str(), ctx.needs_path_resolution))
            .unwrap_or((&self.default_slot, true));

        let model_for_engine = if needs_path {
            self.path_cache.get(model_id)
                .cloned()
                .unwrap_or_else(|| model_id.to_string())
        } else {
            model_id.to_string()
        };

        let slot_name = slot_name.to_string();
        let engine = self.slots
            .get_mut(&slot_name)
            .ok_or_else(|| anyhow::anyhow!("no engine slot '{}'", slot_name))?;

        Ok((engine, model_for_engine))
    }
}

#[async_trait]
impl Engine for DispatchEngine {
    async fn load_model(&mut self, model_id: &str) -> Result<()> {
        let (engine, resolved) = self.resolve_with_model(Some(model_id))?;
        engine.load_model(&resolved).await
    }

    async fn unload_model(&mut self) -> Result<()> {
        for engine in self.slots.values_mut() {
            engine.unload_model().await?;
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.slots.values().any(|e| e.is_loaded())
    }

    fn default_model(&self) -> String {
        self.slots
            .get(&self.default_slot)
            .map(|e| e.default_model())
            .unwrap_or_else(|| "unknown".to_string())
    }

    async fn preload_model(&mut self, model_id: &str, purpose: &str) -> Result<()> {
        let (engine, resolved) = self.resolve_with_model(Some(model_id))?;
        engine.preload_model(&resolved, purpose).await
    }

    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let (engine, resolved) = self.resolve_with_model(model)?;
        engine.infer(Some(&resolved), prompt, config).await
    }

    async fn embed(
        &mut self,
        model: Option<&str>,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let (engine, resolved) = self.resolve_with_model(model)?;
        engine.embed(Some(&resolved), input, config).await
    }

    fn set_model_path(&mut self, model_id: &str, path: &str) {
        self.path_cache.insert(model_id.to_string(), path.to_string());
    }
}

// ── Engine factory functions ──

/// Create the default engine (llama.cpp or stub).
pub async fn create_engine() -> Box<dyn Engine> {
    #[cfg(feature = "real-engine")]
    {
        Box::new(LlamaCppEngine::new())
    }

    #[cfg(not(feature = "real-engine"))]
    {
        Box::new(Pinky::new())
    }
}

/// Create an engine by name.
///
/// For `"mlx"`, builds a `DispatchEngine` with registered contexts:
///   - `embed-embedding` → llama slot, nomic-embed-text
///   - `infer-router`    → llama slot, function-gemma
///   - `infer-thinker`   → mlx slot, Qwen3.5-9B-MLX-4bit
pub async fn create_engine_by_name(name: &str) -> Result<Box<dyn Engine>, String> {
    match name {
        #[cfg(feature = "mlx")]
        "mlx" => {
            use crate::backend::engine_mlx::MLX_DEFAULT_MODEL;

            let mlx = MlxEngine::new()
                .await
                .map_err(|e| format!("MLX engine init failed: {e}"))?;
            let llama = create_engine().await;

            let mut dispatch = DispatchEngine::new("llama");
            dispatch.add_slot("llama", llama);
            dispatch.add_slot("mlx", Box::new(mlx));

            dispatch.register("embed-embedding", "llama", "nomic-embed-text", true);
            dispatch.register(
                "infer-router",
                "llama",
                "lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf",
                true,
            );
            dispatch.register("infer-thinker", "mlx", MLX_DEFAULT_MODEL, false);

            Ok(Box::new(dispatch))
        }
        "llama" | "gguf" | "default" | "" => Ok(create_engine().await),
        other => Err(format!(
            "unknown engine: {other}. available: llama{}",
            if cfg!(feature = "mlx") { ", mlx" } else { "" }
        )),
    }
}
