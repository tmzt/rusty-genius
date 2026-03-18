mod engine_llamacpp;
mod engine_stub;

#[cfg(feature = "genai")]
mod engine_genai;

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
    /// Model name/path to pass to the engine.
    model: String,
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
    /// Key format: "{purpose}-{use_case}" (e.g. "infer-thinker").
    contexts: HashMap<String, EngineContext>,
    /// Reverse lookup: model name → context key.
    model_index: HashMap<String, String>,
    /// Slot name for models with no registered context.
    default_slot: String,
}

impl DispatchEngine {
    pub fn new(default_slot: &str) -> Self {
        Self {
            slots: HashMap::new(),
            contexts: HashMap::new(),
            model_index: HashMap::new(),
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
    /// `model` is the model name/path the engine will load.
    pub fn register(&mut self, key: &str, slot: &str, model: &str) {
        log::info!("dispatch: {} → slot={}, model={}", key, slot, model);
        self.model_index.insert(model.to_string(), key.to_string());
        self.contexts.insert(
            key.to_string(),
            EngineContext {
                slot: slot.to_string(),
                model: model.to_string(),
            },
        );
    }

    /// Resolve model name → engine slot.
    fn resolve(&mut self, model: Option<&str>) -> Result<&mut Box<dyn Engine>> {
        let slot_name = model
            .and_then(|m| self.model_index.get(m))
            .and_then(|key| self.contexts.get(key))
            .map(|ctx| ctx.slot.as_str())
            .unwrap_or(&self.default_slot);

        self.slots
            .get_mut(slot_name)
            .ok_or_else(|| anyhow::anyhow!("no engine slot '{}'", slot_name))
    }
}

#[async_trait]
impl Engine for DispatchEngine {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        self.resolve(Some(model_path))?.load_model(model_path).await
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

    async fn preload_model(&mut self, model_path: &str, purpose: &str) -> Result<()> {
        self.resolve(Some(model_path))?
            .preload_model(model_path, purpose)
            .await
    }

    async fn infer(
        &mut self,
        model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        self.resolve(model)?.infer(model, prompt, config).await
    }

    async fn embed(
        &mut self,
        model: Option<&str>,
        input: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        self.resolve(model)?.embed(model, input, config).await
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

            dispatch.register("embed-embedding", "llama", "nomic-embed-text");
            dispatch.register(
                "infer-router",
                "llama",
                "lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf",
            );
            dispatch.register("infer-thinker", "mlx", MLX_DEFAULT_MODEL);

            Ok(Box::new(dispatch))
        }
        "llama" | "gguf" | "default" | "" => Ok(create_engine().await),
        other => Err(format!(
            "unknown engine: {other}. available: llama{}",
            if cfg!(feature = "mlx") { ", mlx" } else { "" }
        )),
    }
}
