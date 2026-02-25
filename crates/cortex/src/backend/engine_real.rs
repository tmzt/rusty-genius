#![cfg(feature = "real-engine")]

use crate::Engine;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use rusty_genius_core::manifest::EngineConfig;
use rusty_genius_thinkerv1::{EventResponse, Response};

#[cfg(feature = "llama-cpp-2")]
use llama_cpp_2::model::Model;
#[cfg(feature = "llama-cpp-2")]
use llama_cpp_2::context::Context;
#[cfg(feature = "llama-cpp-2")]
use std::path::PathBuf;

pub struct Brain {
    #[cfg(feature = "llama-cpp-2")]
    model: Option<Model>,
    #[cfg(feature = "llama-cpp-2")]
    context: Option<Context>,
}

impl Brain {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "llama-cpp-2")]
            model: None,
            #[cfg(feature = "llama-cpp-2")]
            context: None,
        }
    }
}

#[async_trait]
impl Engine for Brain {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        #[cfg(not(feature = "llama-cpp-2"))]
        {
            Err(anyhow!(
                "Cannot load model: 'llama-cpp-2' feature not enabled for Brain."
            ))
        }

        #[cfg(feature = "llama-cpp-2")]
        {
            // Placeholder for real llama-cpp-2 loading logic
            // Requires: `model::new` and `context::new`
            // For now, just simulate success.
            self.model = Some(todo!("Load Model from path")); // Replace with actual Model loading
            self.context = Some(todo!("Create Context from model and config")); // Replace with actual Context creation
            Ok(())
        }
    }

    async fn unload_model(&mut self) -> Result<()> {
        #[cfg(not(feature = "llama-cpp-2"))]
        {
            Ok(()) // Nothing to unload if feature not enabled
        }

        #[cfg(feature = "llama-cpp-2")]
        {
            self.model = None;
            self.context = None;
            Ok(())
        }
    }

    fn is_loaded(&self) -> bool {
        #[cfg(not(feature = "llama-cpp-2"))]
        {
            false
        }

        #[cfg(feature = "llama-cpp-2")]
        {
            self.model.is_some() && self.context.is_some()
        }
    }

    fn default_model(&self) -> String {
        "llama-2-7b-chat.Q4_K_M".to_string()
    }

    async fn infer(
        &mut self,
        id: String,
        prompt: &str,
        _config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<Response>>> {
        let (mut tx, rx) = mpsc::channel(100);
        let id_cloned = id.clone();

        #[cfg(not(feature = "llama-cpp-2"))]
        {
            let _ = tx.send(Err(anyhow!(
                "Cannot infer: 'llama-cpp-2' feature not enabled for Brain."
            ))).await;
        }

        #[cfg(feature = "llama-cpp-2")]
        {
            // Placeholder for real llama-cpp-2 inference logic
            // For now, simulate an immediate complete.
            let _ = tx.send(Ok(Response::Event(EventResponse::Content {
                id: id_cloned.clone(),
                content: format!("Brain says: I would infer '{}' but `llama-cpp-2` is a todo!", prompt),
            }))).await;
            let _ = tx.send(Ok(Response::Event(EventResponse::Complete { id: id_cloned }))).await;
        }
        Ok(rx)
    }

    async fn embed(
        &mut self,
        id: String,
        input: &str,
        _config: EngineConfig,
    ) -> Result<mpsc::Receiver<Result<Response>>> {
        let (mut tx, rx) = mpsc::channel(100);
        let id_cloned = id.clone();

        #[cfg(not(feature = "llama-cpp-2"))]
        {
            let _ = tx.send(Err(anyhow!(
                "Cannot embed: 'llama-cpp-2' feature not enabled for Brain."
            ))).await;
        }

        #[cfg(feature = "llama-cpp-2")]
        {
            // Placeholder for real llama-cpp-2 embedding logic
            let _ = tx.send(Ok(Response::Event(EventResponse::Embedding {
                id: id_cloned.clone(),
                vector_hex: format!("Embedding for '{}' is a todo!", input),
            }))).await;
            let _ = tx.send(Ok(Response::Event(EventResponse::Complete { id: id_cloned }))).await;
        }
        Ok(rx)
    }
}
