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
use std::sync::Arc;

pub struct Genius {
    input_tx: mpsc::Sender<BrainstemInput>,
    output_rx: Arc<Mutex<mpsc::Receiver<BrainstemOutput>>>,
    context_tx: mpsc::Sender<ContextInput>,
    context_rx: Arc<Mutex<mpsc::Receiver<ContextOutput>>>,
}

impl Genius {
    pub async fn new() -> Result<Self> {
        let (input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        let mut orchestrator = Orchestrator::new().await?;

        // Spawn the brainstem orchestrator
        async_std::task::spawn(async move {
            if let Err(e) = orchestrator.run(input_rx, output_tx).await {
                eprintln!("Orchestrator error: {}", e);
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
        })
    }

    #[cfg(feature = "redis-context")]
    async fn create_store() -> Result<Box<dyn rusty_genius_core::context::ContextStore>> {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let prefix = std::env::var("REDIS_CONTEXT_PREFIX").ok();
        match rusty_genius_stem::RedisContextStore::new(&url, prefix).await {
            Ok(store) => Ok(Box::new(store)),
            Err(e) => {
                eprintln!(
                    "WARN: Failed to connect to Redis ({}), falling back to in-memory store",
                    e
                );
                Ok(Box::new(InMemoryContextStore::new()))
            }
        }
    }

    #[cfg(not(feature = "redis-context"))]
    async fn create_store() -> Result<Box<dyn rusty_genius_core::context::ContextStore>> {
        Ok(Box::new(InMemoryContextStore::new()))
    }

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

    pub async fn infer(
        &mut self,
        model: Option<String>,
        prompt: String,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<InferenceEvent>> {
        let request_id = format!(
            "facade-chat-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        );

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
                // Ignore events for other request IDs
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
                        eprintln!("Error: {}", e);
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
        let request_id = format!(
            "facade-embed-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros()
        );

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
                // Ignore events for other request IDs
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
                        eprintln!("Error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(rx)
    }
}
