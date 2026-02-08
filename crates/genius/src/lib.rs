use anyhow::Result;
use async_std::sync::Mutex;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceEvent,
};
use rusty_genius_stem::Orchestrator;
use std::sync::Arc;

pub struct Genius {
    input_tx: mpsc::Sender<BrainstemInput>,
    output_rx: Arc<Mutex<mpsc::Receiver<BrainstemOutput>>>,
}

impl Genius {
    pub async fn new() -> Result<Self> {
        let (input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        let mut orchestrator = Orchestrator::new().await?;

        // Spawn the brainstem
        async_std::task::spawn(async move {
            if let Err(e) = orchestrator.run(input_rx, output_tx).await {
                eprintln!("Orchestrator error: {}", e);
            }
        });

        Ok(Self {
            input_tx,
            output_rx: Arc::new(Mutex::new(output_rx)),
        })
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
