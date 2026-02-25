use anyhow::Result;
use async_std::sync::Mutex;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{BrainstemCommand, BrainstemInput, BrainstemOutput, BrainstemBody};
use rusty_genius_stem::Orchestrator;
use rusty_genius_thinkerv1::{
    new_embed_request, new_ensure_request, new_inference_request, InferenceConfig, ModelConfig,
    Request, Response,
};
use rusty_genius_thinkerv1 as thinkerv1; // Alias it locally
use std::sync::Arc;

pub struct Genius {
    input_tx: mpsc::Sender<BrainstemInput>,
    output_rx: Arc<Mutex<mpsc::Receiver<BrainstemOutput>>>,
}

impl Genius {
    pub async fn new(default_unload_after: Option<u64>) -> Result<Self> {
        let (input_tx, input_rx) = mpsc::channel(100);
        let (output_tx, output_rx) = mpsc::channel(100);

        let mut orchestrator = Orchestrator::new(default_unload_after).await?;

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

    /// Ensures a model is loaded and ready, streaming status updates.
    pub async fn ensure_model_stream(
        &mut self,
        model: String,
        report_status: bool,
        model_config: Option<ModelConfig>,
    ) -> Result<mpsc::Receiver<Response>> {
        let request = new_ensure_request(
            model,
            report_status,
            model_config,
        );
        let request_id = request.get_id().to_string();

        self.input_tx
            .send(BrainstemInput {
                id: request_id.clone(),
                command: BrainstemCommand::EnsureModel(match request {
                    Request::Ensure(req) => req,
                    _ => unreachable!(), // new_ensure_request always returns Request::Ensure
                }),
            })
            .await?;

        let (mut tx, rx) = mpsc::channel(100);
        let output_rx_arc = self.output_rx.clone();

        async_std::task::spawn(async move {
            let mut output_rx = output_rx_arc.lock().await;
            while let Some(output) = output_rx.next().await {
                if output.id != request_id {
                    continue;
                }

                if let BrainstemBody::Thinker(response) = output.body {
                    let is_final = matches!(
                        &response,
                        Response::Status(s) if s.status == "ready" || s.status == "error"
                    );

                    if tx.send(response).await.is_err() {
                        // Receiver dropped, stop processing
                        break;
                    }

                    if is_final {
                        break;
                    }
                } else if let BrainstemBody::Error(e) = output.body {
                    eprintln!("Error from orchestrator for request {}: {}", request_id, e);
                    let _ = tx
                        .send(Response::Status(thinkerv1::StatusResponse {
                            id: request_id,
                            status: "error".to_string(),
                            progress: None,
                            message: Some(e),
                        }))
                        .await;
                    break;
                }
            }
        });

        Ok(rx)
    }

    /// Performs inference, streaming response events.
    pub async fn infer_stream(
        &mut self,
        prompt: String,
        inference_config: Option<InferenceConfig>,
    ) -> Result<mpsc::Receiver<Response>> {
        let request = new_inference_request(prompt, inference_config);
        let request_id = request.get_id().to_string();

        self.input_tx
            .send(BrainstemInput {
                id: request_id.clone(),
                command: BrainstemCommand::Inference(match request {
                    Request::Inference(req) => req,
                    _ => unreachable!(), // new_inference_request always returns Request::Inference
                }),
            })
            .await?;

        let (mut tx, rx) = mpsc::channel(100);
        let output_rx_arc = self.output_rx.clone();

        async_std::task::spawn(async move {
            let mut output_rx = output_rx_arc.lock().await;
            while let Some(output) = output_rx.next().await {
                if output.id != request_id {
                    continue;
                }

                if let BrainstemBody::Thinker(response) = output.body {
                    let is_final = matches!(
                        &response,
                        Response::Event(thinkerv1::EventResponse::Complete { .. })
                    );

                    if tx.send(response).await.is_err() {
                        // Receiver dropped, stop processing
                        break;
                    }

                    if is_final {
                        break;
                    }
                } else if let BrainstemBody::Error(e) = output.body {
                    eprintln!("Error from orchestrator for request {}: {}", request_id, e);
                    let _ = tx
                        .send(Response::Status(thinkerv1::StatusResponse {
                            id: request_id,
                            status: "error".to_string(),
                            progress: None,
                            message: Some(e),
                        }))
                        .await;
                    break;
                }
            }
        });

        Ok(rx)
    }

    /// Generates embeddings.
    pub async fn embed(&mut self, text: String) -> Result<mpsc::Receiver<Response>> {
        let request = new_embed_request(text);
        let request_id = request.get_id().to_string();

        self.input_tx
            .send(BrainstemInput {
                id: request_id.clone(),
                command: BrainstemCommand::Embed(match request {
                    Request::Embed(req) => req,
                    _ => unreachable!(), // new_embed_request always returns Request::Embed
                }),
            })
            .await?;

        let (mut tx, rx) = mpsc::channel(100);
        let output_rx_arc = self.output_rx.clone();

        async_std::task::spawn(async move {
            let mut output_rx = output_rx_arc.lock().await;
            while let Some(output) = output_rx.next().await {
                if output.id != request_id {
                    continue;
                }

                if let BrainstemBody::Thinker(response) = output.body {
                    let is_final = matches!(
                        &response,
                        Response::Event(thinkerv1::EventResponse::Embedding { .. })
                    );

                    if tx.send(response).await.is_err() {
                        // Receiver dropped, stop processing
                        break;
                    }

                    if is_final {
                        break;
                    }
                } else if let BrainstemBody::Error(e) = output.body {
                    eprintln!("Error from orchestrator for request {}: {}", request_id, e);
                    let _ = tx
                        .send(Response::Status(thinkerv1::StatusResponse {
                            id: request_id,
                            status: "error".to_string(),
                            progress: None,
                            message: Some(e),
                        }))
                        .await;
                    break;
                }
            }
        });

        Ok(rx)
    }
}
