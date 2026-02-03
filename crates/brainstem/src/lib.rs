use anyhow::{anyhow, Result};
use facecrab::AssetAuthority;
use rusty_genius_brain_cortex::{create_engine, Engine};
use rusty_genius_core::protocol::{BrainstemInput, BrainstemOutput, InferenceEvent};
use tokio::sync::mpsc;

pub struct Orchestrator {
    engine: Box<dyn Engine>,
    // asset_authority: AssetAuthority, // TODO: Initialize this properly
}

impl Orchestrator {
    pub async fn new() -> Result<Self> {
        // In a real app, we might need configuration here
        let engine = create_engine().await;
        // let asset_authority = AssetAuthority::new(...);
        Ok(Self { engine })
    }

    /// Run the main event loop
    /// Consumes BrainstemInput stream, produces BrainstemOutput stream
    pub async fn run(
        &mut self,
        mut input_rx: mpsc::Receiver<BrainstemInput>,
        output_tx: mpsc::Sender<BrainstemOutput>,
    ) -> Result<()> {
        while let Some(msg) = input_rx.recv().await {
            match msg {
                BrainstemInput::LoadModel(path) => {
                    // In reality, we'd use facecrab to resolve this path/hash first
                    match self.engine.load_model(&path).await {
                        Ok(_) => {
                            // Maybe send a success event?
                        }
                        Err(e) => {
                            let _ = output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                        }
                    }
                }
                BrainstemInput::Infer { prompt, config: _ } => {
                    // Trigger inference
                    match self.engine.infer(&prompt).await {
                        Ok(mut event_rx) => {
                            // Forward events to output
                            while let Some(event_res) = event_rx.recv().await {
                                match event_res {
                                    Ok(event) => {
                                        if let Err(_) = output_tx.send(BrainstemOutput::Event(event)).await {
                                            break; // Receiver dropped
                                        }
                                    }
                                    Err(e) => {
                                        let _ = output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                        }
                    }
                }
                BrainstemInput::Stop => {
                    break;
                }
            }
        }
        Ok(())
    }
}
