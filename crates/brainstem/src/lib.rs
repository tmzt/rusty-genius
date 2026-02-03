use anyhow::Result;
use facecrab::AssetAuthority;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{BrainstemInput, BrainstemOutput};
use rusty_genius_cortex::{create_engine, Engine};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum CortexStrategy {
    Immediate,
    HibernateAfter(Duration),
    KeepAlive,
}

pub struct Orchestrator {
    engine: Box<dyn Engine>,
    asset_authority: AssetAuthority,
    strategy: CortexStrategy,
    last_activity: Instant,
}

impl Orchestrator {
    pub async fn new() -> Result<Self> {
        // In a real app, we might need configuration here
        let engine = create_engine().await;
        let asset_authority = AssetAuthority::new()?;
        Ok(Self {
            engine,
            asset_authority,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)), // Default 5 mins
            last_activity: Instant::now(),
        })
    }

    /// Run the main event loop
    /// Consumes BrainstemInput stream, produces BrainstemOutput stream
    pub async fn run(
        &mut self,
        mut input_rx: mpsc::Receiver<BrainstemInput>,
        mut output_tx: mpsc::Sender<BrainstemOutput>,
    ) -> Result<()> {
        loop {
            // Determine timeout based on strategy
            let timeout_duration = match self.strategy {
                CortexStrategy::HibernateAfter(duration) => Some(duration),
                CortexStrategy::Immediate => Some(Duration::ZERO), // Or very small
                CortexStrategy::KeepAlive => None,
            };

            let next_activity = if let Some(d) = timeout_duration {
                // Calculate when we should hibernate if no activity
                let elapsed = self.last_activity.elapsed();
                if elapsed >= d {
                    // Time to hibernate!
                    if let Err(e) = self.engine.unload_model().await {
                        eprintln!("Failed to hibernate engine: {}", e);
                    }
                    // Wait for next message indefinitely since we are hibernated/unloaded
                    None
                } else {
                    Some(d - elapsed)
                }
            } else {
                None
            };

            let msg_option = if let Some(wait_time) = next_activity {
                match async_std::future::timeout(wait_time, input_rx.next()).await {
                    Ok(msg) => msg,
                    Err(_) => {
                        // Timeout expired, loop back to check (should trigger hibernation)
                        continue;
                    }
                }
            } else {
                // Wait indefinitely
                input_rx.next().await
            };

            match msg_option {
                Some(msg) => {
                    self.last_activity = Instant::now(); // Update activity
                    match msg {
                        BrainstemInput::LoadModel(name_or_path) => {
                            // Try to resolve as a registry model first
                            // If ensure_model fails (e.g. not in registry), we assume it's a direct path
                            // Note: ensure_model returns Error if not found in registry currently.
                            // We might want to check if it's a file path first?
                            // For simplicity: Try registry, if error, treat as raw path.

                            let model_path =
                                match self.asset_authority.ensure_model(&name_or_path).await {
                                    Ok(path) => path.to_string_lossy().to_string(),
                                    Err(e) => {
                                        println!("Asset Authority Error: {:?}", e);
                                        name_or_path // Fallback to raw path string
                                    }
                                };

                            match self.engine.load_model(&model_path).await {
                                Ok(_) => {
                                    // Maybe send a success event?
                                }
                                Err(e) => {
                                    let _ =
                                        output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                                }
                            }
                        }
                        BrainstemInput::Infer { prompt, config: _ } => {
                            // Trigger inference
                            match self.engine.infer(&prompt).await {
                                Ok(mut event_rx) => {
                                    // Forward events to output
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if let Err(_) = output_tx
                                                    .send(BrainstemOutput::Event(event))
                                                    .await
                                                {
                                                    break; // Receiver dropped
                                                }
                                            }
                                            Err(e) => {
                                                let _ = output_tx
                                                    .send(BrainstemOutput::Error(e.to_string()))
                                                    .await;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ =
                                        output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                                }
                            }
                        }
                        BrainstemInput::Stop => {
                            break;
                        }
                    }
                }
                None => {
                    break; // Channel closed
                }
            }
        }
        Ok(())
    }
}
