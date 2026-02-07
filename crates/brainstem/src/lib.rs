use anyhow::Result;
use facecrab::AssetAuthority;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{AssetEvent, BrainstemInput, BrainstemOutput};
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
    last_model_name: Option<String>,
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
            last_model_name: None,
        })
    }

    pub fn set_strategy(&mut self, strategy: CortexStrategy) {
        self.strategy = strategy;
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
                            let mut events =
                                self.asset_authority.ensure_model_stream(&name_or_path);
                            let mut path_to_load = name_or_path.clone();

                            while let Some(event) = events.next().await {
                                if let AssetEvent::Complete(path) = &event {
                                    path_to_load = path.clone();
                                }
                                if output_tx.send(BrainstemOutput::Asset(event)).await.is_err() {
                                    break;
                                }
                            }

                            // Finally load into engine
                            if let Err(e) = self.engine.load_model(&path_to_load).await {
                                let _ = output_tx.send(BrainstemOutput::Error(e.to_string())).await;
                            } else {
                                self.last_model_name = Some(name_or_path);
                            }
                        }
                        BrainstemInput::Infer {
                            model,
                            prompt,
                            config,
                        } => {
                            // Cold Start Logic: If engine is not loaded
                            if !self.engine.is_loaded() {
                                // Prefer the model requested in the message, fallback to last_model_name, fallback to engine default
                                let model_to_load = model
                                    .or_else(|| self.last_model_name.clone())
                                    .unwrap_or_else(|| self.engine.default_model());

                                let model_name = model_to_load;
                                let start = Instant::now();
                                let model_path =
                                    self.asset_authority.ensure_model(&model_name).await;
                                match model_path {
                                    Ok(path) => {
                                        if let Err(e) =
                                            self.engine.load_model(path.to_str().unwrap()).await
                                        {
                                            let _ = output_tx
                                                .send(BrainstemOutput::Error(format!(
                                                    "Cold reload failed: {}",
                                                    e
                                                )))
                                                .await;
                                            continue;
                                        } else {
                                            self.last_model_name = Some(model_name);
                                            let duration = start.elapsed();
                                            println!("NOTICE: Model reload took {:?}. Increase --unload-after or use --no-unload to avoid delay.", duration);
                                        }
                                    }
                                    Err(e) => {
                                        let _ = output_tx
                                            .send(BrainstemOutput::Error(format!(
                                                "Cold reload asset resolution failed: {}",
                                                e
                                            )))
                                            .await;
                                        continue;
                                    }
                                }
                            }

                            // Trigger inference
                            match self.engine.infer(&prompt, config).await {
                                Ok(mut event_rx) => {
                                    // Forward events to output
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx
                                                    .send(BrainstemOutput::Event(event))
                                                    .await
                                                    .is_err()
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
                        BrainstemInput::Embed {
                            model,
                            input,
                            config,
                        } => {
                            // Cold Start Logic: Same as Infer
                            if !self.engine.is_loaded() {
                                let model_to_load = model
                                    .or_else(|| self.last_model_name.clone())
                                    .unwrap_or_else(|| self.engine.default_model());

                                let model_name = model_to_load;
                                let start = Instant::now();
                                let model_path =
                                    self.asset_authority.ensure_model(&model_name).await;
                                match model_path {
                                    Ok(path) => {
                                        if let Err(e) =
                                            self.engine.load_model(path.to_str().unwrap()).await
                                        {
                                            let _ = output_tx
                                                .send(BrainstemOutput::Error(format!(
                                                    "Cold reload failed: {}",
                                                    e
                                                )))
                                                .await;
                                            continue;
                                        } else {
                                            self.last_model_name = Some(model_name);
                                            let duration = start.elapsed();
                                            println!("NOTICE: Model reload took {:?}. Increase --unload-after or use --no-unload to avoid delay.", duration);
                                        }
                                    }
                                    Err(e) => {
                                        let _ = output_tx
                                            .send(BrainstemOutput::Error(format!(
                                                "Cold reload asset resolution failed: {}",
                                                e
                                            )))
                                            .await;
                                        continue;
                                    }
                                }
                            }

                            // Trigger embedding
                            match self.engine.embed(&input, config).await {
                                Ok(mut event_rx) => {
                                    // Forward events to output
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx
                                                    .send(BrainstemOutput::Event(event))
                                                    .await
                                                    .is_err()
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
