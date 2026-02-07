use anyhow::Result;
use facecrab::AssetAuthority;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{
    AssetEvent, BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput,
};
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
        let engine = create_engine().await;
        let asset_authority = AssetAuthority::new()?;
        Ok(Self {
            engine,
            asset_authority,
            strategy: CortexStrategy::HibernateAfter(Duration::from_secs(300)),
            last_activity: Instant::now(),
            last_model_name: None,
        })
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
                        eprintln!("Failed to hibernate engine: {}", e);
                    }
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

                    match msg.command {
                        BrainstemCommand::LoadModel(name_or_path) => {
                            let mut events =
                                self.asset_authority.ensure_model_stream(&name_or_path);
                            let mut path_to_load = name_or_path.clone();

                            while let Some(event) = events.next().await {
                                if let AssetEvent::Complete(path) = &event {
                                    path_to_load = path.clone();
                                }
                                if output_tx
                                    .send(BrainstemOutput {
                                        id: Some(request_id.clone()),
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
                                        id: Some(request_id),
                                        body: BrainstemBody::Error(e.to_string()),
                                    })
                                    .await;
                            } else {
                                self.last_model_name = Some(name_or_path);
                            }
                        }
                        BrainstemCommand::Infer {
                            model,
                            prompt,
                            config,
                        } => {
                            if !self.engine.is_loaded() {
                                let model_to_load = model
                                    .clone()
                                    .or_else(|| self.last_model_name.clone())
                                    .unwrap_or_else(|| self.engine.default_model());

                                let model_name = model_to_load;
                                let start = Instant::now();
                                match self.asset_authority.ensure_model(&model_name).await {
                                    Ok(path) => {
                                        if let Err(e) =
                                            self.engine.load_model(path.to_str().unwrap()).await
                                        {
                                            let _ = output_tx
                                                .send(BrainstemOutput {
                                                    id: Some(request_id),
                                                    body: BrainstemBody::Error(format!(
                                                        "Cold reload failed: {}",
                                                        e
                                                    )),
                                                })
                                                .await;
                                            continue;
                                        }
                                        self.last_model_name = Some(model_name);
                                        println!(
                                            "NOTICE: Model reload took {:?}.",
                                            start.elapsed()
                                        );
                                    }
                                    Err(e) => {
                                        let _ = output_tx
                                            .send(BrainstemOutput {
                                                id: Some(request_id),
                                                body: BrainstemBody::Error(format!(
                                                    "Cold reload asset fail: {}",
                                                    e
                                                )),
                                            })
                                            .await;
                                        continue;
                                    }
                                }
                            }

                            match self.engine.infer(&prompt, config).await {
                                Ok(mut event_rx) => {
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx
                                                    .send(BrainstemOutput {
                                                        id: Some(request_id.clone()),
                                                        body: BrainstemBody::Event(event),
                                                    })
                                                    .await
                                                    .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                let _ = output_tx
                                                    .send(BrainstemOutput {
                                                        id: Some(request_id.clone()),
                                                        body: BrainstemBody::Error(e.to_string()),
                                                    })
                                                    .await;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output_tx
                                        .send(BrainstemOutput {
                                            id: Some(request_id),
                                            body: BrainstemBody::Error(e.to_string()),
                                        })
                                        .await;
                                }
                            }
                        }
                        BrainstemCommand::Embed {
                            model,
                            input,
                            config,
                        } => {
                            if !self.engine.is_loaded() {
                                let model_to_load = model
                                    .clone()
                                    .or_else(|| self.last_model_name.clone())
                                    .unwrap_or_else(|| self.engine.default_model());

                                let model_name = model_to_load;
                                let start = Instant::now();
                                match self.asset_authority.ensure_model(&model_name).await {
                                    Ok(path) => {
                                        if let Err(e) =
                                            self.engine.load_model(path.to_str().unwrap()).await
                                        {
                                            let _ = output_tx
                                                .send(BrainstemOutput {
                                                    id: Some(request_id),
                                                    body: BrainstemBody::Error(format!(
                                                        "Cold reload failed: {}",
                                                        e
                                                    )),
                                                })
                                                .await;
                                            continue;
                                        }
                                        self.last_model_name = Some(model_name);
                                        println!(
                                            "NOTICE: Model reload took {:?}.",
                                            start.elapsed()
                                        );
                                    }
                                    Err(e) => {
                                        let _ = output_tx
                                            .send(BrainstemOutput {
                                                id: Some(request_id),
                                                body: BrainstemBody::Error(format!(
                                                    "Cold reload asset fail: {}",
                                                    e
                                                )),
                                            })
                                            .await;
                                        continue;
                                    }
                                }
                            }

                            match self.engine.embed(&input, config).await {
                                Ok(mut event_rx) => {
                                    while let Some(event_res) = event_rx.next().await {
                                        match event_res {
                                            Ok(event) => {
                                                if output_tx
                                                    .send(BrainstemOutput {
                                                        id: Some(request_id.clone()),
                                                        body: BrainstemBody::Event(event),
                                                    })
                                                    .await
                                                    .is_err()
                                                {
                                                    break;
                                                }
                                            }
                                            Err(e) => {
                                                let _ = output_tx
                                                    .send(BrainstemOutput {
                                                        id: Some(request_id.clone()),
                                                        body: BrainstemBody::Error(e.to_string()),
                                                    })
                                                    .await;
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let _ = output_tx
                                        .send(BrainstemOutput {
                                            id: Some(request_id),
                                            body: BrainstemBody::Error(e.to_string()),
                                        })
                                        .await;
                                }
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
}
