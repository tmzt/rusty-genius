//! # Ogenius: The Voice
//! The `ogenius` CLI provides an interactive chat REPL and an OpenAI-compatible API server,
//! with automatic model downloading from Huggingface.

mod api;

use anyhow::Result;
use api::{chat_completions, list_models, ApiState};
use async_std::sync::Mutex;
use clap::{Parser, Subcommand};
use colored::*;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{
    AssetEvent, BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceConfig,
    InferenceEvent,
};
use rusty_genius_stem::Orchestrator;
use std::io::{self, Write};
use std::sync::Arc;
use tide_websockets::{Message, WebSocket};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a model from HuggingFace
    Download {
        /// HuggingFace model repo (e.g., Qwen/Qwen2.5-1.5B-Instruct)
        repo: String,
    },
    /// Start interactive chat in CLI
    Serve {
        /// HTTP server address
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// WebSocket server address
        #[arg(long, default_value = "127.0.0.1:8081")]
        ws_addr: String,
        /// Model repository to pre-load
        #[arg(long)]
        model: Option<String>,
        /// Do not open the browser automatically
        #[arg(long)]
        no_open: bool,
        /// Unload model after inactivity (seconds)
        #[arg(long, default_value = "300")]
        unload_after: u64,
        /// Quantization level (e.g. Q4_K_M)
        #[arg(long, default_value = "Q4_K_M")]
        quant: String,
        /// Context size
        #[arg(long, default_value = "2048")]
        context_size: u32,
        /// Show thinking tokens
        #[arg(long, default_value = "true")]
        show_thinking: bool,
    },
    /// Start interactive chat in CLI
    Chat {
        /// Model repository
        #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
        model: String,
        /// Quantization level
        #[arg(long, default_value = "Q4_K_M")]
        quant: String,
        /// Context size
        #[arg(long, default_value = "2048")]
        context_size: u32,
        /// Show thinking tokens
        #[arg(long, default_value = "true")]
        show_thinking: bool,
    },
    /// Generate embeddings for input text
    Embed {
        /// Model repository
        #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
        model: String,
        /// Quantization level
        #[arg(long, default_value = "Q4_K_M")]
        quant: String,
        /// Text input to embed
        #[arg(long)]
        input: String,
        /// Context size
        #[arg(long, default_value = "2048")]
        context_size: u32,
    },
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download { repo } => {
            println!("📥 Downloading {}", repo.cyan());
            let mut orchestrator = Orchestrator::new().await?;
            let (mut input_tx, input_rx) = mpsc::channel(10);
            let (output_tx, mut output_rx) = mpsc::channel(10);

            async_std::task::spawn(async move {
                let _ = orchestrator.run(input_rx, output_tx).await;
            });

            input_tx
                .send(BrainstemInput {
                    id: None,
                    command: BrainstemCommand::LoadModel(repo),
                })
                .await?;

            while let Some(output) = output_rx.next().await {
                match output.body {
                    BrainstemBody::Asset(AssetEvent::Progress(curr, total)) => {
                        let pct = if total > 0 {
                            (curr as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        };
                        print!("\rProgress: {:.1}% ({}/{})", pct, curr, total);
                        io::stdout().flush()?;
                    }
                    BrainstemBody::Asset(AssetEvent::Complete(path)) => {
                        println!("\n✅ Download complete: {}", path.green());
                        break;
                    }
                    BrainstemBody::Asset(AssetEvent::Error(e)) => {
                        eprintln!("\n❌ Error: {}", e.red());
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        eprintln!("\n❌ Orchestrator Error: {}", e.red());
                        break;
                    }
                    _ => {}
                }
            }
        }
        Commands::Chat {
            model,
            quant: _,
            context_size,
            show_thinking,
        } => {
            println!("💬 Starting chat with {}", model.cyan());
            let mut orchestrator = Orchestrator::new().await?;
            let (mut input_tx, input_rx) = mpsc::channel(10);
            let (output_tx, mut output_rx) = mpsc::channel(10);

            async_std::task::spawn(async move {
                let _ = orchestrator.run(input_rx, output_tx).await;
            });

            let config = InferenceConfig {
                context_size: Some(context_size),
                show_thinking,
                ..Default::default()
            };

            // Pre-load model
            input_tx
                .send(BrainstemInput {
                    id: None,
                    command: BrainstemCommand::LoadModel(model.clone()),
                })
                .await?;
            println!("⏳ Loading model...");

            while let Some(output) = output_rx.next().await {
                match output.body {
                    BrainstemBody::Asset(AssetEvent::Complete(_)) => break,
                    BrainstemBody::Error(e) => {
                        eprintln!("❌ Failed to load: {}", e.red());
                        return Ok(());
                    }
                    _ => {}
                }
            }
            println!("✅ Model loaded!");
            println!("(Type 'exit' to quit)\n");

            let stdin = io::stdin();
            let mut line = String::new();
            loop {
                print!("{} ", "YOU >".bright_white());
                io::stdout().flush()?;
                line.clear();
                if stdin.read_line(&mut line)? == 0 {
                    break;
                }
                let input = line.trim();
                if input == "exit" || input == "quit" {
                    break;
                }
                let prompt = input.trim();
                if prompt.is_empty() {
                    continue;
                }

                input_tx
                    .send(BrainstemInput {
                        id: None,
                        command: BrainstemCommand::Infer {
                            model: Some(model.clone()),
                            prompt: prompt.to_string(),
                            config: config.clone(),
                        },
                    })
                    .await?;

                print!("{} ", "AI >".bright_green());
                io::stdout().flush()?;

                while let Some(output) = output_rx.next().await {
                    match output.body {
                        BrainstemBody::Event(InferenceEvent::Content(c)) => {
                            print!("{}", c);
                            io::stdout().flush()?;
                        }
                        BrainstemBody::Event(InferenceEvent::Complete) => {
                            println!();
                            break;
                        }
                        BrainstemBody::Error(e) => {
                            eprintln!("\n❌ Error: {}", e.red());
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
        Commands::Embed {
            model,
            quant: _,
            input,
            context_size,
        } => {
            println!("🔢 Generating embeddings using {}", model.cyan());
            let mut orchestrator = Orchestrator::new().await?;
            let (mut input_tx, input_rx) = mpsc::channel(10);
            let (output_tx, mut output_rx) = mpsc::channel(10);

            async_std::task::spawn(async move {
                let _ = orchestrator.run(input_rx, output_tx).await;
            });

            let config = InferenceConfig {
                context_size: Some(context_size),
                show_thinking: false,
                ..Default::default()
            };

            // Pre-load model
            input_tx
                .send(BrainstemInput {
                    id: None,
                    command: BrainstemCommand::LoadModel(model.clone()),
                })
                .await?;
            println!("⏳ Loading model...");

            while let Some(output) = output_rx.next().await {
                match output.body {
                    BrainstemBody::Asset(AssetEvent::Complete(_)) => break,
                    BrainstemBody::Error(e) => {
                        eprintln!("❌ Failed to load: {}", e.red());
                        return Ok(());
                    }
                    _ => {}
                }
            }
            println!("✅ Model loaded!");

            // Send embedding request
            input_tx
                .send(BrainstemInput {
                    id: None,
                    command: BrainstemCommand::Embed {
                        model: Some(model),
                        input,
                        config,
                    },
                })
                .await?;

            println!("⏳ Generating embedding...");

            while let Some(output) = output_rx.next().await {
                match output.body {
                    BrainstemBody::Event(InferenceEvent::Embedding(emb)) => {
                        println!("✅ Embedding generated ({} dimensions)", emb.len());
                        println!("First 10 values: {:?}", &emb[..10.min(emb.len())]);
                        break;
                    }
                    BrainstemBody::Event(InferenceEvent::Complete) => {
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        eprintln!("❌ Error: {}", e.red());
                        break;
                    }
                    _ => {}
                }
            }
        }
        Commands::Serve {
            addr,
            ws_addr,
            model,
            no_open,
            unload_after: _,
            quant: _,
            context_size,
            show_thinking,
        } => {
            println!("🚀 Starting server at {}", addr.cyan());
            println!("🔌 WebSocket endpoint at {}", ws_addr.cyan());

            let mut orchestrator = Orchestrator::new().await?;
            let (input_tx, input_rx) = mpsc::channel(100);
            let (output_tx, mut output_rx) = mpsc::channel(100);

            // Shared senders for all active clients (WS and API)
            let broadcast_senders: Arc<Mutex<Vec<mpsc::Sender<BrainstemOutput>>>> =
                Arc::new(Mutex::new(Vec::new()));

            let state = ApiState {
                input_tx: input_tx.clone(),
                output_senders: broadcast_senders.clone(),
                ws_addr: ws_addr.clone(),
            };

            async_std::task::spawn(async move {
                let _ = orchestrator.run(input_rx, output_tx).await;
            });

            // Bridge orchestrator output to all broadcast senders
            let bridge_senders = broadcast_senders.clone();
            async_std::task::spawn(async move {
                while let Some(msg) = output_rx.next().await {
                    let mut senders = bridge_senders.lock().await;
                    let mut to_remove = Vec::new();
                    for (i, sender) in senders.iter_mut().enumerate() {
                        if sender.send(msg.clone()).await.is_err() {
                            to_remove.push(i);
                        }
                    }
                    // Clean up closed senders
                    for i in to_remove.into_iter().rev() {
                        senders.remove(i);
                    }
                }
            });

            let inference_config = InferenceConfig {
                context_size: Some(context_size),
                show_thinking,
                ..Default::default()
            };

            if let Some(m) = model {
                println!("⏳ Pre-loading model {}...", m.cyan());
                input_tx
                    .clone()
                    .send(BrainstemInput {
                        id: None,
                        command: BrainstemCommand::LoadModel(m),
                    })
                    .await?;
                println!("✅ Model pre-load triggered!");
            }

            let mut app = tide::with_state(state);

            app.at("/").get(|_| async {
                let html = include_str!("index.html");
                Ok(tide::Response::builder(200)
                    .content_type(tide::http::mime::HTML)
                    .body(html)
                    .build())
            });

            app.at("/v1/models").get(list_models);
            app.at("/v1/chat_completions").post(chat_completions);
            app.at("/v1/embeddings").post(api::embeddings);
            app.at("/v1/config").get(api::get_config);

            // WS server with real streaming
            let input_tx_ws = input_tx.clone();
            let bc_senders = broadcast_senders.clone();
            let _ws_server = async_std::task::spawn(async move {
                let mut ws_app = tide::new();
                ws_app.at("/").get(WebSocket::new(move |_req, mut stream| {
                    let mut input_tx = input_tx_ws.clone();
                    let bc_senders = bc_senders.clone();
                    let inference_config = inference_config.clone();
                    async move {
                        // Create a channel for THIS client
                        let (tx, mut rx) = mpsc::channel(100);
                        {
                            let mut senders = bc_senders.lock().await;
                            senders.push(tx);
                        }

                        // Spawn a task to forward events from our private channel to WS
                        let stream_clone = stream.clone();
                        async_std::task::spawn(async move {
                            while let Some(event) = rx.next().await {
                                if let Ok(json) = serde_json::to_string(&event) {
                                    if stream_clone.send_string(json).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        });

                        while let Some(Ok(Message::Text(input))) = stream.next().await {
                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&input) {
                                let prompt = json["prompt"].as_str().unwrap_or("").to_string();
                                let model = json["model"].as_str().map(|s| s.to_string());
                                let _ = input_tx
                                    .send(BrainstemInput {
                                        id: None,
                                        command: BrainstemCommand::Infer {
                                            model,
                                            prompt,
                                            config: inference_config.clone(),
                                        },
                                    })
                                    .await;
                            }
                        }
                        Ok(())
                    }
                }));
                let _ = ws_app.listen(ws_addr).await;
            });

            if !no_open {
                let url = if addr.starts_with(':') {
                    format!("http://127.0.0.1{}", addr)
                } else if addr.starts_with('0') {
                    addr.replace("0.0.0.0", "127.0.0.1")
                } else {
                    format!("http://{}", addr)
                };
                let _ = open_browser(&url).await;
            }

            app.listen(addr).await?;
        }
    }

    Ok(())
}

async fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(target_os = "linux")]
    let cmd = "xdg-open";
    #[cfg(target_os = "windows")]
    let cmd = "start";

    #[cfg(not(target_os = "windows"))]
    let status = async_std::process::Command::new(cmd)
        .arg(url)
        .status()
        .await?;

    #[cfg(target_os = "windows")]
    let status = async_std::process::Command::new("cmd")
        .arg("/c")
        .arg(cmd)
        .arg(url)
        .status()
        .await?;

    if !status.success() {
        eprintln!("⚠️ Failed to open browser: {}", url);
    }

    Ok(())
}
