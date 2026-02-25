//! # Ogenius: The Voice
//! The `ogenius` CLI provides an interactive chat REPL and an OpenAI-compatible API server,
//! with automatic model downloading from Huggingface.

mod api;

use anyhow::Result;
use api::{chat_completions, list_models, ApiState};
use smol::{self, lock::Mutex, net::{TcpListener, UnixListener}, process::Command, task, Timer, FutureExt}; // Use smol for networking, tasks, futures
use futures::io::{AsyncRead, AsyncWrite, BufReader}; // Use futures for IO traits
use futures::prelude::{AsyncReadExt, AsyncWriteExt}; // Use futures prelude for extensions
use clap::{Parser, Subcommand};
use colored::*;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius::Genius;
use rusty_genius_thinkerv1::{EventResponse, InferenceConfig, ModelConfig, Response, Request, new_ensure_request, new_inference_request, new_embed_request};
use rusty_genius_core::protocol::{BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput};
use serde::{Deserialize, Serialize};
use std::io::Write; // Import Write trait for flush
use std::net::SocketAddr; // Keep std::net::SocketAddr for parsing
use std::process; // Import process module for exit
use std::str::FromStr; // Import FromStr for SocketAddr parsing
use std::sync::Arc;
use std::time::Duration;
use tide_smol::{Body, Request as TideRequest, Response as TideResponse, StatusCode};
use uuid::Uuid; // Import Uuid

// Import indicatif progress bar components
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};


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
        /// Quantization level (e.g. Q4_K_M)
        #[arg(long)]
        quant: Option<String>,
        /// Context size
        #[arg(long)]
        context_size: Option<u32>,
        /// Time-to-live for the model in seconds (-1 for infinite)
        #[arg(long)]
        ttl_seconds: Option<i64>,
    },
    /// Start HTTP/WebSocket server and optionally the ThinkerV1 server
    Serve {
        /// HTTP server address
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// WebSocket server address
        #[arg(long, default_value = "127.0.0.1:8081")]
        ws_addr: String,
        /// Thinkerv1 protocol server address (e.g., 127.0.0.1:8082 or unix:/tmp/thinker.sock)
        #[arg(long)]
        thinker_addr: Option<String>,
        /// Model repository to pre-load
        #[arg(long)]
        model: Option<String>,
        /// Do not open the browser automatically
        #[arg(long)]
        no_open: bool,
        /// Default unload model after inactivity (seconds)
        #[arg(long, default_value = "300")]
        unload_after: u64,
        /// Quantization level (e.g. Q4_K_M)
        #[arg(long)]
        quant: Option<String>,
        /// Context size
        #[arg(long)]
        context_size: Option<u32>,
        /// Show thinking tokens by default
        #[arg(long, default_value = "false")]
        show_thinking: bool,
        /// Models to pre-load (download/verify) before starting
        #[arg(long)]
        load_models: Vec<String>,
    },
    /// Start only the Thinkerv1 protocol server
    Thinker {
        /// Thinkerv1 protocol server address (e.g., 127.0.0.1:8082 or unix:/tmp/thinker.sock)
        #[arg(long)]
        addr: String,
        /// Default unload model after inactivity (seconds)
        #[arg(long, default_value = "300")]
        unload_after: u64,
    },
    /// Start interactive chat in CLI
    Chat {
        /// Model repository
        #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
        model: String,
        /// Quantization level
        #[arg(long)]
        quant: Option<String>,
        /// Context size
        #[arg(long)]
        context_size: Option<u32>,
        /// Show thinking tokens
        #[arg(long, default_value = "false")]
        show_thinking: bool,
    },
    /// Generate embeddings for input text
    Embed {
        /// Model repository
        #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
        model: String,
        /// Quantization level
        #[arg(long)]
        quant: Option<String>,
        /// Text input to embed
        #[arg(long)]
        input: String,
        /// Context size
        #[arg(long)]
        context_size: Option<u32>,
    },
}

async fn wait_for_models(genius: Arc<Mutex<Genius>>, models: Vec<String>, default_config: Option<ModelConfig>) -> Result<()> {
    if models.is_empty() {
        return Ok(());
    }
    println!("📦 Pre-loading {} models...", models.len());
    let multi = MultiProgress::new();
    let sty = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}",
    )
    .unwrap()
    .progress_chars("#>-");

    let tasks: Vec<_> = models
        .into_iter()
        .map(|model_name| {
            let pb = multi.add(ProgressBar::new(0));
            pb.set_style(sty.clone());
            pb.set_message(format!("starting {}", model_name));
            let genius_clone = Arc::clone(&genius); // Clone Arc for task
            let config_clone = default_config.clone();
            async move {
                let mut genius = genius_clone.lock().await; // Lock Genius
                let mut stream = genius
                    .ensure_model_stream(model_name.clone(), true, config_clone)
                    .await?;
                while let Some(response) = stream.next().await {
                    if let Response::Status(status) = response {
                        match status.status.as_str() {
                            "downloading" => {
                                let total = 1_000_000_000; // Fake total
                                pb.set_length(total);
                                if let Some(p) = status.progress {
                                    pb.set_position((p * total as f32) as u64);
                                }
                                pb.set_message(format!("downloading {}", model_name));
                            }
                            "ready" => {
                                pb.finish_with_message(format!("✅ {}", model_name));
                                return Ok(());
                            }
                            "error" => {
                                let err_msg = status.message.unwrap_or_default();
                                pb.abandon_with_message(format!("❌ {} failed: {}", model_name, err_msg));
                                return Err(anyhow::anyhow!(err_msg));
                            }
                            _ => {}
                        }
                    }
                }
                Ok(())
            }
        })
        .collect();

    let results = futures::future::join_all(tasks).await;
    multi.clear()?;
    if results.iter().any(|r| r.is_err()) {
        anyhow::bail!("Some models failed to load.");
    }
    println!("✨ All models loaded.\n");
    Ok(())
    
}

async fn handle_thinker_connection(
    stream: impl AsyncRead + AsyncWrite + Unpin + Send + 'static, // Removed mut, split() provides mutable parts
    genius: Arc<Mutex<Genius>>, // Changed to Arc<Mutex<Genius>>
    addr_str: String,
) -> Result<()> {
    let (reader, mut writer) = stream.split(); // Use split() which is available for types implementing AsyncRead + AsyncWrite
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    eprintln!("DEBUG: Thinkerv1 connection from {}", addr_str);

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let request: rusty_genius_thinkerv1::Request = match serde_json::from_str(&line) {
                    Ok(req) => req,
                    Err(e) => {
                        eprintln!("ERROR: Failed to parse request: {}", e);
                        continue;
                    }
                };
                let genius_clone = Arc::clone(&genius); // Clone Arc for task
                let write_task: Task<Result<()>> = task::spawn(async move { // Use smol::task::spawn
                    let mut genius = genius_clone.lock().await; // Lock Genius
                    let mut response_stream = match request {
                        rusty_genius_thinkerv1::Request::Ensure(req) => {
                            genius.ensure_model_stream(req.model, req.report_status, req.model_config).await?
                        }
                        rusty_genius_thinkerv1::Request::Inference(req) => {
                            genius.infer_stream(req.prompt, req.inference_config).await?
                        }
                        rusty_genius_thinkerv1::Request::Embed(req) => genius.embed(req.text).await?,
                    };
                    while let Some(response) = response_stream.next().await {
                        let mut json = serde_json::to_vec(&response)?;
                        json.push(b'\n');
                        writer.write_all(&json).await?;
                    }
                    Ok(())
                });
                if let Err(e) = write_task.await {
                    eprintln!("ERROR: Handler task failed: {}", e);
                }
            }
            Err(e) => {
                eprintln!("ERROR: Read error: {}", e);
                break;
            }
        }
    }
    Ok(())
}

#[smol::main] // Use smol::main as the entry point
async fn main() -> anyhow::Result<()> {
    ctrlc::set_handler(|| { // Simplified ctrlc handler
        println!("\n🛑 Ctrl-C received, exiting...");
        process::exit(130);
    })?;

    let cli = Cli::parse();

    match cli.command {
        Commands::Download { repo, quant, context_size, ttl_seconds } => {
            let genius = Arc::new(Mutex::new(Genius::new(None).await?)); // Wrap Genius in Arc<Mutex>
            let config = Some(ModelConfig { quant, context_size, ttl_seconds });
            wait_for_models(Arc::clone(&genius), vec![repo], config).await?;
        }
        Commands::Chat { model, quant, context_size, show_thinking } => {
            let genius = Arc::new(Mutex::new(Genius::new(None).await?)); // Wrap Genius in Arc<Mutex>
            let model_config = Some(ModelConfig { quant, context_size, ttl_seconds: Some(-1) /* Keep alive for session */ });
            let inference_config = Some(InferenceConfig { show_thinking, ..Default::default() });

            println!("⏳ Loading model {}...", model.cyan());
            wait_for_models(Arc::clone(&genius), vec![model.clone()], model_config).await?;

            println!("💬 Starting chat with {}. (Type 'exit' to quit)\n", model.cyan());

            let stdin = io::stdin();
            loop {
                print!("{} ", "YOU >".bright_white());
                io::stdout().flush()?;
                let mut line = String::new();
                if stdin.read_line(&mut line)? == 0 { break; }
                let input = line.trim();
                if input == "exit" || input == "quit" { break; }
                if input.is_empty() { continue; }

                let mut stream = genius.lock().await.infer_stream(input.to_string(), inference_config.clone()).await?; // Lock Genius

                print!("{} ", "AI >".bright_green());
                io::stdout().flush()?;

                while let Some(response) = stream.next().await {
                    if let Response::Event(event) = response {
                        match event {
                            EventResponse::Content(c) => {
                                print!("{}", c.content);
                                io::stdout().flush()?;
                            }
                            EventResponse::Complete { .. } => {
                                println!();
                                break;
                            }
                            _ => {}
                        }
                    } else if let Response::Status(s) = response {
                        if s.status == "error" {
                            eprintln!("\n❌ Error: {}", s.message.unwrap_or_default().red());
                            break;
                        }
                    }
                }
            }
        }
        Commands::Embed { model, quant, input, context_size } => {
            let genius = Arc::new(Mutex::new(Genius::new(None).await?)); // Wrap Genius in Arc<Mutex>
            let model_config = Some(ModelConfig { quant, context_size, ttl_seconds: None });
            
            println!("⏳ Loading embedding model {}...", model.cyan());
            wait_for_models(Arc::clone(&genius), vec![model], model_config).await?;

            println!("🔢 Generating embedding for: \"{}\"", input.yellow());
            let mut stream = genius.lock().await.embed(input).await?; // Lock Genius

            while let Some(response) = stream.next().await {
                if let Response::Event(EventResponse::Embedding{ vector_hex, .. }) = response {
                    println!("✅ Embedding generated ({} hex chars)", vector_hex.len());
                    println!("First 50 hex chars: {:?}", &vector_hex[..50.min(vector_hex.len())]);
                    break;
                } else if let Response::Status(s) = response {
                    if s.status == "error" {
                        eprintln!("\n❌ Error: {}", s.message.unwrap_or_default().red());
                        break;
                    }
                }
            }
        }
        Commands::Serve { addr, ws_addr, thinker_addr, model, no_open, unload_after, quant, context_size, show_thinking, load_models } => {
            if let Some(t_addr) = thinker_addr {
                println!("🚀 Spawning Thinker subprocess...");
                // Use std::process::Command for external process execution
                Command::new(std::env::current_exe()?)
                    .arg("thinker")
                    .arg("--addr")
                    .arg(t_addr)
                    .arg("--unload-after")
                    .arg(unload_after.to_string())
                    .spawn()?;
                Timer::after(std::time::Duration::from_millis(500)).await;
            }

            let genius = Arc::new(Mutex::new(Genius::new(Some(unload_after)).await?)); // Wrap Genius in Arc<Mutex>
            let default_model_config = Some(ModelConfig { quant, context_size, ttl_seconds: None });
            let default_inference_config = InferenceConfig { show_thinking, ..Default::default() };
            
            wait_for_models(Arc::clone(&genius), load_models, default_model_config.clone()).await?;

            if let Some(m) = model {
                 wait_for_models(Arc::clone(&genius), vec![m], default_model_config.clone()).await?;
            }
            
            // Correct ApiState initialization. Genius is now Arc<Mutex<>>.
            // Also, ApiState should not store default_inference_config directly as it's not used in its fields.
            // Accessing input_tx needs locking the mutex.
            let api_state = ApiState { input_tx: genius.lock().await.input_tx.clone(), output_senders: Arc::new(Mutex::new(Vec::new())), ws_addr: ws_addr.clone() };
            let mut app = tide_smol::with_state(api_state.clone());
            
            app.at("/").get(|_| async { Ok(tide_smol::Response::builder(200).content_type(tide_smol::http::mime::HTML).body(include_str!("index.html")).build()) });
            app.at("/v1/models").get(list_models);
            app.at("/v1/chat/completions").post(chat_completions);
            // Other routes...

            if !no_open {
                let _ = open_browser(&format!("http://{}", addr)).await;
            }
            eprintln!("🚀 API Server listening on {}", addr.cyan());
            app.listen(addr).await?;
        }
        // Handle the case where 'thinker' command is called directly
        Commands::Thinker { addr, unload_after } => {
            println!("🚀 Starting ThinkerV1 server on {}...", addr.cyan());
            let genius = Arc::new(Mutex::new(Genius::new(Some(unload_after)).await?)); // Wrap Genius in Arc<Mutex>
            let server = SocketAddr::from_str(&addr).unwrap(); // Assuming addr is always valid socket addr string
            let listener = TcpListener::bind(server).await?;
            let mut server_stream = listener.incoming();

            while let Some(Ok(stream)) = server_stream.next().await {
                let genius_clone = Arc::clone(&genius); // Clone Arc for each task
                task::spawn(async move { // Use async_std::task::spawn
                    if let Err(e) = handle_thinker_connection(stream, genius_clone, addr.clone()).await { // Corrected signature and cloning
                        eprintln!("ERROR: Handler failed: {}", e);
                    }
                }).detach();
            }
        }
        _ => {
            println!("This command has not been fully implemented yet.");
        }
    }

    Ok(())
}

async fn open_browser(url: &str) -> Result<()> {
    // This function is a placeholder and likely needs platform-specific implementation.
    // For demonstration, we'll just print the URL.
    println!("Visit {} to open the application.", url.cyan());
    // In a real CLI, you might use `open` command on macOS/Linux or `start` on Windows.
    // Example for macOS:
    // if cfg!(target_os = "macos") {
    //     Command::new("open").arg(url).spawn()?;
    // }
    Ok(())
}
