use clap::{Parser, Subcommand};
use colored::*;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius::core::protocol::{
    AssetEvent, BrainstemInput, BrainstemOutput, InferenceConfig, InferenceEvent,
};
use rusty_genius::Orchestrator;
use std::io::{self, Write};
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
    Chat {
        /// Model repository
        #[arg(long, default_value = "Qwen/Qwen2.5-1.5B-Instruct")]
        model: String,
    },
    /// Run the API and Web server
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

            input_tx.send(BrainstemInput::LoadModel(repo)).await?;

            while let Some(output) = output_rx.next().await {
                match output {
                    BrainstemOutput::Asset(AssetEvent::Progress(curr, total)) => {
                        let pct = if total > 0 {
                            (curr as f64 / total as f64) * 100.0
                        } else {
                            0.0
                        };
                        print!("\rProgress: {:.1}% ({}/{})", pct, curr, total);
                        io::stdout().flush()?;
                    }
                    BrainstemOutput::Asset(AssetEvent::Complete(path)) => {
                        println!("\n✅ Download complete: {}", path.green());
                        break;
                    }
                    BrainstemOutput::Asset(AssetEvent::Error(e)) => {
                        eprintln!("\n❌ Error: {}", e.red());
                        break;
                    }
                    BrainstemOutput::Error(e) => {
                        eprintln!("\n❌ Orchestrator Error: {}", e.red());
                        break;
                    }
                    _ => {}
                }
            }
        }
        Commands::Chat { model } => {
            println!("💬 Starting chat with {}", model.cyan());
            let mut orchestrator = Orchestrator::new().await?;
            let (mut input_tx, input_rx) = mpsc::channel(10);
            let (output_tx, mut output_rx) = mpsc::channel(10);

            async_std::task::spawn(async move {
                let _ = orchestrator.run(input_rx, output_tx).await;
            });

            // Pre-load model
            input_tx
                .send(BrainstemInput::LoadModel(model.clone()))
                .await?;
            println!("⏳ Loading model...");

            while let Some(output) = output_rx.next().await {
                match output {
                    BrainstemOutput::Asset(AssetEvent::Complete(_)) => break,
                    BrainstemOutput::Asset(AssetEvent::Progress(0, _)) => {
                        // Just waiting
                    }
                    BrainstemOutput::Error(e) => {
                        eprintln!("❌ Failed to load: {}", e.red());
                        return Ok(());
                    }
                    _ => {}
                }
            }
            println!("✅ Ready!");

            loop {
                print!("{} ", "You >".bright_blue());
                io::stdout().flush()?;
                let mut input = String::new();
                if io::stdin().read_line(&mut input)? == 0 {
                    break;
                }
                let prompt = input.trim();
                if prompt.is_empty() {
                    continue;
                }

                input_tx
                    .send(BrainstemInput::Infer {
                        prompt: prompt.to_string(),
                        config: InferenceConfig::default(),
                    })
                    .await?;

                print!("{} ", "AI >".bright_green());
                io::stdout().flush()?;

                while let Some(output) = output_rx.next().await {
                    match output {
                        BrainstemOutput::Event(InferenceEvent::Content(c)) => {
                            print!("{}", c);
                            io::stdout().flush()?;
                        }
                        BrainstemOutput::Event(InferenceEvent::Complete) => {
                            println!();
                            break;
                        }
                        BrainstemOutput::Error(e) => {
                            eprintln!("\n❌ Error: {}", e.red());
                            break;
                        }
                        _ => {}
                    }
                }
            }
        }
        Commands::Serve {
            addr,
            ws_addr,
            model,
        } => {
            println!("🚀 Starting server at {}", addr.cyan());
            println!("🔌 WebSocket endpoint at {}", ws_addr.cyan());

            if let Some(m) = model {
                println!("⏳ Pre-loading model {}...", m.cyan());
                let mut orchestrator = Orchestrator::new().await?;
                let (mut input_tx, input_rx) = mpsc::channel(10);
                let (output_tx, mut output_rx) = mpsc::channel(10);

                async_std::task::spawn(async move {
                    let _ = orchestrator.run(input_rx, output_tx).await;
                });

                input_tx.send(BrainstemInput::LoadModel(m)).await?;
                while let Some(output) = output_rx.next().await {
                    match output {
                        BrainstemOutput::Asset(AssetEvent::Complete(_)) => break,
                        BrainstemOutput::Error(e) => {
                            eprintln!("❌ Failed to pre-load: {}", e.red());
                            break;
                        }
                        _ => {}
                    }
                }
                println!("✅ Model pre-loaded!");
            }

            let mut app = tide::new();

            app.at("/").get(|_| async {
                let html = include_str!("index.html");
                Ok(tide::Response::builder(200)
                    .content_type(tide::http::mime::HTML)
                    .body(html)
                    .build())
            });

            // Minimal API stubs for now
            app.at("/v1/models").get(|_| async {
                Ok(tide::Response::builder(200)
                    .body(serde_json::to_string(&vec!["Qwen/Qwen2.5-1.5B-Instruct"]).unwrap())
                    .build())
            });

            // WS echo stub
            let _ws_server = async_std::task::spawn(async move {
                let mut ws_app = tide::new();
                ws_app
                    .at("/")
                    .get(WebSocket::new(|_req, mut stream| async move {
                        while let Some(Ok(Message::Text(input))) = stream.next().await {
                            let _ = stream.send_string(format!("Echo: {}", input)).await;
                        }
                        Ok(())
                    }));
                let _ = ws_app.listen(ws_addr).await;
            });

            app.listen(addr).await?;
        }
    }

    Ok(())
}
