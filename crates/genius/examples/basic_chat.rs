use futures::{channel::mpsc, sink::SinkExt, StreamExt};

use rusty_genius_core::protocol::{
    AssetEvent, BrainstemBody, BrainstemCommand, BrainstemInput, InferenceEvent,
};
use rusty_genius_stem::Orchestrator;

#[async_std::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Core orchestration setup
    let mut genius = Orchestrator::new().await?;
    let (mut input, rx) = mpsc::channel(100);
    let (tx, mut output) = mpsc::channel(100);

    async_std::task::spawn(async move {
        if let Err(e) = genius.run(rx, tx).await {
            eprintln!("Orchestrator error: {}", e);
        }
    });

    // 2. Select model
    let model_name = "tiny-model";
    println!("Loading model: {}...", model_name);
    input
        .send(BrainstemInput {
            id: None,
            command: BrainstemCommand::LoadModel(model_name.into()),
        })
        .await?;

    // 3. Submit prompt
    let prompt = "Once upon a time, in the world of systems programming, there was a language called Rust...";
    println!("Sending prompt: '{}'", prompt);
    input
        .send(BrainstemInput {
            id: None,
            command: BrainstemCommand::Infer {
                model: Some(model_name.into()),
                prompt: prompt.into(),
                config: Default::default(),
            },
        })
        .await?;

    // 4. Stream results
    println!("--- Messages ---");
    while let Some(msg) = output.next().await {
        match msg.body {
            BrainstemBody::Asset(a) => match a {
                AssetEvent::Started(s) => println!("[Asset] Starting: {}", s),
                AssetEvent::Progress(c, t) => {
                    let pct = (c as f64 / t as f64) * 100.0;
                    print!("\r[Asset] Downloading: {:.1}%", pct);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                AssetEvent::Complete(s) => println!("\n[Asset] Ready: {}", s),
                AssetEvent::Error(e) => eprintln!("\n[Asset] Error: {}", e),
            },
            BrainstemBody::Event(e) => match e {
                InferenceEvent::Content(c) => {
                    print!("{}", c);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
                InferenceEvent::Complete => {
                    println!("\n--- Inference Complete ---");
                    break;
                }
                InferenceEvent::ProcessStart => {
                    println!("\n[Inference started]");
                }
                InferenceEvent::Thought(_t) => {
                    // Optionally handle thoughts here
                }
                _ => {}
            },
            BrainstemBody::Error(err) => {
                eprintln!("\nBrainstem Error: {}", err);
                break;
            }
            BrainstemBody::ModelList(_) => {
                // Ignored in this example
            }
        }
    }

    Ok(())
}
