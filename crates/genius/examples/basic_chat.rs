use futures::{channel::mpsc, sink::SinkExt, StreamExt};

use rusty_genius::core::protocol::{AssetEvent, BrainstemInput, BrainstemOutput, InferenceEvent};
use rusty_genius::Orchestrator;

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

    // 2. Select model (downloads and verifies automatically)
    let model_name = "tiny-model";
    println!("Loading model: {}...", model_name);
    input
        .send(BrainstemInput::LoadModel(model_name.into()))
        .await?;

    // 3. Submit prompt
    let prompt = "Once upon a time, in the world of systems programming, there was a language called Rust...";
    println!("Sending prompt: '{}'", prompt);
    input
        .send(BrainstemInput::Infer {
            prompt: prompt.into(),
            config: Default::default(),
        })
        .await?;

    // 4. Stream results
    println!("--- Messages ---");
    while let Some(msg) = output.next().await {
        match msg {
            BrainstemOutput::Asset(a) => match a {
                AssetEvent::Started(s) => println!("[Asset] Starting: {}", s),
                AssetEvent::Progress(c, t) => {
                    let pct = (c as f64 / t as f64) * 100.0;
                    print!("\r[Asset] Downloading: {:.1}%", pct);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                AssetEvent::Complete(s) => println!("\n[Asset] Ready: {}", s),
                AssetEvent::Error(e) => eprintln!("\n[Asset] Error: {}", e),
            },
            BrainstemOutput::Event(e) => match e {
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
            },
            BrainstemOutput::Error(err) => {
                eprintln!("\nBrainstem Error: {}", err);
                break;
            }
        }
    }

    Ok(())
}
