use futures::{channel::mpsc, sink::SinkExt, StreamExt};

use rusty_genius::core::protocol::{BrainstemInput, BrainstemOutput, InferenceEvent};
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
    println!("Loading model (this may take a while if downloading)...");
    input
        .send(BrainstemInput::LoadModel("qwen-2.5-3b-instruct".into()))
        .await?;

    // 3. Submit prompt
    println!("Sending prompt: 'Why Rust?'");
    input
        .send(BrainstemInput::Infer {
            prompt: "Why Rust?".into(),
            config: Default::default(),
        })
        .await?;

    // 4. Stream results
    println!("--- Response ---");
    while let Some(msg) = output.next().await {
        match msg {
            BrainstemOutput::Event(e) => match e {
                InferenceEvent::Content(c) => {
                    print!("{}", c);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
                InferenceEvent::Complete => {
                    println!("\n--- Complete ---");
                    break;
                }
                InferenceEvent::ProcessStart => {
                    println!("[Inference started]");
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
