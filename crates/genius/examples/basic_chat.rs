use futures::{channel::mpsc, sink::SinkExt, StreamExt};
use rusty_genius_core::protocol::{BrainstemBody, BrainstemCommand, BrainstemInput}; // Removed BrainstemOutput
use rusty_genius_stem::Orchestrator;
use rusty_genius_thinkerv1::{new_ensure_request, new_inference_request, EventResponse, InferenceConfig, Response, Request}; // Removed ModelConfig, StatusResponse
use uuid::Uuid;

#[async_std::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Core orchestration setup
    let mut orchestrator = Orchestrator::new(None).await?; // Pass None for default_unload_after
    let (input_tx, input_rx) = mpsc::channel(100);
    let (output_tx, mut output_rx) = mpsc::channel(100);

    async_std::task::spawn(async move {
        if let Err(e) = orchestrator.run(input_rx, output_tx).await {
            eprintln!("Orchestrator error: {}", e);
        }
    });

    // We'll use this sender to send commands to the orchestrator
    let mut input_sender = input_tx.clone();

    // 2. Select model
    let model_name = "tiny-model";
    println!("Loading model: {}...", model_name);

    let ensure_req_enum = new_ensure_request(model_name.to_string(), true, None);
    let ensure_req_id = ensure_req_enum.get_id().to_string();
    let ensure_req = match ensure_req_enum {
        Request::Ensure(req) => req,
        _ => unreachable!(), // new_ensure_request always returns Request::Ensure
    };

    input_sender
        .send(BrainstemInput {
            id: ensure_req_id.clone(),
            command: BrainstemCommand::EnsureModel(ensure_req),
        })
        .await?;

    // Wait for model to be ready
    loop {
        if let Some(msg) = output_rx.next().await {
            if msg.id != ensure_req_id {
                continue;
            }
            if let BrainstemBody::Thinker(Response::Status(status)) = msg.body {
                println!("[Client] Model Status: {:?}", status);
                if status.status == "ready" {
                    break;
                }
                if status.status == "error" {
                    return Err(anyhow::anyhow!("Failed to load model: {}", status.message.unwrap_or_default()).into());
                }
            } else if let BrainstemBody::Error(e) = msg.body {
                return Err(anyhow::anyhow!("Orchestrator error during model ensure: {}", e).into());
            }
        } else {
            return Err(anyhow::anyhow!("Orchestrator output stream closed unexpectedly during model ensure.").into());
        }
    }
    println!("Model is ready.");

    // 3. Submit prompt
    let prompt = "Once upon a time, in the world of systems programming, there was a language called Rust...";
    println!("Sending prompt: '{}'", prompt);

    let infer_req_enum = new_inference_request(prompt.to_string(), Some(InferenceConfig::default()));
    let infer_req_id = infer_req_enum.get_id().to_string();
    let infer_req = match infer_req_enum {
        Request::Inference(req) => req,
        _ => unreachable!(), // new_inference_request always returns Request::Inference
    };

    input_sender
        .send(BrainstemInput {
            id: infer_req_id.clone(),
            command: BrainstemCommand::Inference(infer_req),
        })
        .await?;

    // 4. Stream results
    println!("--- Messages ---");
    while let Some(msg) = output_rx.next().await {
        if msg.id != infer_req_id {
            continue;
        }

        match msg.body {
            BrainstemBody::Thinker(Response::Event(event)) => match event {
                EventResponse::Content{ content, .. } => {
                    print!("{}", content);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
                EventResponse::Complete{ .. } => {
                    println!("\n--- Inference Complete ---");
                    break;
                }
                EventResponse::Thought{ content, .. } => {
                    eprintln!("\n[Thought] {}", content);
                }
                _ => {} // Ignore other EventResponses
            },
            BrainstemBody::Thinker(Response::Status(status)) => {
                if status.status == "error" {
                     return Err(anyhow::anyhow!("Inference error: {}", status.message.unwrap_or_default()).into());
                }
            }
            BrainstemBody::Error(err) => {
                eprintln!("\nOrchestrator Error during inference: {}", err);
                break;
            }
            _ => {} // Ignore other BrainstemBody types
        }
    }

    Ok(())
}
