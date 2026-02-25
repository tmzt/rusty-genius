use facecrab::AssetAuthority;
use futures::StreamExt;
use rusty_genius_thinkerv1::{Response, StatusResponse}; // Use Response and StatusResponse
use std::error::Error;

#[async_std::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let authority = AssetAuthority::new()?;
    let model_name = "tiny-model";

    println!("--- Mode 1: Simple One-Shot ---");
    println!("Checking model: {}", model_name);
    // Provide a request ID and None for ModelConfig
    let path = authority.ensure_model("downloader-one-shot".to_string(), model_name, None).await?;
    println!("Model ready at: {:?}\n", path);

    println!("--- Mode 2: Event-Based Streaming ---");
    // We'll delete the file first to force a download progress for demonstration
    if path.exists() {
        println!("Cleaning up cached model for demonstration...");
        std::fs::remove_file(&path)?;
    }

    // Provide a request ID and None for ModelConfig
    let mut events = authority.ensure_model_stream("downloader-stream".to_string(), model_name, None);
    while let Some(response) = events.next().await {
        if let Response::Status(status) = response {
            match status.status.as_str() {
                "downloading" => {
                    if let Some(progress) = status.progress {
                        print!("\rProgress: {:.1}%", progress * 100.0);
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                },
                "ready" => println!("\nModel ready! Path: {}", status.message.unwrap_or_default()),
                "error" => eprintln!("\nError: {}", status.message.unwrap_or_default()),
                _ => println!("\nStatus: {}", status.status),
            }
        }
    }

    Ok(())
}
