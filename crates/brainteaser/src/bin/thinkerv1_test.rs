use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use rusty_genius_thinkerv1::{new_inference_request, InferenceConfig};
use rusty_genius_thinkerv1_client::{Address, Client};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the Unix Domain Socket for the ThinkerV1 server
    #[arg(long, default_value = "/tmp/thinker.sock")]
    uds_path: PathBuf,

    /// The prompt to send to the model
    #[arg(long, default_value = "Tell me a short story about a robot.")]
    prompt: String,

    /// Whether to show the model's "thinking" process
    #[arg(long)]
    show_thinking: bool,
}

#[async_std::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!(
        "Connecting to ThinkerV1 server at: {}",
        cli.uds_path.display()
    );

    let client = Client::connect(Address::Uds(cli.uds_path)).await?;
    println!("Connection successful.");

    let inference_config = Some(InferenceConfig {
        show_thinking: cli.show_thinking,
        ..Default::default()
    });

    let request = new_inference_request(cli.prompt, inference_config);

    println!("Sending inference request (id: {})...", request.get_id());

    let mut stream = client.request(request).await?;

    println!("--- Server Response ---");
    while let Some(response_result) = stream.next().await {
        match response_result {
            Ok(response) => {
                println!("{:?}", response);
            }
            Err(e) => {
                eprintln!("ERROR: Received an error from the client stream: {}", e);
                break;
            }
        }
    }
    println!("--- End of Response ---");

    Ok(())
}