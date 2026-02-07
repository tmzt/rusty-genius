use anyhow::{anyhow, Result};
use async_std::process::{Child, Command, Stdio};
use async_std::task;
use serde_json::json;
use std::time::{Duration, Instant};

/// Helper to spawn ogenius server and return the base URL
async fn setup_test_server(binary_path: &str, port: u16) -> Result<(Child, String)> {
    let addr = format!("127.0.0.1:{}", port);
    let ws_addr = format!("127.0.0.1:{}", port + 1);

    println!(
        "🚀 Starting temporary server at {} using {}...",
        addr, binary_path
    );

    // Launch ogenius serve
    let child = Command::new(binary_path)
        .args(["serve", "--addr", &addr, "--ws-addr", &ws_addr, "--no-open"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    // Wait for server to start
    let base_url = format!("http://{}", addr);
    for _ in 0..30 {
        task::sleep(Duration::from_millis(200)).await;
        if surf::get(format!("{}/v1/models", base_url)).await.is_ok() {
            return Ok((child, base_url));
        }
    }

    Err(anyhow!("Server failed to start within timeout"))
}

#[async_std::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Check if we should spawn our own server
    let (server_proc, url) = if let Ok(test_binary) = std::env::var("TEST_BINARY") {
        if !std::path::Path::new(&test_binary).exists() {
            return Err(anyhow!(
                "TEST_BINARY set to '{}' but file does not exist",
                test_binary
            ));
        }
        let (proc, base_url) = setup_test_server(&test_binary, 9999).await?;
        (Some(proc), base_url)
    } else {
        let url = args
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("http://127.0.0.1:8080");
        (None, url.to_string())
    };

    let input = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("The quick brown fox jumps over the lazy dog.");

    println!("📡 Testing Embedding API at: {}", url);
    println!("📝 Input: \"{}\"", input);

    let start = Instant::now();
    let response: serde_json::Value = surf::post(format!("{}/v1/embeddings", url))
        .body_json(&json!({
            "model": "any",
            "input": input
        }))
        .map_err(|e| anyhow!("Body error: {}", e))?
        .recv_json()
        .await
        .map_err(|e| anyhow!("Request failed: {}", e))?;

    let duration = start.elapsed();

    if let Some(data) = response["data"].as_array() {
        if let Some(emb) = data.first() {
            let vec = emb["embedding"].as_array().unwrap();
            println!("✅ Success! Dimension: {}", vec.len());
            println!("⏱️ Latency: {:?}", duration);
            println!("📊 First 5 values: {:?}", &vec[..5.min(vec.len())]);
        }
    } else {
        println!("❌ Error: Unexpected response format: {}", response);
    }

    // Cleanup server if we started it
    if let Some(mut proc) = server_proc {
        println!("🛑 Shutting down temporary server...");
        proc.kill()?;
    }

    Ok(())
}
