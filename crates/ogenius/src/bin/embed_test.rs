use anyhow::{anyhow, Result};
use async_std::io::BufReader;
use async_std::prelude::*;
use async_std::process::{Child, Command, Stdio};
use async_std::task;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};

#[derive(Deserialize, Debug)]
struct ModelResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    object: String,
    purpose: String,
}

#[derive(Deserialize, Debug)]
struct ModelList {
    data: Vec<ModelResponse>,
}

/// Helper to spawn ogenius server and return the base URL
async fn setup_test_server(binary_path: &str, port: u16) -> Result<(Child, String)> {
    let addr = format!("127.0.0.1:{}", port);
    let ws_addr = format!("127.0.0.1:{}", port + 1);

    println!(
        "🚀 Starting temporary server at {} using {}...",
        addr, binary_path
    );

    // Launch ogenius serve
    let mut child = Command::new(binary_path)
        .args(["serve", "--addr", &addr, "--ws-addr", &ws_addr, "--no-open"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Spawn tasks to pipe output to our stdout/stderr with a prefix
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    task::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        while let Some(line) = reader.next().await {
            if let Ok(l) = line {
                println!("[SERVER OUT] {}", l);
            }
        }
    });

    task::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Some(line) = reader.next().await {
            if let Ok(l) = line {
                eprintln!("[SERVER ERR] {}", l);
            }
        }
    });

    // Wait for server to start
    let base_url = format!("http://{}", addr);
    for i in 0..50 {
        task::sleep(Duration::from_millis(200)).await;
        if surf::get(format!("{}/v1/models", base_url)).await.is_ok() {
            println!("✅ Server is up and responding on {}", base_url);
            return Ok((child, base_url));
        }
        if i % 10 == 0 && i > 0 {
            println!("⏳ Waiting for server ({}ms elapsed)...", i * 200);
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
    println!("� Searching for embedding model...");

    // 1. List models
    let mut list_res = surf::get(format!("{}/v1/models", url))
        .await
        .map_err(|e| anyhow!("Failed to list models: {}", e))?;

    if !list_res.status().is_success() {
        return Err(anyhow!(
            "List models failed: {}",
            list_res.body_string().await.unwrap_or_default()
        ));
    }

    let list_body: ModelList = list_res
        .body_json()
        .await
        .map_err(|e| anyhow!("Failed to parse model list: {}", e))?;

    // 2. Filter for embedding model
    // Note: The API returns { id: "name", object: "model" }
    // It does not currently return the purpose.
    // We need to update Ogenius API to return the purpose or detailed info.
    // FOR NOW: We will rely on the name still, until we update the API.
    // Wait, the task is to use the purpose.
    // I need to update `crates/ogenius/src/api.rs` to include `purpose` in `ModelResponse`.

    let model_id = list_body
        .data
        .iter()
        .find(|m| m.purpose == "Embedding")
        .map(|m| m.id.clone())
        .ok_or_else(|| {
            anyhow!(
                "No embedding model found in registry! Available: {:?}",
                list_body.data
            )
        })?;

    println!("✅ Found model: {}", model_id);
    println!("📝 Input: \"{}\"", input);

    let start = Instant::now();
    let mut response = surf::post(format!("{}/v1/embeddings", url))
        .body_json(&json!({
            "model": model_id,
            "input": input
        }))
        .map_err(|e| anyhow!("Body error: {}", e))?
        .send()
        .await
        .map_err(|e| anyhow!("Request failed: {}", e))?;

    let duration = start.elapsed();

    if response.status().is_success() {
        let body: serde_json::Value = response
            .body_json()
            .await
            .map_err(|e| anyhow!("Failed to parse JSON response: {}", e))?;

        if let Some(data) = body["data"].as_array() {
            if let Some(emb) = data.first() {
                if let Some(vec) = emb["embedding"].as_array() {
                    println!("✅ Success! Dimension: {}", vec.len());
                    println!("⏱️ Latency: {:?}", duration);
                    // println!("📊 First 5 values: {:?}", &vec[..5.min(vec.len())]);
                } else {
                    println!("❌ Error: 'embedding' field is missing or not an array");
                }
            } else {
                println!("❌ Error: 'data' array is empty");
            }
        } else {
            println!("❌ Error: Unexpected response format: {}", body);
        }
    } else {
        let status = response.status();
        let body_text = response.body_string().await.unwrap_or_default();
        println!(
            "❌ Error: Server returned status {} with body: \"{}\"",
            status, body_text
        );
    }

    // Cleanup server if we started it
    if let Some(mut proc) = server_proc {
        println!("🛑 Shutting down temporary server...");
        let _ = proc.kill();
    }

    Ok(())
}
