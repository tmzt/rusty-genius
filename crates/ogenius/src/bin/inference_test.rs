use anyhow::{anyhow, Result};

use async_std::task;
use serde::Deserialize;
use serde_json::json;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Deserialize, Debug)]
struct ModelResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    object: String,
    // #[allow(dead_code)]
    // #[serde(required = false)]
    // pub role: String,
    #[allow(dead_code)]
    purpose: String,
}

#[derive(Deserialize, Debug)]
struct ModelList {
    data: Vec<ModelResponse>,
}

#[derive(Deserialize, Debug)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Deserialize, Debug)]
struct ChatMessage {
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

/// Helper to spawn ogenius server and return the base URL
async fn setup_test_server(binary_path: &str, port: u16) -> Result<(Child, String)> {
    let addr = format!("127.0.0.1:{}", port);
    let ws_addr = format!("127.0.0.1:{}", port + 1);

    println!(
        "🚀 Starting temporary server at {} using {}...",
        addr, binary_path
    );

    // // Use system temp dir to avoid sandbox permission issues in target/tmp
    // let genius_home = std::env::temp_dir().join("rusty-genius-test-home");
    // if genius_home.exists() {
    //     let _ = std::fs::remove_dir_all(&genius_home);
    // }
    // std::fs::create_dir_all(&genius_home)?;

    // println!("🏠 using GENIUS_HOME={:?}", genius_home);

    // Launch ogenius serve
    let mut child = Command::new(binary_path)
        .args(["serve", "--addr", &addr, "--ws-addr", &ws_addr, "--no-open"])
        // .env("GENIUS_HOME", genius_home)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    // No need to spawn threads as we inherit stdio

    // Wait for server to start
    let base_url = format!("http://{}", addr);
    for i in 0..300 {
        if let Ok(Some(status)) = child.try_wait() {
            panic!("Server process exited unexpectedly with status: {}", status);
        }
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
        let (proc, base_url) = setup_test_server(&test_binary, 10101).await?;
        (Some(proc), base_url)
    } else {
        let url = args
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("http://127.0.0.1:8080");
        (None, url.to_string())
    };

    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Explain quantum computing in one sentence.");

    println!("📡 Testing Inference API at: {}", url);
    println!("🔍 Searching for TinyLlama model...");

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

    // 2. Filter for tiny-llama, ensuring purpose is Inference
    let candidates: Vec<&ModelResponse> = list_body
        .data
        .iter()
        .filter(|m| m.purpose == "Inference")
        .collect();

    let model_id = candidates
        .iter()
        .find(|m| m.id.contains("tiny-llama"))
        .or_else(|| {
            candidates
                .iter()
                .find(|m| m.id.contains("tiny") || m.id.contains("llama"))
        })
        .map(|m| m.id.clone())
        .ok_or_else(|| {
            anyhow!(
                "No suitable tiny inference model found in registry! Available: {:?}",
                list_body.data
            )
        })?;

    println!("✅ Found model: {}", model_id);
    println!("📝 Prompt: \"{}\"", prompt);

    let start = Instant::now();
    let mut response = surf::post(format!("{}/v1/chat/completions", url))
        .body_json(&json!({
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": false
        }))
        .map_err(|e| anyhow!("Body error: {}", e))?
        .send()
        .await
        .map_err(|e| anyhow!("Request failed: {}", e))?;

    let duration = start.elapsed();

    if response.status().is_success() {
        let body: ChatCompletionResponse = response
            .body_json()
            .await
            .map_err(|e| anyhow!("Failed to parse JSON response: {}", e))?;

        if let Some(choice) = body.choices.first() {
            println!("✅ Success!");
            println!("⏱️ Latency: {:?}", duration);
            println!("🤖 Response: {}", choice.message.content);
        } else {
            println!("❌ Error: No choices in response");
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
