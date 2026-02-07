use async_std::task;
use std::process::Stdio;
use std::time::Duration;

/// Helper to spawn ogenius server and return the base URL
async fn setup_test_server(port: u16) -> (async_std::process::Child, String) {
    let addr = format!("127.0.0.1:{}", port);
    let ws_addr = format!("127.0.0.1:{}", port + 1);

    // Check for binary in order of preference:
    // 1. TEST_BINARY env var
    // 2. Release build
    // 3. Debug build
    let binary_path = if let Ok(test_binary) = std::env::var("TEST_BINARY") {
        if std::path::Path::new(&test_binary).exists() {
            test_binary
        } else {
            panic!(
                "TEST_BINARY set to '{}' but file does not exist",
                test_binary
            );
        }
    } else {
        let release_path = "../../target/release/ogenius";
        let debug_path = "../../target/debug/ogenius";

        if std::path::Path::new(release_path).exists() {
            release_path.to_string()
        } else if std::path::Path::new(debug_path).exists() {
            debug_path.to_string()
        } else {
            panic!(
                "ogenius binary not found. Run 'cargo build --bin ogenius' or set TEST_BINARY env var."
            );
        }
    };

    // Launch ogenius serve
    let child = async_std::process::Command::new(&binary_path)
        .args(&["serve", "--addr", &addr, "--ws-addr", &ws_addr, "--no-open"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start ogenius");

    // Wait for server to start
    let base_url = format!("http://{}", addr);
    for _ in 0..30 {
        task::sleep(Duration::from_millis(200)).await;
        if surf::get(&format!("{}/v1/models", base_url)).await.is_ok() {
            return (child, base_url);
        }
    }

    panic!("Server failed to start within timeout");
}

#[async_std::test]
async fn test_list_models() {
    let (mut server, base_url) = setup_test_server(9001).await;

    let response: serde_json::Value = surf::get(format!("{}/v1/models", base_url))
        .recv_json()
        .await
        .expect("Failed to get models");

    assert_eq!(response["object"], "list");
    assert!(response["models"].is_array());

    // Should have at least the default model
    let models = response["models"].as_array().unwrap();
    assert!(!models.is_empty(), "Should return at least one model");

    // Verify model structure
    if let Some(first_model) = models.first() {
        assert!(first_model["id"].is_string());
        assert_eq!(first_model["object"], "model");
    }

    // Cleanup
    let _ = server.kill();
}

#[async_std::test]
async fn test_chat_completions() {
    let (mut server, base_url) = setup_test_server(9003).await;

    let request = serde_json::json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    });

    let response: serde_json::Value = surf::post(format!("{}/v1/chat_completions", base_url))
        .body_json(&request)
        .unwrap()
        .recv_json()
        .await
        .expect("Failed to get chat completion");

    assert_eq!(response["object"], "chat.completion");
    assert!(response["choices"].is_array());

    let choices = response["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);

    let choice = &choices[0];
    assert_eq!(choice["index"], 0);
    assert_eq!(choice["finish_reason"], "stop");

    let message = &choice["message"];
    assert_eq!(message["role"], "assistant");

    // Pinky should respond with "Pinky says: ..."
    let content = message["content"].as_str().unwrap();
    assert!(content.contains("Pinky says:"), "Expected Pinky response");

    // Cleanup
    let _ = server.kill();
}

#[async_std::test]
async fn test_embeddings() {
    let (mut server, base_url) = setup_test_server(9005).await;

    let request = serde_json::json!({
        "model": "test-model",
        "input": "Hello world"
    });

    let response: serde_json::Value = surf::post(format!("{}/v1/embeddings", base_url))
        .body_json(&request)
        .unwrap()
        .recv_json()
        .await
        .expect("Failed to get embeddings");

    assert_eq!(response["object"], "list");
    assert!(response["data"].is_array());

    let data = response["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let embedding_data = &data[0];
    assert_eq!(embedding_data["object"], "embedding");
    assert_eq!(embedding_data["index"], 0);

    // Verify embedding vector
    let embedding = embedding_data["embedding"].as_array().unwrap();
    assert_eq!(
        embedding.len(),
        384,
        "Pinky should return 384-dim embeddings"
    );

    // Verify values are floats
    for value in embedding {
        assert!(value.is_f64() || value.is_i64());
    }

    // Cleanup
    let _ = server.kill();
}

#[async_std::test]
async fn test_config_endpoint() {
    let (mut server, base_url) = setup_test_server(9007).await;

    let response: serde_json::Value = surf::get(format!("{}/v1/config", base_url))
        .recv_json()
        .await
        .expect("Failed to get config");

    assert!(response["ws_addr"].is_string());
    let ws_addr = response["ws_addr"].as_str().unwrap();
    assert!(
        ws_addr.contains("127.0.0.1"),
        "Should contain localhost address"
    );

    // Cleanup
    let _ = server.kill();
}
