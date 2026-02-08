use async_std::process::{Child, Command};
use async_std::task;
use std::process::Stdio;
use std::time::Duration;

const INFER_MODEL: &str = "tiny-llama";
const EMBED_MODEL: &str = "embedding-gemma";

struct TestServer {
    child: Child,
    base_url: String,
}

impl TestServer {
    async fn new(port: u16) -> Self {
        let addr = format!("127.0.0.1:{}", port);
        let ws_addr = format!("127.0.0.1:{}", port + 1);

        let binary_path = if let Ok(test_binary) = std::env::var("TEST_BINARY") {
            test_binary
        } else {
            let release_path = "../../target/release/ogenius";
            let debug_path = "../../target/debug/ogenius";
            if std::path::Path::new(release_path).exists() {
                release_path.to_string()
            } else {
                debug_path.to_string()
            }
        };

        let child = Command::new(&binary_path)
            .args(&["serve", "--addr", &addr, "--ws-addr", &ws_addr, "--no-open"])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to start ogenius");

        let base_url = format!("http://{}", addr);
        // Wait for server to start
        for _ in 0..50 {
            task::sleep(Duration::from_millis(200)).await;
            if surf::get(&format!("{}/v1/models", base_url)).await.is_ok() {
                return Self { child, base_url };
            }
        }
        panic!("Server failed to start on {}", addr);
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[async_std::test]
async fn test_list_models() {
    let server = TestServer::new(10001).await;
    let response: serde_json::Value = surf::get(format!("{}/v1/models", server.base_url))
        .recv_json()
        .await
        .expect("Failed to get models");

    assert_eq!(response["object"], "list");
    assert!(response["data"].is_array());
}

#[async_std::test]
async fn test_chat_completions() {
    let server = TestServer::new(10003).await;
    let request = serde_json::json!({
        "model": INFER_MODEL,
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": false
    });

    let response: serde_json::Value =
        surf::post(format!("{}/v1/chat/completions", server.base_url))
            .body_json(&request)
            .unwrap()
            .recv_json()
            .await
            .expect("Failed to get chat completion");

    assert_eq!(response["object"], "chat.completion");
}

#[async_std::test]
async fn test_embeddings() {
    let server = TestServer::new(10005).await;
    let request = serde_json::json!({
        "model": EMBED_MODEL,
        "input": "Hello world"
    });

    let response: serde_json::Value = surf::post(format!("{}/v1/embeddings", server.base_url))
        .body_json(&request)
        .unwrap()
        .recv_json()
        .await
        .expect("Failed to get embeddings");

    assert_eq!(response["object"], "list");
    assert!(response["data"].is_array());
}

#[async_std::test]
async fn test_config_endpoint() {
    let server = TestServer::new(10007).await;
    let response: serde_json::Value = surf::get(format!("{}/v1/config", server.base_url))
        .recv_json()
        .await
        .expect("Failed to get config");

    assert!(response["ws_addr"].is_string());
}

#[async_std::test]
async fn test_reset_engine() {
    let server = TestServer::new(10009).await;
    // Reset should return 200 OK
    let response = surf::post(format!("{}/v1/engine/reset", server.base_url))
        .await
        .expect("Failed to reset engine");

    assert_eq!(response.status(), surf::StatusCode::Ok);
}
