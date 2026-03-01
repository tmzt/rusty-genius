#![cfg(feature = "cortex-engine-genai")]

use rusty_genius_cortex::backend::{
    build_embed_body, build_infer_body, embed_url, infer_url, parse_sse_line, GeminiApiConfig,
    GeminiEngine, Engine,
};
use rusty_genius_core::manifest::InferenceConfig;

// ── URL construction tests ──

#[test]
fn test_ai_studio_infer_url() {
    let config = GeminiApiConfig::AiStudio {
        api_key: "test-key".to_string(),
    };
    let url = infer_url(&config, "gemini-2.0-flash");
    assert_eq!(
        url,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
    );
}

#[test]
fn test_ai_studio_embed_url() {
    let config = GeminiApiConfig::AiStudio {
        api_key: "test-key".to_string(),
    };
    let url = embed_url(&config, "text-embedding-004");
    assert_eq!(
        url,
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
    );
}

#[test]
fn test_vertex_ai_infer_url() {
    let config = GeminiApiConfig::VertexAi {
        project_id: "my-project".to_string(),
        location: "us-central1".to_string(),
        access_token: "tok".to_string(),
    };
    let url = infer_url(&config, "gemini-2.0-flash");
    assert_eq!(
        url,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
    );
}

#[test]
fn test_vertex_ai_embed_url() {
    let config = GeminiApiConfig::VertexAi {
        project_id: "my-project".to_string(),
        location: "europe-west4".to_string(),
        access_token: "tok".to_string(),
    };
    let url = embed_url(&config, "text-embedding-004");
    assert_eq!(
        url,
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project/locations/europe-west4/publishers/google/models/text-embedding-004:embedContent"
    );
}

// ── JSON body construction tests ──

#[test]
fn test_infer_body_structure() {
    let config = InferenceConfig {
        temperature: 0.5,
        top_p: Some(0.95),
        top_k: Some(40),
        max_tokens: Some(1024),
        ..InferenceConfig::default()
    };
    let body = build_infer_body("Hello world", &config);

    // Validate contents
    let contents = body["contents"].as_array().expect("contents array");
    assert_eq!(contents.len(), 1);
    assert_eq!(contents[0]["role"], "user");
    assert_eq!(contents[0]["parts"][0]["text"], "Hello world");

    // Validate generation config
    let gen = &body["generationConfig"];
    assert!((gen["temperature"].as_f64().unwrap() - 0.5).abs() < 0.001);
    assert!((gen["topP"].as_f64().unwrap() - 0.95).abs() < 0.001);
    assert_eq!(gen["topK"], 40);
    assert_eq!(gen["maxOutputTokens"], 1024);
}

#[test]
fn test_infer_body_omits_none_fields() {
    let config = InferenceConfig {
        temperature: 0.7,
        top_p: None,
        top_k: None,
        max_tokens: None,
        ..InferenceConfig::default()
    };
    let body = build_infer_body("test", &config);

    let gen = &body["generationConfig"];
    assert!(gen["topP"].is_null(), "topP should be omitted when None");
    assert!(gen["topK"].is_null(), "topK should be omitted when None");
    assert!(
        gen["maxOutputTokens"].is_null(),
        "maxOutputTokens should be omitted when None"
    );
}

#[test]
fn test_embed_body_structure() {
    let body = build_embed_body("encode this text");

    let parts = body["content"]["parts"].as_array().expect("parts array");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0]["text"], "encode this text");
}

// ── SSE parsing tests ──

#[test]
fn test_parse_sse_content_chunk() {
    let line = r#"data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]},"finishReason":null}]}"#;
    let (text, finish, is_thought) = parse_sse_line(line).expect("should parse");
    assert_eq!(text, Some("Hello".to_string()));
    assert!(finish.is_none());
    assert!(!is_thought);
}

#[test]
fn test_parse_sse_thought_chunk() {
    let line = r#"data: {"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true}]},"finishReason":null}]}"#;
    let (text, finish, is_thought) = parse_sse_line(line).expect("should parse");
    assert_eq!(text, Some("thinking...".to_string()));
    assert!(finish.is_none());
    assert!(is_thought);
}

#[test]
fn test_parse_sse_stop_finish() {
    let line =
        r#"data: {"candidates":[{"content":{"parts":[{"text":"done"}]},"finishReason":"STOP"}]}"#;
    let (text, finish, _) = parse_sse_line(line).expect("should parse");
    assert_eq!(text, Some("done".to_string()));
    assert_eq!(finish, Some("STOP".to_string()));
}

#[test]
fn test_parse_sse_empty_text() {
    let line =
        r#"data: {"candidates":[{"content":{"parts":[{}]},"finishReason":"STOP"}]}"#;
    let (text, finish, _) = parse_sse_line(line).expect("should parse");
    assert!(text.is_none());
    assert_eq!(finish, Some("STOP".to_string()));
}

#[test]
fn test_parse_sse_non_data_line() {
    assert!(parse_sse_line(": keep-alive").is_none());
    assert!(parse_sse_line("").is_none());
    assert!(parse_sse_line("event: message").is_none());
}

#[test]
fn test_parse_sse_invalid_json() {
    assert!(parse_sse_line("data: not-json").is_none());
}

// ── Engine state machine tests ──

#[test]
fn test_default_model() {
    let engine = GeminiEngine::new(GeminiApiConfig::AiStudio {
        api_key: "k".to_string(),
    });
    assert_eq!(engine.default_model(), "gemini-2.0-flash");
}

#[test]
fn test_engine_not_loaded_initially() {
    let engine = GeminiEngine::new(GeminiApiConfig::AiStudio {
        api_key: "k".to_string(),
    });
    assert!(!engine.is_loaded());
}

#[smol_potat::test]
async fn test_engine_load_unload_cycle() {
    let mut engine = GeminiEngine::new(GeminiApiConfig::AiStudio {
        api_key: "k".to_string(),
    });
    assert!(!engine.is_loaded());

    engine.load_model("gemini-2.0-flash").await.unwrap();
    assert!(engine.is_loaded());

    engine.unload_model().await.unwrap();
    assert!(!engine.is_loaded());
}

#[smol_potat::test]
async fn test_engine_load_sets_model_name() {
    let mut engine = GeminiEngine::new(GeminiApiConfig::VertexAi {
        project_id: "p".to_string(),
        location: "us-east1".to_string(),
        access_token: "t".to_string(),
    });
    engine.load_model("gemini-1.5-pro").await.unwrap();
    assert!(engine.is_loaded());
    // The default_model is fixed, but internal model should be updated
    assert_eq!(engine.default_model(), "gemini-2.0-flash");
}

#[smol_potat::test]
async fn test_infer_without_load_errors() {
    let mut engine = GeminiEngine::new(GeminiApiConfig::AiStudio {
        api_key: "k".to_string(),
    });
    let result = engine.infer("test", InferenceConfig::default()).await;
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("no model loaded"),
        "Should mention no model loaded"
    );
}

#[smol_potat::test]
async fn test_embed_without_load_errors() {
    let mut engine = GeminiEngine::new(GeminiApiConfig::AiStudio {
        api_key: "k".to_string(),
    });
    let result = engine.embed("test", InferenceConfig::default()).await;
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("no model loaded"),
        "Should mention no model loaded"
    );
}
