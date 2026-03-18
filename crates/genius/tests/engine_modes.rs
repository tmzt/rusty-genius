//! Tests for rusty-genius in various engine modes (embed, infer).
//!
//! Tests gated behind `real-models` download and cache actual model files.
//! Stub tests run without any models.
//!
//! Run stub tests:
//!   cargo test -p rusty-genius --test engine_modes
//!
//! Run real model tests:
//!   cargo test -p rusty-genius --test engine_modes --features real-models -- --nocapture

use futures::StreamExt;
use rusty_genius::Genius;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;

// ── Helpers ──

fn init_logging() {
    let _ = env_logger::builder().is_test(true).try_init();
}

async fn collect_inference(genius: &mut Genius, model: Option<String>, prompt: &str) -> (String, bool) {
    let config = InferenceConfig {
        max_tokens: Some(64),
        ..InferenceConfig::default()
    };
    let mut rx = genius
        .infer(model, prompt.to_string(), config)
        .await
        .expect("infer call failed");

    let mut content = String::new();
    let mut saw_complete = false;
    while let Some(event) = rx.next().await {
        match event {
            InferenceEvent::Content(tok) => content.push_str(&tok),
            InferenceEvent::Complete => { saw_complete = true; break; }
            _ => {}
        }
    }
    (content, saw_complete)
}

async fn collect_embedding(genius: &mut Genius, model: Option<String>, input: &str) -> (Vec<f32>, bool) {
    let config = InferenceConfig {
        context_size: Some(512),
        ..InferenceConfig::default()
    };
    let mut rx = genius
        .embed(model, input.to_string(), config)
        .await
        .expect("embed call failed");

    let mut embedding = Vec::new();
    let mut saw_complete = false;
    while let Some(event) = rx.next().await {
        match event {
            InferenceEvent::Embedding(vec) => embedding = vec,
            InferenceEvent::Complete => { saw_complete = true; break; }
            _ => {}
        }
    }
    (embedding, saw_complete)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-8 && norm_b > 1e-8 { dot / (norm_a * norm_b) } else { 0.0 }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Stub tests (no real models needed) ──

#[test]
fn test_genius_init() {
    init_logging();
    smol::block_on(async {
        let genius = Genius::new().await;
        assert!(genius.is_ok(), "Genius::new() should succeed: {:?}", genius.err());
    });
}

#[test]
fn test_genius_with_engine_name_default() {
    init_logging();
    smol::block_on(async {
        let genius = Genius::with_engine_name("default").await;
        assert!(genius.is_ok(), "with_engine_name('default') should succeed");
    });
}

#[test]
fn test_genius_with_engine_name_invalid() {
    init_logging();
    smol::block_on(async {
        let genius = Genius::with_engine_name("nonexistent").await;
        assert!(genius.is_err(), "with_engine_name('nonexistent') should fail");
    });
}

#[test]
fn test_genius_resident_tracking() {
    init_logging();
    smol::block_on(async {
        let genius = Genius::new().await.expect("init failed");
        assert!(genius.list_resident().is_empty(), "should start with no resident contexts");
    });
}

// ── Real model tests: embedding ──

#[cfg(feature = "real-models")]
mod embed {
    use super::*;

    const EMBEDDING_MODEL: &str = "nomic-embed-text";

    #[test]
    fn test_embed_basic() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            let (emb, complete) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "hello world",
            ).await;

            assert!(complete, "should receive Complete event");
            assert_eq!(emb.len(), 768, "nomic-embed-text produces 768-dim embeddings, got {}", emb.len());

            let norm = l2_norm(&emb);
            assert!((norm - 1.0).abs() < 0.1, "embedding should be normalized, got norm={norm}");
        });
    }

    #[test]
    fn test_embed_similarity() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            let (emb_cat, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "the cat sat on the mat",
            ).await;

            let (emb_cat2, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "a cat is sitting on a mat",
            ).await;

            let (emb_quantum, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "quantum mechanics wave function",
            ).await;

            let sim_similar = cosine_similarity(&emb_cat, &emb_cat2);
            let sim_diff = cosine_similarity(&emb_cat, &emb_quantum);

            eprintln!("similar pair: {sim_similar:.4}, dissimilar pair: {sim_diff:.4}");
            assert!(sim_similar > 0.5, "similar texts should have cosine > 0.5, got {sim_similar}");
            assert!(sim_similar > sim_diff, "similar pair should score higher than dissimilar");
        });
    }

    #[test]
    fn test_embed_deterministic() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            let (emb1, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "deterministic test",
            ).await;

            let (emb2, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "deterministic test",
            ).await;

            let sim = cosine_similarity(&emb1, &emb2);
            assert!(sim > 0.999, "same input should produce near-identical embeddings, got {sim}");
        });
    }

    #[test]
    fn test_embed_edge_cases() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            // Single character
            let (emb, _) = collect_embedding(&mut genius, Some(EMBEDDING_MODEL.to_string()), "a").await;
            assert_eq!(emb.len(), 768);

            // Long text
            let long = "The quick brown fox. ".repeat(50);
            let (emb, _) = collect_embedding(&mut genius, Some(EMBEDDING_MODEL.to_string()), &long).await;
            assert_eq!(emb.len(), 768);

            // Unicode
            let (emb, _) = collect_embedding(&mut genius, Some(EMBEDDING_MODEL.to_string()), "café résumé naïve 日本語").await;
            assert_eq!(emb.len(), 768);
        });
    }
}

// ── Real model tests: inference ──

#[cfg(feature = "real-models")]
mod infer {
    use super::*;

    // Use qwen (works); function-gemma has a KV cache bug in current llama.cpp
    const INFER_MODEL: &str = "qwen-2.5-1.5b-instruct";

    const ROUTER_MODEL: &str =
        "lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf";

    #[test]
    fn test_infer_basic() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(INFER_MODEL.to_string(), "infer".to_string())
                .await
                .expect("preload failed");

            let (content, complete) = collect_inference(
                &mut genius,
                Some(INFER_MODEL.to_string()),
                "What is 2 + 2?",
            ).await;

            assert!(complete, "should receive Complete event");
            assert!(!content.is_empty(), "should produce some output, got empty");
            eprintln!("infer output ({} chars): {:?}", content.len(), &content[..content.len().min(200)]);
        });
    }

    #[test]
    fn test_infer_sequential() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(INFER_MODEL.to_string(), "infer".to_string())
                .await
                .expect("preload failed");

            for prompt in &["hello", "what is rust", "explain gravity briefly"] {
                let (content, complete) = collect_inference(
                    &mut genius,
                    Some(INFER_MODEL.to_string()),
                    prompt,
                ).await;
                assert!(complete, "should complete for prompt: {prompt}");
                assert!(!content.is_empty(), "should produce output for prompt: {prompt}");
            }
        });
    }

    #[test]
    fn test_infer_router_model() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(ROUTER_MODEL.to_string(), "infer".to_string())
                .await
                .expect("preload failed");

            let (content, complete) = collect_inference(
                &mut genius,
                Some(ROUTER_MODEL.to_string()),
                "What is 2 + 2?",
            ).await;

            assert!(complete, "should receive Complete event");
            assert!(!content.is_empty(), "should produce some output");
        });
    }
}

// ── Real model tests: preload and resident tracking ──

#[cfg(feature = "real-models")]
mod resident {
    use super::*;
    use rusty_genius::ContextPurpose;

    const EMBEDDING_MODEL: &str = "nomic-embed-text";
    const INFER_MODEL: &str = "qwen-2.5-1.5b-instruct";
    const ROUTER_MODEL: &str =
        "lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf";

    #[test]
    fn test_preload_registers_resident() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            assert!(!genius.is_resident(EMBEDDING_MODEL, ContextPurpose::Embed));

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            assert!(genius.is_resident(EMBEDDING_MODEL, ContextPurpose::Embed));
        });
    }

    #[test]
    fn test_preload_multiple_models() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius.preload(EMBEDDING_MODEL.to_string(), "embed".to_string()).await.expect("embed preload");
            genius.preload(INFER_MODEL.to_string(), "infer".to_string()).await.expect("infer preload");

            assert!(genius.is_resident(EMBEDDING_MODEL, ContextPurpose::Embed));
            assert!(genius.is_resident(INFER_MODEL, ContextPurpose::Infer));
            assert_eq!(genius.list_resident().len(), 2);
        });
    }

    #[test]
    fn test_keep_resident_indefinite() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .keep_resident(EMBEDDING_MODEL.to_string(), "embed".to_string(), None)
                .await
                .expect("keep_resident failed");

            let ctx = genius.get_resident(EMBEDDING_MODEL, ContextPurpose::Embed);
            assert!(ctx.is_some(), "should be resident");
            assert!(!ctx.unwrap().is_expired(), "indefinite should not expire");
        });
    }

    #[test]
    fn test_embed_after_preload() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("preload failed");

            // Embedding should work after preload
            let (emb, complete) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "test after preload",
            ).await;

            assert!(complete);
            assert_eq!(emb.len(), 768);
        });
    }

    #[test]
    fn test_infer_after_preload() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::new().await.expect("init failed");

            genius
                .preload(INFER_MODEL.to_string(), "infer".to_string())
                .await
                .expect("preload failed");

            let (content, complete) = collect_inference(
                &mut genius,
                Some(INFER_MODEL.to_string()),
                "hello",
            ).await;

            assert!(complete);
            assert!(!content.is_empty());
        });
    }
}

// ── MLX engine tests ──

#[cfg(feature = "mlx-models")]
mod mlx {
    use super::*;

    const MLX_MODEL: &str = "mlx-community/Qwen3.5-9B-MLX-4bit";
    const EMBEDDING_MODEL: &str = "nomic-embed-text";

    #[test]
    fn test_mlx_engine_init() {
        init_logging();
        smol::block_on(async {
            let genius = Genius::with_engine_name("mlx").await;
            assert!(genius.is_ok(), "MLX engine should init: {:?}", genius.err());
        });
    }

    #[test]
    fn test_mlx_embed_via_dispatch() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::with_engine_name("mlx").await.expect("init failed");

            // Embedding should route to llama.cpp via DispatchEngine
            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("embed preload failed");

            let (emb, complete) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "test mlx dispatch embedding",
            ).await;

            assert!(complete, "should get Complete from llama.cpp embed path");
            assert_eq!(emb.len(), 768, "nomic-embed-text should produce 768-dim");
        });
    }

    #[test]
    #[ignore = "Qwen3.5 hybrid attention architecture not yet implemented in MLX engine"]
    fn test_mlx_infer_basic() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::with_engine_name("mlx").await.expect("init failed");

            genius
                .preload(MLX_MODEL.to_string(), "infer".to_string())
                .await
                .expect("MLX model preload failed");

            let (content, complete) = collect_inference(
                &mut genius,
                Some(MLX_MODEL.to_string()),
                "What is 2 + 2? Answer in one word.",
            ).await;

            assert!(complete, "should receive Complete event");
            assert!(!content.is_empty(), "MLX model should produce output");
            eprintln!("MLX output ({} chars): {:?}", content.len(), &content[..content.len().min(200)]);
        });
    }

    #[test]
    #[ignore = "Qwen3.5 hybrid attention architecture not yet implemented in MLX engine"]
    fn test_mlx_dispatch_both() {
        init_logging();
        smol::block_on(async {
            let mut genius = Genius::with_engine_name("mlx").await.expect("init failed");

            // Preload both: embed goes to llama, infer goes to MLX
            genius
                .preload(EMBEDDING_MODEL.to_string(), "embed".to_string())
                .await
                .expect("embed preload failed");
            genius
                .preload(MLX_MODEL.to_string(), "infer".to_string())
                .await
                .expect("MLX preload failed");

            // Embed via llama.cpp
            let (emb, _) = collect_embedding(
                &mut genius,
                Some(EMBEDDING_MODEL.to_string()),
                "dispatch test",
            ).await;
            assert_eq!(emb.len(), 768);

            // Infer via MLX
            let (content, complete) = collect_inference(
                &mut genius,
                Some(MLX_MODEL.to_string()),
                "Say hello in one sentence.",
            ).await;
            assert!(complete);
            assert!(!content.is_empty());
            eprintln!("MLX+llama dispatch: embed=768d, infer={} chars", content.len());
        });
    }
}
