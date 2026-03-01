#![cfg(feature = "wllama")]

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceEvent,
};
use rusty_genius_stem::engine_wllama::WllamaEngine;
use rusty_genius_stem::{CortexStrategy, Orchestrator};
use std::time::Duration;

fn load_wasm_engine() -> WllamaEngine {
    let wasm_bytes = include_bytes!("../../../target/wasm32-wasip1/release/wasm_guest.wasm");
    WllamaEngine::from_wasm_bytes(wasm_bytes).expect("failed to load wasm guest module")
}

// ── Direct Engine Tests ──

#[test]
fn test_wllama_engine_load_model() {
    smol::block_on(async {
        let mut engine = load_wasm_engine();
        assert!(!engine.is_loaded());

        engine.load_model("test-model").await.unwrap();
        assert!(engine.is_loaded());
    });
}

#[test]
fn test_wllama_engine_unload_model() {
    smol::block_on(async {
        let mut engine = load_wasm_engine();

        engine.load_model("test-model").await.unwrap();
        assert!(engine.is_loaded());

        engine.unload_model().await.unwrap();
        assert!(!engine.is_loaded());
    });
}

#[test]
fn test_wllama_engine_infer_streaming() {
    smol::block_on(async {
        let mut engine = load_wasm_engine();
        engine.load_model("test-model").await.unwrap();

        let config = InferenceConfig::default();
        let mut rx = engine.infer("hello world test", config).await.unwrap();

        let mut events = vec![];
        while let Some(event) = rx.next().await {
            events.push(event.unwrap());
        }

        // Should start with ProcessStart
        assert!(
            matches!(events.first(), Some(InferenceEvent::ProcessStart)),
            "expected ProcessStart, got {:?}",
            events.first()
        );

        // Should end with Complete
        assert!(
            matches!(events.last(), Some(InferenceEvent::Complete)),
            "expected Complete, got {:?}",
            events.last()
        );

        // Should have Content tokens between start and complete
        let content_tokens: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                InferenceEvent::Content(s) => Some(s.clone()),
                _ => None,
            })
            .collect();

        assert!(
            !content_tokens.is_empty(),
            "expected at least one Content token"
        );

        // The stub splits "hello world test" into words and emits each + <|end|>
        assert!(content_tokens.contains(&"hello".to_string()));
        assert!(content_tokens.contains(&"world".to_string()));
        assert!(content_tokens.contains(&"test".to_string()));
        assert!(content_tokens.contains(&"<|end|>".to_string()));
    });
}

#[test]
fn test_wllama_engine_infer_without_load() {
    smol::block_on(async {
        let mut engine = load_wasm_engine();
        // Do NOT load a model

        let config = InferenceConfig::default();
        let mut rx = engine.infer("hello", config).await.unwrap();

        // Should get ProcessStart, then an error (guest returns -1), then Complete
        let mut events = vec![];
        while let Some(event) = rx.next().await {
            events.push(event);
        }

        // Check that we got an error somewhere
        let has_error = events.iter().any(|e| e.is_err());
        assert!(has_error, "expected an error event when model not loaded");
    });
}

#[test]
fn test_wllama_engine_embed() {
    smol::block_on(async {
        let mut engine = load_wasm_engine();
        engine.load_model("test-model").await.unwrap();

        let config = InferenceConfig::default();
        let mut rx = engine.embed("test input", config).await.unwrap();

        let mut events = vec![];
        while let Some(event) = rx.next().await {
            events.push(event.unwrap());
        }

        // Should have ProcessStart
        assert!(matches!(events.first(), Some(InferenceEvent::ProcessStart)));

        // Should have an Embedding event
        let embedding = events.iter().find_map(|e| match e {
            InferenceEvent::Embedding(v) => Some(v.clone()),
            _ => None,
        });
        assert!(embedding.is_some(), "expected Embedding event");

        let emb = embedding.unwrap();
        assert_eq!(emb.len(), 384, "expected 384-dim embedding");

        // Verify deterministic values: (0..384).map(|i| (i as f32 * 0.01).sin())
        let expected_first = (0.0_f32 * 0.01).sin();
        let expected_second = (1.0_f32 * 0.01).sin();
        assert!((emb[0] - expected_first).abs() < 1e-6);
        assert!((emb[1] - expected_second).abs() < 1e-6);

        // Should end with Complete
        assert!(matches!(events.last(), Some(InferenceEvent::Complete)));
    });
}

// ── Orchestrator Tests ──

/// Helper: create an Orchestrator with the WASM engine
fn make_orchestrator() -> Orchestrator {
    let engine = load_wasm_engine();
    Orchestrator::with_engine(Box::new(engine))
}

#[test]
fn test_orchestrator_wllama_full_flow() {
    smol::block_on(async {
        let mut orch = make_orchestrator();
        orch.set_strategy(CortexStrategy::KeepAlive);

        let (mut in_tx, in_rx) = mpsc::channel::<BrainstemInput>(16);
        let (out_tx, mut out_rx) = mpsc::channel::<BrainstemOutput>(64);

        // Run orchestrator in background
        let orch_handle = smol::spawn(async move {
            orch.run(in_rx, out_tx).await.unwrap();
        });

        // 1. LoadModel
        in_tx
            .send(BrainstemInput {
                id: Some("r1".into()),
                command: BrainstemCommand::LoadModel("test-model".into()),
            })
            .await
            .unwrap();

        // Wait for load to complete (no output for successful load in non-cortex path)
        smol::Timer::after(Duration::from_millis(50)).await;

        // 2. Infer
        in_tx
            .send(BrainstemInput {
                id: Some("r2".into()),
                command: BrainstemCommand::Infer {
                    model: None,
                    prompt: "hello world".into(),
                    config: InferenceConfig::default(),
                },
            })
            .await
            .unwrap();

        // Collect inference events
        let mut infer_events = vec![];
        loop {
            let msg = smol::future::or(
                async {
                    out_rx.next().await
                },
                async {
                    smol::Timer::after(Duration::from_secs(2)).await;
                    None
                },
            )
            .await;

            match msg {
                Some(output) => {
                    let is_complete =
                        matches!(&output.body, BrainstemBody::Event(InferenceEvent::Complete));
                    infer_events.push(output);
                    if is_complete {
                        break;
                    }
                }
                None => break,
            }
        }

        assert!(
            !infer_events.is_empty(),
            "expected inference events from orchestrator"
        );

        // Verify we got ProcessStart and Complete
        let has_start = infer_events.iter().any(|e| {
            matches!(&e.body, BrainstemBody::Event(InferenceEvent::ProcessStart))
        });
        let has_complete = infer_events.iter().any(|e| {
            matches!(&e.body, BrainstemBody::Event(InferenceEvent::Complete))
        });
        assert!(has_start, "expected ProcessStart event");
        assert!(has_complete, "expected Complete event");

        // 3. Reset
        in_tx
            .send(BrainstemInput {
                id: Some("r3".into()),
                command: BrainstemCommand::Reset,
            })
            .await
            .unwrap();

        // Wait for reset response
        let reset_msg = smol::future::or(
            async { out_rx.next().await },
            async {
                smol::Timer::after(Duration::from_secs(1)).await;
                None
            },
        )
        .await;
        assert!(reset_msg.is_some(), "expected reset response");

        // 4. Stop
        in_tx
            .send(BrainstemInput {
                id: Some("r4".into()),
                command: BrainstemCommand::Stop,
            })
            .await
            .unwrap();

        orch_handle.await;
    });
}

#[test]
fn test_orchestrator_wllama_cold_reload() {
    smol::block_on(async {
        let mut orch = make_orchestrator();
        orch.set_strategy(CortexStrategy::KeepAlive);

        let (mut in_tx, in_rx) = mpsc::channel::<BrainstemInput>(16);
        let (out_tx, mut out_rx) = mpsc::channel::<BrainstemOutput>(64);

        let orch_handle = smol::spawn(async move {
            orch.run(in_rx, out_tx).await.unwrap();
        });

        // Infer WITHOUT pre-loading — orchestrator should auto-load default model
        in_tx
            .send(BrainstemInput {
                id: Some("cold".into()),
                command: BrainstemCommand::Infer {
                    model: None,
                    prompt: "cold start test".into(),
                    config: InferenceConfig::default(),
                },
            })
            .await
            .unwrap();

        // Collect events
        let mut events = vec![];
        loop {
            let msg = smol::future::or(
                async { out_rx.next().await },
                async {
                    smol::Timer::after(Duration::from_secs(2)).await;
                    None
                },
            )
            .await;

            match msg {
                Some(output) => {
                    let is_complete =
                        matches!(&output.body, BrainstemBody::Event(InferenceEvent::Complete));
                    events.push(output);
                    if is_complete {
                        break;
                    }
                }
                None => break,
            }
        }

        // Should have succeeded with auto-loaded model
        let has_complete = events.iter().any(|e| {
            matches!(&e.body, BrainstemBody::Event(InferenceEvent::Complete))
        });
        assert!(has_complete, "expected Complete after cold reload");

        // Stop
        in_tx
            .send(BrainstemInput {
                id: Some("stop".into()),
                command: BrainstemCommand::Stop,
            })
            .await
            .unwrap();

        orch_handle.await;
    });
}

#[test]
fn test_orchestrator_wllama_multiple_infers() {
    smol::block_on(async {
        let mut orch = make_orchestrator();
        orch.set_strategy(CortexStrategy::KeepAlive);

        let (mut in_tx, in_rx) = mpsc::channel::<BrainstemInput>(16);
        let (out_tx, mut out_rx) = mpsc::channel::<BrainstemOutput>(128);

        let orch_handle = smol::spawn(async move {
            orch.run(in_rx, out_tx).await.unwrap();
        });

        // Load model first
        in_tx
            .send(BrainstemInput {
                id: Some("load".into()),
                command: BrainstemCommand::LoadModel("test-model".into()),
            })
            .await
            .unwrap();

        smol::Timer::after(Duration::from_millis(50)).await;

        // Send 3 sequential infers
        for i in 0..3 {
            let req_id = format!("infer-{}", i);
            in_tx
                .send(BrainstemInput {
                    id: Some(req_id.clone()),
                    command: BrainstemCommand::Infer {
                        model: None,
                        prompt: format!("prompt {}", i),
                        config: InferenceConfig::default(),
                    },
                })
                .await
                .unwrap();

            // Wait for Complete
            loop {
                let msg = smol::future::or(
                    async { out_rx.next().await },
                    async {
                        smol::Timer::after(Duration::from_secs(2)).await;
                        None
                    },
                )
                .await;

                match msg {
                    Some(output) => {
                        if matches!(
                            &output.body,
                            BrainstemBody::Event(InferenceEvent::Complete)
                        ) {
                            break;
                        }
                    }
                    None => panic!("timed out waiting for infer-{} to complete", i),
                }
            }
        }

        // Stop
        in_tx
            .send(BrainstemInput {
                id: Some("stop".into()),
                command: BrainstemCommand::Stop,
            })
            .await
            .unwrap();

        orch_handle.await;
    });
}
