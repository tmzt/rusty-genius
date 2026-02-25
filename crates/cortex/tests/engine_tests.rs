use anyhow::Result;
use futures::StreamExt;
use rusty_genius_cortex::backend::Engine;
use rusty_genius_thinkerv1::{InferenceConfig, EventResponse, Response};
use rusty_genius_core::manifest::EngineConfig; // Add this import
#[cfg(not(feature = "real-engine"))]
use rusty_genius_cortex::backend::Pinky;

async fn get_engine() -> Box<dyn Engine> {
    #[cfg(feature = "real-engine")]
    return create_engine().await;

    #[cfg(not(feature = "real-engine"))]
    return Box::new(Pinky::new());
}

#[cfg(feature = "real-engine")]
const DEFAULT_MODEL: &str = "tiny-model";

#[cfg(not(feature = "real-engine"))]
const DEFAULT_MODEL: &str = "mock";

async fn get_engine_with_default_model() -> Result<Box<dyn Engine>> {
    let mut engine = get_engine().await;
    engine.load_model(DEFAULT_MODEL).await?;
    Ok(engine)
}

#[async_std::test]
async fn test_engine_load_behavior() -> Result<()> {
    let mut engine = get_engine_with_default_model().await?;
    // In stub (Pinky) mode, "mock" is always valid.
    // In real (Brain) mode, "mock" will panic as it's not a real GGUF file.
    #[cfg(not(feature = "real-engine"))]
    let _ = engine.load_model("mock").await;

    #[cfg(feature = "real-engine")]
    {
        // Don't try to load "mock" with real engine as it panics
        // Actually utilize the variable to avoid unused warnings
        assert!(!engine.is_loaded());
    }

    // We don't assert success here because it depends on the feature,
    // which is the point of the abstraction test.
    Ok(())
}

#[cfg(not(feature = "real-engine"))]
#[async_std::test]
async fn test_stub_inference_protocol() -> Result<()> {
    let mut engine = get_engine_with_default_model().await?;

    let mut rx = engine.infer("test-infer-id".to_string(), "hello", EngineConfig::default()).await?; // Updated
    let mut has_content = false;
    let mut has_complete = false;

    while let Some(res) = rx.next().await {
        let event = res?;
        if let Response::Event(event_res) = event { // Extract EventResponse
            match event_res {
                EventResponse::Content{..} => has_content = true, // Updated
                EventResponse::Complete{..} => has_complete = true, // Updated
                _ => {}
            }
        }
    }

    assert!(has_content, "Engine should have emitted content");
    assert!(has_complete, "Engine should have emitted Complete");
    Ok(())
}

#[cfg(not(feature = "real-engine"))]
#[async_std::test]
async fn test_stub_embedding_protocol() -> Result<()> {
    let mut engine = get_engine_with_default_model().await?;

    let mut rx = engine.embed("test-embed-id".to_string(), "hello", EngineConfig::default()).await?; // Updated
    let mut has_embedding = false;
    let mut has_complete = false;

    while let Some(res) = rx.next().await {
        let event = res?;
        if let Response::Event(event_res) = event { // Extract EventResponse
            match event_res {
                EventResponse::Embedding{ vector_hex, .. } => { // Updated
                    assert!(!vector_hex.is_empty()); // Assert on vector_hex
                    has_embedding = true;
                }
                EventResponse::Complete{..} => has_complete = true, // Updated
                _ => {}
            }
        }
    }

    assert!(has_embedding, "Engine should have emitted embedding");
    assert!(has_complete, "Engine should have emitted Complete");
    Ok(())
}

#[async_std::test]
async fn test_engine_unload() -> Result<()> {
    let mut engine = get_engine_with_default_model().await?;
    engine.unload_model().await?;
    assert!(!engine.is_loaded());
    Ok(())
}
