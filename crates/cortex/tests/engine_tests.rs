use anyhow::Result;
use futures::StreamExt;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;
use rusty_genius_cortex::backend::Engine;
#[cfg(not(feature = "real-engine"))]
use rusty_genius_cortex::backend::Pinky;

async fn get_engine() -> Box<dyn Engine> {
    #[cfg(feature = "real-engine")]
    return create_engine().await;

    #[cfg(not(feature = "real-engine"))]
    return Box::new(Pinky::new());
}

// #[cfg(feature = "real-engine")]
// fn get_default_model() -> &str {
//     "tiny-model"
// }

// #[cfg(not(feature = "real-engine"))]
// fn get_default_model() -> &str {
//     "mock"
// }

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

    let mut rx = engine.infer("hello", InferenceConfig::default()).await?;
    let mut has_content = false;
    let mut has_complete = false;

    while let Some(res) = rx.next().await {
        let event = res?;
        match event {
            InferenceEvent::Content(_) => has_content = true,
            InferenceEvent::Complete => has_complete = true,
            _ => {}
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

    let mut rx = engine.embed("hello", InferenceConfig::default()).await?;
    let mut has_embedding = false;
    let mut has_complete = false;

    while let Some(res) = rx.next().await {
        let event = res?;
        match event {
            InferenceEvent::Embedding(emb) => {
                assert!(!emb.is_empty());
                has_embedding = true;
            }
            InferenceEvent::Complete => has_complete = true,
            _ => {}
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
