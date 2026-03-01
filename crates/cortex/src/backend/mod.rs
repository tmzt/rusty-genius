mod engine_real;
mod engine_stub;

#[cfg(feature = "genai")]
mod engine_genai;

pub use rusty_genius_core::engine::Engine;

#[cfg(feature = "real-engine")]
pub use engine_real::Brain;

#[cfg(not(feature = "real-engine"))]
pub use engine_stub::Pinky;

#[cfg(feature = "genai")]
pub use engine_genai::{GeminiApiConfig, GeminiEngine};

// Re-export URL / body builders for testing and advanced use
#[cfg(feature = "genai")]
pub use engine_genai::{build_embed_body, build_infer_body, embed_url, infer_url, parse_sse_line};

pub async fn create_engine() -> Box<dyn Engine> {
    #[cfg(feature = "real-engine")]
    {
        Box::new(Brain::new())
    }

    #[cfg(not(feature = "real-engine"))]
    {
        Box::new(Pinky::new())
    }
}
