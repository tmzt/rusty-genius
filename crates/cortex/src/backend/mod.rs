mod engine_real;
mod engine_stub;

pub use crate::Engine;

#[cfg(feature = "real-engine")]
pub use engine_real::Brain;

#[cfg(not(feature = "real-engine"))]
pub use engine_stub::Pinky;

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
