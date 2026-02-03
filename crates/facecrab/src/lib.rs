//! # Facecrab: The Supplier
//!
//! **Asset management, model registry, and high-performance LLM downloads.**
//!
//! Facecrab is an autonomous asset authority designed for the `rusty-genius` ecosystem,
//! but it is also usable as a standalone crate. It handles model resolution (via HuggingFace),
//! registry management, and background downloading with progress tracking.
//!
//! ## Core Features
//!
//! - **Registry Management**: Uses `registry.toml` to map friendly names to HuggingFace repositories.
//! - **HuggingFace Integration**: Automatically resolves and downloads GGUF assets.
//! - **Streaming Downloads**: Provides an event-based API for tracking download progress (bytes/total).
//! - **Local Caching**: Deduplicates downloads and manages assets in `~/.config/rusty-genius/`.
//!
//! ## Usage
//!
//! ### 1. Simple One-Shot Download
//!
//! For most use cases, you just want to get the local path to a model.
//!
//! ```no_run
//! use facecrab::AssetAuthority;
//!
//! #[async_std::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let authority = AssetAuthority::new()?;
//!
//!     // Resolves the name in the local registry or HuggingFace path.
//!     // Downloads the model if not already cached.
//!     let path = authority.ensure_model("qwen-2.5-3b-instruct").await?;
//!
//!     println!("Model available at: {:?}", path);
//!     Ok(())
//! }
//! ```
//!
//! ### 2. Event-Based Download (Progress Tracking)
//!
//! If you need to show a progress bar or handle download lifecycle events, use the streaming API.
//!
//! ```no_run
//! use facecrab::AssetAuthority;
//! use rusty_genius_core::protocol::AssetEvent;
//! use futures::StreamExt;
//!
//! #[async_std::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let authority = AssetAuthority::new()?;
//!     let mut events = authority.ensure_model_stream("qwen-2.5-1.5b-instruct");
//!
//!     while let Some(event) = events.next().await {
//!         match event {
//!             AssetEvent::Started(name) => println!("Starting download: {}", name),
//!             AssetEvent::Progress(current, total) => {
//!                 let pct = (current as f64 / total as f64) * 100.0;
//!                 print!("\rProgress: {:.2}% ({}/{})", pct, current, total);
//!             }
//!             AssetEvent::Complete(_) => println!("\nDownload finished!"),
//!             AssetEvent::Error(err) => eprintln!("Error: {}", err),
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

/// Logic for downloading and caching assets from remote sources.
pub mod assets;

/// Management of the local model registry and configuration.
pub mod registry;

pub use assets::AssetAuthority;
pub use registry::ModelRegistry;
