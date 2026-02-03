//! # Rusty-Genius: The Nervous System for AI
//!
//! **A high-performance, modular, local-first AI orchestration library written in Rust.**
//!
//! Rusty-Genius is built for **on-device orchestration**, prioritizing absolute privacy, zero latency,
//! and offline reliability. It decouples protocol, orchestration, engine, and tooling to provide a
//! flexible foundation for modern AI applications.
//!
//! ## Architecture
//!
//! The project follows a biological metaphor, where each component serves a specific function in the "nervous system":
//!
//! - **Genius** (this crate): The Public Facade. Re-exports internal crates and provides the primary user API.
//! - **Brainstem** ([`brainstem`]): The Orchestrator. Manages the central event loop, engine lifecycle (TTL), and state transitions.
//! - **Cortex** ([`cortex`]): The Muscle. Provides direct bindings to `llama.cpp` for inference, handling KV caching and token streaming.
//! - **Facecrab** ([`facecrab`]): The Supplier. An autonomous asset authority that handles model resolution (HuggingFace), registry management, and downloads.
//! - **Core** ([`core`]): The Shared Vocabulary. Contains protocol enums, manifests, and error definitions.
//!
//! ## Quick Start
//!
//! The most robust way to use Rusty-Genius is via the [`Orchestrator`]. It manages the background event loop,
//! model lifecycle (loading/unloading), and hardware stubs.
//!
//! ```no_run
//! use rusty_genius::Orchestrator;
//! use rusty_genius::core::protocol::{AssetEvent, BrainstemInput, BrainstemOutput, InferenceEvent};
//! use futures::{StreamExt, sink::SinkExt, channel::mpsc};
//!
//! #[async_std::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Initialize the orchestrator (with default 5m TTL)
//!     let mut genius = Orchestrator::new().await?;
//!     let (mut input, rx) = mpsc::channel(100);
//!     let (tx, mut output) = mpsc::channel(100);
//!
//!     // Spawn the Brainstem event loop in a background task
//!     async_std::task::spawn(async move {
//!         if let Err(e) = genius.run(rx, tx).await {
//!             eprintln!("Orchestrator error: {}", e);
//!         }
//!     });
//!
//!     // 2. Load a model (downloads from HuggingFace if not cached)
//!     // The AssetAuthority (Facecrab) handles resolution and downloading automatically.
//!     input.send(BrainstemInput::LoadModel(
//!         "tiny-model".into()
//!     )).await?;
//!
//!     // 3. Submit a prompt
//!     input.send(BrainstemInput::Infer {
//!         prompt: "Once upon a time...".into(),
//!         config: Default::default(),
//!     }).await?;
//!
//!     // 4. Stream results
//!     // The Cortex engine streams tokens back through the channel
//!     while let Some(msg) = output.next().await {
//!         match msg {
//!             BrainstemOutput::Asset(a) => match a {
//!                 AssetEvent::Complete(path) => println!("Model ready at: {}", path),
//!                 AssetEvent::Error(e) => eprintln!("Download error: {}", e),
//!                 _ => {}
//!             },
//!             BrainstemOutput::Event(e) => match e {
//!                 InferenceEvent::Content(c) => print!("{}", c),
//!                 InferenceEvent::Complete => break,
//!                 _ => {}
//!             },
//!             BrainstemOutput::Error(err) => {
//!                 eprintln!("Error: {}", err);
//!                 break;
//!             }
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Hardware Acceleration
//!
//! To enable hardware acceleration, ensure you enable the appropriate feature in `Cargo.toml`:
//!
//! - **Metal**: `features = ["metal"]` (macOS Apple Silicon/Intel)
//! - **CUDA**: `features = ["cuda"]` (NVIDIA GPUs)
//! - **Vulkan**: `features = ["vulkan"]` (Generic/Intel GPUs)

/// The Supplier: Asset management, model registry, and downloads (Facecrab).
pub use facecrab;

/// The Shared Vocabulary: Protocol enums, manifests, and error definitions.
pub use rusty_genius_core as core;

/// The Muscle: Inference engine bindings, KV cache, and logic processing (Cortex).
pub use rusty_genius_cortex as cortex;

/// The Orchestrator: Central event loop, lifecycle management, and strategy (Brainstem).
pub use rusty_genius_stem as brainstem;

// Convenience exports

/// Main entry point for the orchestration event loop.
pub use brainstem::Orchestrator;

/// Top-level error type for the library.
pub use core::error::GeniusError;
