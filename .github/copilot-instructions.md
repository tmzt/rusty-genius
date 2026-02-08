# Copilot Instructions for rusty-genius

## Project Overview

**rusty-genius** is a high-performance, modular, local-first AI orchestration library written in Rust. It follows a biological metaphor where components act like parts of a "nervous system" for AI.

The architecture decouples protocol, orchestration, engine, and tooling to enable on-device AI with absolute privacy, zero latency, and offline reliability.

## Build, Test, and Lint Commands

### Building

```bash
# Build all workspace members
cargo build

# Build with Metal acceleration (macOS)
cargo build --features metal

# Build with CUDA acceleration
cargo build --features cuda

# Build ogenius CLI with Metal
cargo build --release -p ogenius --features metal

# Build specific crate
cargo build -p rusty-genius-core
```

### Testing

```bash
# Run all tests (uses stub engine by default)
cargo test

# Test with real engine
cargo test --features real-engine

# Test specific crate
cargo test -p rusty-genius-core

# Run integration tests for ogenius (requires Metal build)
make ogenius_tests

# Run specific test file
cargo test -p ogenius --test http_tests -- --test-threads=1
```

### Linting and Formatting

```bash
# Format all code
cargo fmt

# Check formatting without changes
cargo fmt -- --check

# Run clippy (linter)
cargo clippy

# Run clippy on all targets
cargo clippy --all-targets --all-features
```

### Running Examples

```bash
# Test asset downloader (facecrab)
cargo run -p facecrab --example downloader

# Test chat with Metal
cargo run -p rusty-genius --example basic_chat --features metal

# Test chat with CUDA
cargo run -p rusty-genius --example basic_chat --features cuda

# Test chat with CPU only
cargo run -p rusty-genius --example basic_chat --features real-engine
```

### Running ogenius CLI

```bash
# Build and run the server (via Makefile)
make ogenius_metal
make run_ogenius

# Or directly
cargo run -p ogenius --features metal -- serve --addr 127.0.0.1:9099
```

## Architecture

### Crate Organization (Biological Metaphor)

The workspace follows a **bottom-up dependency hierarchy**. Lower layers have zero knowledge of higher layers:

```
Core (Shared Vocabulary)
  ↑
  ├── Facecrab (Asset Authority)
  ├── Cortex (Inference Engine)
  └── Brainstem (Orchestrator)
      ↑
      └── Genius (Public Facade)
          ↑
          └── ogenius (CLI Application)
```

**Key Crates:**

- **`core`** (`rusty-genius-core`): Protocol definitions, error types, manifest structures. Zero internal dependencies. All other crates depend on this.

- **`facecrab`**: Autonomous asset authority for model resolution, registry management, and HuggingFace downloads. Usable as standalone crate.

- **`cortex`** (`rusty-genius-cortex`): Inference engine abstraction. Uses feature flags to switch between:
  - **Stub engine** (`Pinky`): Default, no-op implementation for testing
  - **Real engine** (`Brain`): Actual llama.cpp bindings via `llama-cpp-2` crate
  
- **`brainstem`** (`rusty-genius-stem`): The orchestrator. Runs the central event loop, manages engine lifecycle with TTL (5 min default), coordinates between facecrab and cortex.

- **`genius`** (`rusty-genius`): Public facade that re-exports internal crates. This is what library consumers import.

- **`ogenius`**: CLI application providing interactive chat REPL and OpenAI-compatible API server.

- **`brainteaser`** (`rusty-genius-teaser`): QA harness for integration testing via filesystem fixtures.

### Key Protocol Flow

Communication happens via message passing with `mpsc` channels:

1. **User → Brainstem**: Send `BrainstemInput` containing `BrainstemCommand` (LoadModel, Infer, Embed, etc.)
2. **Brainstem → Facecrab**: Request model path via `AssetAuthority::ensure_model()`
3. **Brainstem → Cortex**: Load model and execute inference via `Engine` trait
4. **Cortex → Brainstem → User**: Stream `BrainstemOutput` containing `BrainstemBody` (Event, Asset, ModelList, Error)

### Engine Lifecycle & TTL

The `Orchestrator` uses a `CortexStrategy` to manage memory:
- **Immediate**: Unload after every inference
- **HibernateAfter(Duration)**: Default 5 minutes, unload after inactivity
- **KeepAlive**: Never unload

State transitions: `Unloaded → Loading → Loaded → Inferring → Loaded → (timeout) → Unloaded`

## Key Conventions

### Feature Flags for Hardware Acceleration

The `cortex` crate uses conditional compilation:
- **No features**: Stub engine only (for testing without llama.cpp)
- **`real-engine`**: Enables llama.cpp bindings
- **`metal`**: Implies `real-engine` + Metal GPU support (macOS)
- **`cuda`**: Implies `real-engine` + CUDA support
- **`vulkan`**: Implies `real-engine` + Vulkan support

When adding features to higher-level crates (`genius`, `ogenius`), pass them through to `cortex`.

### Protocol Message Structure

All protocol messages use strongly-typed enums defined in `core::protocol`:

- `BrainstemInput { id: Option<String>, command: BrainstemCommand }` - Requests to orchestrator
- `BrainstemOutput { id: Option<String>, body: BrainstemBody }` - Responses from orchestrator
- `BrainstemCommand` - LoadModel, Infer, Embed, ListModels, Reset, Stop
- `BrainstemBody` - Event(InferenceEvent), Asset(AssetEvent), ModelList, Error
- `InferenceEvent` - ProcessStart, Thought, Content, Embedding, Complete

The `id` field is optional for request/response correlation.

### Async Runtime

**Always use `async-std`**, not tokio. This is a project-wide convention:
- Use `#[async_std::main]` for binaries
- Use `async_std::task::spawn` for spawning tasks
- Use `async_std::process::Command` for process management

### Model Configuration

Models are managed via two files:
- **`manifest.toml`** (in `$GENIUS_HOME`): Static, user-editable model definitions
- **`registry.toml`** (in `$GENIUS_CACHE`): Dynamic, system-maintained download registry

When adding model support, update the built-in manifest in `core::manifest` or document how users can extend via their local `manifest.toml`.

### Error Handling

- Use `anyhow::Result` for application code
- Use `thiserror` for library error types (see `core::error::GeniusError`)
- Propagate errors through the protocol as `BrainstemBody::Error(String)`

### Testing Patterns

- Unit tests use the stub engine (no features needed)
- Integration tests requiring inference should use `#[cfg(feature = "real-engine")]`
- The `brainteaser` crate provides fixture-based testing for complex scenarios
- HTTP tests in `ogenius` require a pre-built binary (see `Makefile` for `ogenius_tests`)

### Workspace Version Management

All crates share workspace-level metadata in root `Cargo.toml`:
- version = "0.1.3"
- edition = "2021"
- authors, license, repository, keywords, categories

Use `.workspace = true` in crate `Cargo.toml` files to inherit these values.

## Common Patterns

### Creating an Orchestrator Instance

```rust
use rusty_genius::Orchestrator;
let mut genius = Orchestrator::new().await?;
```

### Loading a Model

Models are loaded by name (resolved via manifest/registry):
```rust
input.send(BrainstemInput {
    id: None,
    command: BrainstemCommand::LoadModel("tiny-model".into())
}).await?;
```

### Running Inference

```rust
input.send(BrainstemInput {
    id: Some("req-1".into()),
    command: BrainstemCommand::Infer {
        model: None,  // Uses last loaded model
        prompt: "Your prompt here".into(),
        config: Default::default(),
    }
}).await?;
```

### Handling Streamed Output

Process messages from output channel:
```rust
while let Some(msg) = output.next().await {
    match msg.body {
        BrainstemBody::Event(InferenceEvent::Content(text)) => print!("{}", text),
        BrainstemBody::Event(InferenceEvent::Complete) => break,
        BrainstemBody::Asset(AssetEvent::Progress(current, total)) => { /* ... */ },
        BrainstemBody::Error(err) => eprintln!("Error: {}", err),
        _ => {}
    }
}
```

### Creating Engine Implementations

When implementing `Engine` trait in cortex:
- Stub implementation returns mock data immediately
- Real implementation delegates to llama-cpp-2 bindings
- Use `#[cfg(feature = "real-engine")]` to guard real engine code

## Environment Variables

- `GENIUS_HOME`: Config directory (default: `~/.config/rusty-genius`)
- `GENIUS_CACHE`: Model cache directory (default: `$GENIUS_HOME/cache`)
- `TEST_BINARY`: Path to ogenius binary for integration tests
- `TMPDIR`: Temporary directory (Makefile sets to `./target/tmp`)

## Dependencies

### Build Prerequisites

- Rust 2021 edition or later
- CMake (required for llama.cpp when using real-engine features)
- OS-specific:
  - **macOS**: Xcode Command Line Tools, optionally Homebrew for CMake
  - **Linux**: build-essential, cmake, libclang-dev
  - **Windows**: Visual Studio 2022 with C++ workload, LIBCLANG_PATH env var

### Key External Crates

- **async-std**: Async runtime (project standard)
- **futures**: Stream/channel abstractions
- **llama-cpp-2**: Rust bindings to llama.cpp (optional, behind `real-engine` feature)
- **serde/serde_json**: Serialization
- **thiserror/anyhow**: Error handling
- **clap**: CLI argument parsing (ogenius)
- **tide/tide-websockets**: HTTP server (ogenius)
- **surf**: HTTP client (facecrab)

## Publishing

All crates except `brainteaser` are published to crates.io. The workspace uses:
- `publish.workspace = true` (most crates)
- Version is synchronized across all published crates at 0.1.3

`Cargo.lock` is tracked in this repository for reproducible builds during development, but is ignored when publishing to crates.io.
