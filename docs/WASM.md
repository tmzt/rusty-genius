# WASM Support — Hippocampus

Browser-compatible persistent memory for rusty-genius using IndexedDB + SQLite FTS5.

## Overview

The **hippocampus** crate (`rusty-genius-hippocampus`) provides a `MemoryStore` implementation that runs entirely in the browser via `wasm32-unknown-unknown`. It combines:

- **IndexedDB** (via `rexie`) for content and embedding storage
- **SQLite FTS5** (via `sqlite-wasm-rs` with `RelaxedIdbVFS`) for full-text search indexing

The SQLite database itself is persisted to IndexedDB automatically by the `RelaxedIdbVFS`, so no manual serialization or flushing is needed.

## Architecture

```
IndexedDB "rusty-genius-memory" (via rexie)
├── "memory_objects" store   → MemoryObject JSON (sans embedding)
└── "memory_embeddings" store → { id, embedding: Vec<f32> }

IndexedDB "hippocampus.db" (via RelaxedIdbVFS — automatic)
└── SQLite pages (managed by sqlite-wasm-rs)
    ├── memory_objects table
    ├── memory_fts FTS5 virtual table
    └── Triggers (insert/update/delete sync)
```

### Recall Flow

1. **FTS5 text search** → ranked IDs by relevance
2. **Vector search** → load all embeddings from IDB, cosine similarity in Rust → ranked IDs
3. **Merge** (vector results first, then FTS), deduplicate, apply type filter
4. **Load** full objects from IDB for the result set

## Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Add WASM target
rustup target add wasm32-unknown-unknown
```

## Building

```bash
cd crates/hippocampus
wasm-pack build --target web
```

The output will be in `crates/hippocampus/pkg/`.

## Usage from JavaScript

```javascript
import init, { /* wasm-bindgen exports */ } from './pkg/rusty_genius_hippocampus.js';

await init();
// Use HippocampusWorker or IdbMemoryStore via wasm-bindgen bindings
```

## IdbMemoryStore API

The `IdbMemoryStore` implements the full `MemoryStore` trait:

| Method | Description |
|--------|-------------|
| `store(object)` | Store a MemoryObject (content in IDB, FTS in SQLite) |
| `recall(query, embedding, limit, type)` | Hybrid FTS5 + vector search |
| `recall_by_vector(embedding, limit, type)` | Vector-only cosine similarity search |
| `get(id)` | Load a single object by ID |
| `forget(id)` | Delete an object from both stores |
| `list_by_type(type)` | List all objects of a given type |
| `list_all()` | List all stored objects |
| `flush_all()` | Clear all data |

## HippocampusWorker

The `HippocampusWorker` provides the same channel-based dispatch loop as `NeocortexWorker`, accepting `MemoryInput` and producing `MemoryOutput` messages. It auto-embeds content on Store/Recall if no embedding is provided.

## Limitations

- **Single-tab**: IndexedDB is per-origin, not per-tab. Multiple tabs writing simultaneously may conflict.
- **Relaxed durability**: `RelaxedIdbVFS` trades strict durability for performance. Data may be lost on unexpected tab close.
- **No multi-connection SQLite**: Only one `FtsIndex` instance should be open at a time.
- **Vector search is O(n)**: All embeddings are loaded and compared. For large stores (>10k objects), consider limiting vector search scope.
- **No GPU acceleration**: Cosine similarity runs in pure Rust on the CPU.

## Testing

```bash
# Native tests (cosine, schema validation)
cargo test -p rusty-genius-hippocampus

# Browser integration tests
cd crates/hippocampus
wasm-pack test --headless --chrome
```

## Browser Compatibility

| Browser | IndexedDB | sqlite-wasm-rs | Status |
|---------|-----------|----------------|--------|
| Chrome 80+ | Yes | Yes | Supported |
| Firefox 78+ | Yes | Yes | Supported |
| Safari 14+ | Yes | Yes | Supported |
| Edge 80+ | Yes | Yes | Supported |
| Node.js | No | No | Not supported (use neocortex) |

## Related Crates

- `rusty-genius-neocortex` — Native SQLite+sqlx long-term memory
- `rusty-genius-pfc` — Redis-backed working memory
- `rusty-genius-core` — `MemoryStore` trait definition
