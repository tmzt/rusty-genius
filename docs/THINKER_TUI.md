# Memory TUI Example — Design Document

A `ratatui`-based terminal UI demonstrating the rusty-genius memory system
and inference orchestrator without any external services (no Redis, no real
model weights).

## Motivation

The project already has:

- **InMemoryMemoryStore** + **MockEmbeddingProvider** (`crates/core/src/memory.rs`) —
  full CRUD memory store with cosine-similarity recall and a deterministic
  hash-based embedder.
- **Orchestrator** (`crates/brainstem/src/lib.rs`) with a **Pinky** stub engine
  (`crates/cortex/src/backend/engine_stub.rs`) — produces echo-style inference
  responses with thought events.
- **basic_chat** example (`crates/genius/examples/basic_chat.rs`) — headless
  stdin/stdout demo of the inference pipeline.

What's missing is an **interactive** demo that ties memory and inference
together in a browsable UI. This TUI fills that gap.

---

## Files to Create / Modify

| File | Action |
|---|---|
| `crates/genius/Cargo.toml` | Add `[dev-dependencies]`: `ratatui = "0.29"`, `crossterm = "0.28"` |
| `crates/genius/examples/memory_tui.rs` | New single-file example (~500 lines) |

No library code changes are required.

---

## TUI Layout

```
+---------------------------------------------------------------+
| [rusty-genius Memory TUI]           Mode: BROWSE | q:Quit     |
+---------------------------+-----------------------------------+
|   Memory Items            |   Detail / Discussion             |
|   (scrollable list)       |                                   |
|   > [Fact] rust-ownership |   Shows selected item detail      |
|     [Entity] pinky-brain  |   OR inference chat output        |
|     [Skill] pattern-match |                                   |
+---------------------------+-----------------------------------+
| [Status / Input Bar]  a:Add | d:Discuss | x:Delete | q:Quit  |
+---------------------------------------------------------------+
```

Three vertical rows:

1. **Title bar** — app name + current mode indicator + quit hint.
2. **Main area** — horizontal split: scrollable item list (left) and
   detail/discussion panel (right).
3. **Status / input bar** — keybinding hints (Browse mode) or text input
   (Add / Discuss modes).

---

## Modes

### Browse

Default mode. Navigate the memory item list with `j`/`k`/`Up`/`Down`.
The right panel shows the selected item's fields.

| Key | Action |
|---|---|
| `j` / `Down` | Select next item |
| `k` / `Up` | Select previous item |
| `a` | Switch to Add mode |
| `d` | Switch to Discuss mode (requires selection) |
| `x` | Delete selected item |
| `q` | Quit |

### AddItem

Multi-step form that walks the user through creating a `MemoryObject`:

1. **ShortName** — type a short identifier, press Enter.
2. **Description** — type a description, press Enter.
3. **Content** — type the content body, press Enter.
4. **SelectType** — choose `MemoryObjectType` from a numbered list
   (`1`=Fact, `2`=Observation, `3`=Skill, `4`=Entity, `5`=Custom), press
   the digit.
5. **Confirm** — press `y` to store or `n`/Esc to cancel.

After confirm the item is stored (with a mock embedding) and the list
refreshes. `Esc` at any sub-step cancels back to Browse.

### Discuss

Chat with Pinky about the selected memory item.

- Right panel becomes a scrollable chat log.
- Bottom bar becomes a text input for the user's prompt.
- The prompt is prepended with context from the selected item and sent to
  the Orchestrator as a `BrainstemCommand::Infer`.
- Streamed `InferenceEvent::Content` tokens are appended to the chat log
  in real-time.
- `Esc` returns to Browse mode.

---

## Architecture

### Runtime

The project uses **async-std** throughout. The TUI main loop is
**synchronous** (required by ratatui/crossterm). Async work is bridged via
`async_std::task::block_on()` for memory operations and a channel bridge
for streaming inference.

### Memory layer

Direct use of `InMemoryMemoryStore` + `MockEmbeddingProvider` from
`rusty_genius_core::memory`. No Redis, no network. Items are stored and
recalled with `block_on()` wrappers around the async `MemoryStore` trait
methods.

### Inference layer

`Orchestrator::with_engine(Box::new(Pinky::new()))` — avoids the default
`Orchestrator::new()` which requires the `cortex-engine` feature's
`AssetAuthority`. The Pinky stub engine echoes prompts with "Pinky says: …"
after emitting thought events.

### Event bridge (inference → TUI)

```
Orchestrator (async-std task)
    ↓  futures::mpsc::Sender<BrainstemOutput>
bridge task (async-std::task::spawn)
    ↓  reads futures::mpsc, writes std::sync::mpsc
std::sync::mpsc::Receiver
    ↓  try_recv() in synchronous main loop
TUI render
```

This avoids blocking the main loop while still receiving streamed tokens.

### Main loop

```rust
loop {
    // 1. Poll crossterm events (50ms timeout)
    if crossterm::event::poll(Duration::from_millis(50))? {
        if let Event::Key(key) = crossterm::event::read()? {
            handle_key_event(&mut app, key);
        }
    }
    // 2. Drain inference channel
    while let Ok(msg) = app.inference_rx.try_recv() {
        handle_inference_output(&mut app, msg);
    }
    // 3. Render
    terminal.draw(|f| ui(f, &mut app))?;
    // 4. Check quit flag
    if app.should_quit { break; }
}
```

---

## Key Types

### `App` struct

```rust
struct App {
    mode: Mode,                                     // Browse | AddItem(_) | Discuss
    should_quit: bool,

    // Memory
    store: Arc<InMemoryMemoryStore>,
    embedder: MockEmbeddingProvider,
    items: Vec<MemoryObject>,                       // cached list_all() snapshot
    list_state: ListState,                          // ratatui selection state

    // Add-item form
    add_step: AddStep,                              // ShortName | Description | Content | SelectType | Confirm
    add_input: String,
    add_draft: PartialMemoryObject,                 // fields accumulated so far

    // Discuss / inference
    discussion: Vec<ChatMessage>,                   // role + content pairs
    discuss_input: String,
    inference_tx: mpsc::Sender<BrainstemInput>,      // futures mpsc → orchestrator
    inference_rx: std::sync::mpsc::Receiver<BrainstemOutput>, // sync channel ← bridge
}
```

### Seed data

Three items are inserted at startup so the list is never empty:

| short_name | type | content |
|---|---|---|
| `rust-ownership` | `Fact` | "Rust uses ownership + borrowing to guarantee memory safety without GC." |
| `pinky-brain` | `Entity` | "Pinky is the stub inference engine in rusty-genius." |
| `pattern-match` | `Skill` | "Use Rust match expressions for exhaustive pattern matching." |

---

## Dependencies

Added as **dev-dependencies** only (not required by the library):

```toml
[dev-dependencies]
ratatui = "0.29"
crossterm = "0.28"
```

The example also uses crates already in the dependency tree: `async-std`,
`futures`, `anyhow`, `rusty_genius_core`, `rusty_genius_stem`.

---

## Running

```bash
# Default features are sufficient (cortex-engine includes the Pinky stub)
cargo run -p rusty-genius --example memory_tui
```

### Verification checklist

- [ ] Browse pre-seeded items with arrow keys / j / k
- [ ] Press `a`, fill all fields, confirm — new item appears in list
- [ ] Select an item, press `d`, type a question, see Pinky's streamed response
- [ ] Press `x` to delete the selected item
- [ ] Press `q` to quit cleanly (terminal restored)
- [ ] Force a panic (e.g. resize race) — terminal still restored via panic hook

---

## Key API surface used

### `rusty_genius_core::memory`

| Type / Trait | Used for |
|---|---|
| `InMemoryMemoryStore` | `.store()`, `.list_all()`, `.forget()`, `.get()` |
| `MockEmbeddingProvider` | `.embed_sync(text) → Vec<f32>` |
| `MemoryObject` | 12-field struct (id, short_name, long_name, description, object_type, content, embedding, metadata, created_at, updated_at, ttl) |
| `MemoryObjectType` | Enum: `Fact`, `Observation`, `Preference`, `Skill`, `Entity`, `Relationship`, `Custom(String)` |
| `MemoryStore` trait | async CRUD: `store`, `list_all`, `forget`, `get` |
| `EmbeddingProvider` trait | `async fn embed(&self, text) → Result<Vec<f32>>` |

### `rusty_genius_core::protocol`

| Type | Used for |
|---|---|
| `BrainstemInput` | `{ id, command }` — sent to orchestrator |
| `BrainstemCommand::LoadModel(name)` | Load the Pinky stub model |
| `BrainstemCommand::Infer { model, prompt, config }` | Run inference |
| `BrainstemOutput` | `{ id, body }` — received from orchestrator |
| `BrainstemBody::Event(InferenceEvent)` | Token stream |
| `InferenceEvent::Content(String)` | Streamed text token |
| `InferenceEvent::Complete` | End of generation |
| `InferenceEvent::ProcessStart` | Inference started |
| `InferenceEvent::Thought(ThoughtEvent)` | Pinky's "Narf!" thoughts |

### `rusty_genius_stem::Orchestrator`

| Method | Signature |
|---|---|
| `with_engine(engine)` | `fn with_engine(Box<dyn Engine>) → Self` — bypasses AssetAuthority |
| `run(rx, tx)` | `async fn run(&mut self, Receiver<BrainstemInput>, Sender<BrainstemOutput>) → Result<()>` |

### `rusty_genius_cortex::backend::engine_stub::Pinky`

| Method | Notes |
|---|---|
| `Pinky::new()` | Available when `cortex-engine` feature is active and `real-engine` is not |
| `Engine::infer()` | Echoes prompt as "Pinky says: {prompt}" with thought events |

---

## Future extensions

- **Search mode** — recall by query + embedding similarity instead of
  just listing all items.
- **Persistence** — swap `InMemoryMemoryStore` for Redis-backed store
  when `redis-context` feature is active.
- **Real model** — enable `real-engine` + `metal`/`cuda` features to use
  an actual GGUF model via the cortex engine.
- **Memory consolidation** — demonstrate the PFC → Neocortex `Ship`
  command for long-term memory promotion.
