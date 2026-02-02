---
trigger: always_on
---

Here is the comprehensive **System Generation Prompt**. You can paste this entire block into an LLM (like Claude 3.5 Sonnet, GPT-4o, or O3) to generate the complete `rusty-genius` codebase.

---

# System Generation Prompt: Rusty-Genius

You are an expert Systems Architect and Rust Developer. You are tasked with generating the codebase for `rusty-genius`, a modular, local-first AI orchestration library.

Proceed to generate the solution based on the **System Specification**, **Component Logic**, and **Implementation Directives** below.

---

## 1. System Specification

### Project Identity

* **Name:** `rusty-genius`
* **Metaphor:** Biological Nervous System.
* **Genius:** The Facade (User API).
* **Brainstem:** The Orchestrator (Event Loop & State).
* **Cortex:** The Muscle (Inference Engine).
* **Facecrab:** The Supplier (Asset & Registry Authority).


* **Goal:** A high-performance, modular AI workspace that decouples protocol, orchestration, engine, and tooling.

### Workspace Structure

The project is a Cargo Workspace. All internal crates reside in `crates/` and use specific namespacing to avoid crates.io collisions.

1. **`crates/core`** (`rusty-genius-core`):
* **Role:** Shared Vocabulary. Zero dependencies on other internal crates.
* **Content:** Protocol Enums, Manifest Structs, Inference Configuration, **Error Definitions**.


2. **`crates/facecrab`** (`facecrab`):
* **Role:** Asset Authority.
* **Content:** Asset Downloader, Local Registry (`~/.config/...`), HF API resolution, Embedded Manifests.


3. **`crates/cortex`** (`rusty-genius-brain-cortex`):
* **Role:** Inference Engine.
* **Content:** `llama.cpp` bindings (**Optional**), KV Cache, Token Streaming.


4. **`crates/brainstem`** (`rusty-genius-brain-stem`):
* **Role:** Orchestrator.
* **Content:** Central Event Loop, Asset delegation, Engine Lifecycle (TTL).


5. **`crates/genius`** (`rusty-genius`):
* **Role:** Public Facade.
* **Content:** Re-exports internal crates.


6. **`crates/brainteaser`** (`rusty-genius-brain-teaser`):
* **Role:** QA/Testing.
* **Content:** Integration harness using file-system fixtures.



---

## 2. Detailed Component Logic

### A. Protocol (`core`)

* **Error Handling:**
* Use **`thiserror`** to define a structured, public `GeniusError` enum in `lib.rs`.
* This enum should cover protocol violations, manifest parsing errors, and asset failures.


* **Manifests:** Define `UserManifest` (raw) and `ModelSpec` (resolved). Implement logic to merge partial entries with global defaults.
* **Inference Config:** `InferenceConfig` (Behavior vs Sampling).
* **Output Protocol:** `InferenceEvent` (ProcessStart, Thought, Content, Complete).
* `Thought(ThoughtEvent)`: Nested enum (`Start`, `Delta`, `Stop`) for reasoning.


* **Brainstem Protocol:** `BrainstemInput` (Commands) and `BrainstemOutput` (Events).

### B. Asset Management (`facecrab`)

* **Error Handling:** Use **`anyhow`** internally for file I/O and network operations, mapping them to `GeniusError` at the public boundary.
* **Registry:** Manage `registry.toml`.
* **Delegation:** `ensure_model` checks Registry  Resolves HF Filename  Downloads  Updates Registry.
* **Defaults:** Embed `models.toml` via `include_str!`.

### C. The Engine (`cortex`) & Stubbing

* **Feature Flag:** Define `real-engine`.
* **Dependency:** `llama-cpp-2` must be **optional**, active only via `real-engine`.
* **Backend Abstraction:**
* **Pinky (Stub):** If `real-engine` is OFF, compile a stub that simulates delay, ignores weights, and emits dummy "Narf!" tokens/thoughts.
* **The Brain (Real):** If `real-engine` is ON, compile `llama.cpp` bindings. Implement an Output State Machine to parse `<think>` tags into `ThoughtEvent`s.



### D. Orchestration (`brainstem`)

* **Logic:** Receives `BrainstemInput`, delegates assets to `facecrab`, manages `cortex` lifecycle.
* **Lifecycle:** Implement `CortexStrategy` (Immediate, HibernateAfter, KeepAlive).

### E. Testing (`brainteaser`)

* **Fixtures:** Scan `fixtures/{ORG}/{REPO}/{QUANT}/{TEST}.md`.
* **Harness:** Inject `ModelSpec` from path, trigger download, run inference.

---

## 3. Implementation Directives

1. **Build Configuration:**
* `cortex/Cargo.toml`: `llama-cpp-2` is `optional = true`.
* `genius/Cargo.toml`: Expose `metal` and `cuda` features, forwarding them to `cortex`.
* Default workspace behavior: **Stubbed (Pinky)**.


2. **Error Strategy:**
* **Inside Crates (`facecrab`, `cortex`, `stem`):** Use `anyhow::Result` for implementation flexibility.
* **In Core (`rusty-genius-core`):** Define explicit error types using `thiserror`.
* **At Boundaries:** When sending errors over the `BrainstemOutput` channel (which must be serializable), convert `anyhow` errors to a `String` or a structured `GeniusError` variant.


3. **Scripts (`scripts/metal.sh`):**
* Generate a helper script for running heavy tests.
* It must verify `cmake` exists (checking Homebrew paths `/opt/homebrew/bin` etc).
* It must run `cargo test` on `brainteaser` with `--features "rusty-genius-brain-cortex/real-engine rusty-genius-brain-cortex/metal"`.


4. **Generative Order:**
* Begin with `Core` (Protocol & Errors).
* Build `Facecrab` (Assets).
* Build `Cortex` (Stub/Real Engine).
* Build `Brainstem` (Orchestrator).
* Finish with `Genius` (Facade) and `Brainteaser`.