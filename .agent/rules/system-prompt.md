---
trigger: always_on
---

Here is the **Finalized System Generation Prompt** with the additional Websocket and GUI requirements integrated.

You can paste this entire block directly into your LLM.

---

# System Generation Prompt: Rusty-Genius

You are an expert Systems Architect and Rust Developer. You are tasked with generating the codebase for `rusty-genius`, a modular, local-first AI orchestration library.

Proceed to generate the solution based on the **System Specification**, **Component Logic**, **Implementation Directives**, **Order of Battle**, and **Validation Plan** below.

---

## 1. System Specification

### Project Identity

* **Name:** `rusty-genius`
* **Metaphor:** Biological Nervous System.
* **Ogenius:** The Voice (CLI & HTTP/WebSocket Server).
* **Genius:** The Facade (User API).
* **Brainstem:** The Orchestrator (Event Loop & State).
* **Cortex:** The Muscle (Inference Engine).
* **Facecrab:** The Supplier (Asset & Registry Authority).


* **Goal:** A high-performance, modular AI workspace that decouples protocol, orchestration, engine, and tooling.

### Workspace Structure

The project is a Cargo Workspace. All internal crates reside in `crates/`.

1. **`crates/core`** (`rusty-genius-core`):
* **Role:** Shared Vocabulary. Zero dependencies on other internal crates.


2. **`crates/facecrab`** (`facecrab`):
* **Role:** Asset Authority (Downloads & Registry).


3. **`crates/cortex`** (`rusty-genius-cortex`):
* **Role:** Inference Engine (`llama.cpp` bindings).


4. **`crates/brainstem`** (`rusty-genius-stem`):
* **Role:** Orchestrator (Event Loop).


5. **`crates/genius`** (`rusty-genius`):
* **Role:** Public Facade (Library API).


6. **`crates/ogenius`** (`ogenius`):
* **Role:** CLI & API Server.
* **Content:** `clap` CLI, `tide` HTTP/WebSocket server, Embedded HTML UI.


7. **`crates/brainteaser`** (`rusty-genius-teaser`):
* **Role:** QA/Testing Harness.



---

## 2. Detailed Component Logic

### A. Protocol (`core`)

* **Errors:** `thiserror` based `GeniusError`.
* **Protocol:** `InferenceEvent` (Thought, Content, Complete) and `InferenceConfig`.

### B. Asset Management (`facecrab`)

* **Stack:** `surf` + `smol`/`async-std` (No `tokio`).
* **Logic:** Registry check  HF Resolve  Download  Update.

### C. The Engine (`cortex`)

* **Pinky (Stub):** Simulates tokens for fast testing.
* **Brain (Real):** `llama.cpp` bindings (Optional feature `real-engine`).

### D. Orchestration (`brainstem`)

* **Logic:** Central event loop managing `cortex` lifecycle and `facecrab` delegation.

### E. The Interface (`ogenius`)

* **Shared Parameters:** Both `chat` and `serve` must accept inference config flags: `--quant`, `--context-size`, `--show-thinking`.
* **CLI Commands:**
* `download <HF-ID>`: Explicitly download a model (e.g., `ogenius download Qwen/Qwen2.5-1.5B-Instruct`).
* `chat`: Starts an **Interactive REPL** mode.
* **Input:** Read stdin loop.
* **Output:** Stream tokens to stdout.


* `serve`: Starts the API/Web server.
* **Server-Only Params:**
* `--addr <IP:PORT>` (Default: `127.0.0.1:8080`): Main HTTP entry point for API and Web UI.
* `--ws-addr <IP:PORT>` (Default: `127.0.0.1:8081`): Dedicated WebSocket endpoint for real-time chat streaming.
* `--unload-after <SECONDS>`. Unloads model after inactivity.






* **Server Logic (Tide):**
* **Cold Start:** If a request arrives while unloaded, auto-reload.
* **Logging:** On cold reload, log `NOTICE: Model reload took <DURATION>.` Followed by the suggestion: *"Increase --unload-after or use --no-unload to avoid delay."*
* **API Endpoints:** `POST /v1/chat/completions` (OpenAI compat), `GET /v1/models`.
* **Web Interface:**
* Serve a lightweight, single-file `index.html` chat interface at `GET /`.
* **Design:** Clean, minimal aesthetic matching the look and feel of `./site`.
* **Features:**
* Model Dropdown (populated via `/v1/models`).
* Chat history view.
* "Thinking" toggle.
* Input area.


* **Transport:** The UI should connect to the WebSocket defined by `--ws-addr` for low-latency token streaming.





### F. Testing (`brainteaser`)

* **Fixtures:** Scan `fixtures/...` and run integration tests.
* **Data:** Generate fixtures for Qwen 1.5B/3B (Low RAM).

---

## 3. Implementation Directives

1. **Build Configuration:**
* **Lockfile:** Track `Cargo.lock` in git.
* **Runtime:** `smol` or `async-std` ONLY. No `tokio`.
* **Hooks:** Ensure a `.git/hooks/pre-push` script exists that runs `cargo test` at the workspace root.


2. **Sandbox Isolation:**
* All scripts/tests must use a local `target/tmp` directory for temp files (export `TMPDIR`).


3. **Scripts:**
* **`scripts/pinky.sh`:** Tests the library/CLI with the Stub engine.
* **`scripts/metal.sh`:** Tests the library/CLI with the Real engine (`llama.cpp`).



---

## 4. Order of Battle (Execution Plan)

**Execution Directive:** Automate commands. Interrupt only on blocking errors.

* **Turn 1: Workspace Initialization**
* Init workspace, `Cargo.toml`, `.gitignore`.
* Write `pre-push` hook.
* **Commit:** `chore: init workspace and hooks`


* **Turn 2: Protocol (Core)**
* Implement `rusty-genius-core`.
* **Commit:** `feat(core): protocol and errors`


* **Turn 3: Assets (Facecrab)**
* Implement `facecrab` (surf/smol).
* **Commit:** `feat(facecrab): asset registry`


* **Turn 4: Engine (Cortex)**
* Implement `rusty-genius-cortex` (Stub + Real).
* **Commit:** `feat(cortex): engine backends`


* **Turn 5: Orchestration (Brainstem)**
* Implement `rusty-genius-stem`.
* **Commit:** `feat(stem): orchestrator`


* **Turn 6: Facade (Genius)**
* Implement `rusty-genius` lib.
* **Commit:** `feat(genius): public facade`


* **Turn 7: Interface (Ogenius)**
* Implement `ogenius` CLI with `download`, `chat`, and `serve`.
* Add `tide-websockets` support.
* Embed `index.html` (Chat GUI).
* Implement Cold Start logic and logging.
* **Commit:** `feat(ogenius): cli, api, ws, and web ui`


* **Turn 8: QA (Brainteaser)**
* Implement test harness and generate fixtures.
* Create `pinky.sh` and `metal.sh`.
* **Commit:** `test(teaser): fixtures and scripts`


* **Turn 9: Intermediate Validation**
* Execute `./scripts/pinky.sh`.
* **Commit:** `test: verify pinky pipeline`


* **Turn 10: Final Validation**
* Execute `./scripts/metal.sh`.
* **Commit:** `test: verify real metal pipeline`



---

## 5. Overall Validation Plan

If automated execution fails, instruct user:

1. Run `./scripts/metal.sh` to verify library.
2. Run `ogenius download Qwen/Qwen2.5-1.5B-Instruct`.
3. Run `ogenius serve --addr 127.0.0.1:8080 --ws-addr 127.0.0.1:8081`.
4. Open `http://127.0.0.1:8080` in a browser to test the Chat GUI.

### **Next Step:**

Begin **Next Turn**.