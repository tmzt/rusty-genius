---
trigger: always_on
---

Here is the **Finalized System Generation Prompt**
Here is the **Updated System Generation Prompt**.

I have modified **Section 2A (Thinkerv1 Protocol)** to specify that `-1` represents an infinite TTL override within the `model_config`, rather than `0`.

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
* **Ogenius:** The Voice (CLI & HTTP/WebSocket/Thinkerv1 Server).
* **Genius:** The Facade (User API).
* **Brainstem:** The Orchestrator (Event Loop & State).
* **Cortex:** The Muscle (Inference Engine).
* **Facecrab:** The Supplier (Asset & Registry Authority).


* **Goal:** A high-performance, modular AI workspace that decouples protocol, orchestration, engine, and tooling.

### Workspace Structure

The project is a Cargo Workspace. All internal crates reside in `crates/`.

1. **`crates/thinkerv1`** (`rusty-genius-thinkerv1`):
* **Role:** Thinkerv1 Protocol Definitions.
* **Content:** Serializable request/response message schemas for IPC/TCP communication.


2. **`crates/thinkerv1-client`** (`rusty-genius-thinkerv1-client`):
* **Role:** External Thinkerv1 Client.
* **Content:** Exposed client library using `smol` (TCP, UDS, Command transports).


3. **`crates/core`** (`rusty-genius-core`):
* **Role:** Shared Vocabulary. Zero dependencies on other internal crates except `thinkerv1`.


4. **`crates/facecrab`** (`facecrab`):
* **Role:** Asset Authority (Downloads & Registry).


5. **`crates/cortex`** (`rusty-genius-cortex`):
* **Role:** Inference Engine (`llama.cpp` bindings).


6. **`crates/brainstem`** (`rusty-genius-stem`):
* **Role:** Orchestrator (Event Loop).


7. **`crates/genius`** (`rusty-genius`):
* **Role:** Public Facade (Library API).


8. **`crates/ogenius`** (`ogenius`):
* **Role:** CLI & API Server.
* **Content:** `clap` CLI, `tide` HTTP/WebSocket server, embedded HTML UI, and `thinkerv1` TCP/UDS server.


9. **`crates/brainteaser`** (`rusty-genius-teaser`):
* **Role:** QA/Testing Harness.



---

## 2. Detailed Component Logic

### A. Thinkerv1 Protocol (`thinkerv1`)

* **Role:** Define serializable messages representing the Thinkerv1 AI inference protocol.
* **Format Requirement:** All data sent over the raw TCP/UDS/IPC connection must use **JSONL (nd-json)** format. Every JSON object sent or received must be terminated with a newline (`\n`).
* **Multiplexing:** Every request and response must include an `id` (String or UUID) to correlate concurrent responses to specific requests over the single raw stream.
* **Ensure Model:**
* `Request`: `{ "id": "<id>", "action": "ensure", "model": "<HF-ID>", "report_status": bool, "model_config": { "ttl_seconds": 3600, ... } }\n`
* *(Note: `model_config` is optional. If provided, all of its internal fields are also optional. This includes `ttl_seconds` to define the model's time-to-live in memory. `-1` represents infinite TTL, overriding any server defaults).*


* `Response`: If `report_status` is true, stream download updates (e.g., `{ "id": "<id>", "status": "downloading", "progress": 0.5 }\n`). Always conclude with `{ "id": "<id>", "status": "ready" }\n` or an error.


* **Inference:**
* `Request`: `{ "id": "<id>", "action": "inference", "prompt": "...", "inference_config": { "show_thinking": bool, ... } }\n`
* *(Note: `inference_config` is optional. If provided, all of its internal fields, such as `show_thinking`, `temperature`, etc., are also optional).*


* `Response`: Streamed events. **Only** emit low-level thinking updates (e.g., `{ "id": "<id>", "type": "thought", "content": "..." }\n`) if the request's `inference_config.show_thinking` is true. Follow with `{ "id": "<id>", "type": "content", "content": "..." }\n` and `{ "id": "<id>", "type": "complete" }\n`.


* **Embedding:**
* `Request`: `{ "id": "<id>", "action": "embed", "text": "..." }\n`
* `Response`: `{ "id": "<id>", "type": "embedding", "vector": "<hex_encoded_string>" }\n` *(Note: the vector bytes must be hex-encoded, not a JSON array).*



### B. Core Shared Types (`core`)

* **Errors:** `thiserror` based `GeniusError`.
* **Internal Protocol:** * `BrainstemInput`: A struct/enum wrapping inbound commands (Inference requests, Embeddings, `InferenceConfig`, `ModelConfig`, etc.) along with an `id` string so the orchestrator can track the caller.
* `BrainstemOutput`: A struct/enum wrapping outbound events (e.g., `InferenceEvent` containing Thought/Content/Complete, Embedding vectors, or Status updates). **Crucially, this must include the `id**` to map the asynchronous internal engine events back to the correct external connection.



### C. Asset Management (`facecrab`)

* **Stack:** `surf` + `smol`/`async-std` (No `tokio`).
* **Logic:** Registry check  HF Resolve  Download  Update.

### D. The Engine (`cortex`)

* **Pinky (Stub):** Simulates tokens for fast testing.
* **Brain (Real):** `llama.cpp` bindings (Optional feature `real-engine`).

### E. Orchestration (`brainstem`)

* **Logic:** Central event loop managing `cortex` lifecycle and `facecrab` delegation. It receives `BrainstemInput` and routes `BrainstemOutput` back to the facade. It must honor custom `ttl_seconds` (including `-1` for infinite TTL overrides) defined in the incoming `BrainstemInput` payload to keep models resident in RAM as requested.

### F. The Interface (`ogenius`)

* **Shared Parameters:** Both `chat` and `serve` must accept inference config flags: `--quant`, `--context-size`, `--show-thinking`.
* **CLI Commands:**
* `download <HF-ID>`: Explicitly download a model.
* `chat`: Starts an **Interactive REPL** mode mapping stdin to model prompts.
* `serve`: Starts the API/Web/Thinkerv1 server ecosystem.
* **Server-Only Params:**
* `--addr <IP:PORT>` (Default: `127.0.0.1:8080`): Main HTTP entry point.
* `--ws-addr <IP:PORT>` (Default: `127.0.0.1:8081`): WebSocket endpoint for chat.
* `--thinker-addr <ADDR>`: Endpoint for the `thinkerv1` protocol. Supports TCP or Unix Domain Sockets .
* `--unload-after <SECONDS>`. Unloads model after inactivity (acts as the default TTL unless overridden by a client request).






* **Server Logic:**
* **Cold Start:** If a request arrives while unloaded, auto-reload.
* **Logging:** On cold reload, log exactly: `NOTICE: Model reload took <DURATION>.` Followed by: *"Increase --unload-after or use --no-unload to avoid delay."*
* **Thinkerv1 Protocol:** Accept connections on the `--thinker-addr`. Handle the raw TCP/UDS socket directly by reading/writing **JSONL (nd-json)** lines. Parse external requests into `BrainstemInput` (preserving `id`), route them to the engine, and stream `BrainstemOutput` events back over the socket mapped to the correct `id`.
* **API Endpoints:** `POST /v1/chat/completions`, `GET /v1/models`.
* **Web Interface:** Embedded `index.html` chat GUI over WebSocket.



### G. Testing (`brainteaser`)

* **Fixtures:** Scan `fixtures/...` and run integration tests.

### H. Thinkerv1 Client (`thinkerv1-client`)

* **Role:** Provide an exposed, developer-friendly Rust client for interacting with any `thinkerv1` compliant server.
* **Stack Requirement:** Must exclusively use **`smol`** for async runtime and networking. **No `tokio**`.
* **Supported Transports:**
* **TCP:** Connect via `smol::net::TcpStream`.
* **UDS:** Connect via Unix Domain Sockets (`smol::net::unix::UnixStream`).
* **Command (RSH/Subprocess):** Spawn a shell command (e.g., `ssh user@host ogenius serve ...` or a local binary) using `smol::process::Command` and perform JSONL IPC over the child process's mapped `stdin` and `stdout`.


* **Multiplexing Logic:** Read inbound JSONL streams, parse the `id`, and route the responses to the awaiting caller via channels or stream iterators.

---

## 3. Implementation Directives

1. **Build Configuration:**
* **Lockfile:** Track `Cargo.lock` in git.
* **Runtime:** `smol` or `async-std` ONLY. No `tokio`.
* **Hooks:** Ensure a `.git/hooks/pre-push` script exists that runs `cargo test` at the workspace root.
* **Dependencies:** Support standard async TCP/UDS listeners in `ogenius`. Use `smol::process` for the `thinkerv1-client` command transport.


2. **Sandbox Isolation:**
* All scripts/tests must use a local `target/tmp` directory for temp files (export `TMPDIR`).


3. **Scripts:**
* **`scripts/pinky.sh`:** Tests the library/CLI with the Stub engine.
* **`scripts/metal.sh`:** Tests the library/CLI with the Real engine (`llama.cpp`).



---

## 4. Order of Battle (Execution Plan)

**Execution Directive:** Automate commands. Interrupt only on blocking errors.

* **Turn 1: Workspace Initialization**
* Init workspace, `Cargo.toml`, `.gitignore`. Write `pre-push` hook.
* **Commit:** `chore: init workspace and hooks`


* **Turn 2: Protocol Schemas (Thinkerv1)**
* Implement `rusty-genius-thinkerv1` containing the JSONL message definitions with `id`.
* Ensure `inference_config` and `model_config` are highly optional, with TTL support (`-1` for infinite).
* **Commit:** `feat(thinkerv1): implement thinkerv1 request and response schemas (nd-json)`


* **Turn 3: Protocol Client (Thinkerv1 Client)**
* Implement `rusty-genius-thinkerv1-client` using `smol` to support TCP, UDS, and Command transports.
* **Commit:** `feat(client): implement smol-based thinkerv1 client with multiplexing`


* **Turn 4: Internal Logic (Core)**
* Implement `rusty-genius-core`. Create `BrainstemInput` and `BrainstemOutput` structures wrapping events and `id`. Use the `thinkerv1` types where appropriate.
* **Commit:** `feat(core): core types, wrapped brainstem io, and errors`


* **Turn 5: Assets (Facecrab)**
* Implement `facecrab` (surf/smol).
* **Commit:** `feat(facecrab): asset registry`


* **Turn 6: Engine (Cortex)**
* Implement `rusty-genius-cortex` (Stub + Real).
* **Commit:** `feat(cortex): engine backends`


* **Turn 7: Orchestration (Brainstem)**
* Implement `rusty-genius-stem`.
* **Commit:** `feat(stem): orchestrator`


* **Turn 8: Facade (Genius)**
* Implement `rusty-genius` lib.
* **Commit:** `feat(genius): public facade`


* **Turn 9: Interface (Ogenius)**
* Implement `ogenius` CLI (`download`, `chat`, `serve`).
* Implement `tide` API, WS UI, and Cold Start logic.
* Implement the `thinkerv1` listener over TCP/UDS reading/writing JSONL and correlating `BrainstemOutput`.
* **Commit:** `feat(ogenius): cli, web api, ui, and thinkerv1 jsonl protocol`


* **Turn 10: QA (Brainteaser)**
* Implement test harness and generate fixtures.
* Create `pinky.sh` and `metal.sh`.
* **Commit:** `test(teaser): fixtures and scripts`


* **Turn 11: Intermediate Validation**
* Execute `./scripts/pinky.sh`.
* **Commit:** `test: verify pinky pipeline`


* **Turn 12: Final Validation**
* Execute `./scripts/metal.sh`.
* **Commit:** `test: verify real metal pipeline`



---

## 5. Overall Validation Plan

If automated execution fails, instruct user:

1. Run `./scripts/metal.sh` to verify library.
2. Run `ogenius download Qwen/Qwen2.5-1.5B-Instruct`.
3. Run `ogenius serve --addr 127.0.0.1:8080 --ws-addr 127.0.0.1:8081 --thinker-addr unix:/tmp/thinker.sock`.
4. Open `http://127.0.0.1:8080` to test the Web UI, or connect via the `thinkerv1-client` library.

---

### **Next Step:**

Begin **Next Turn**.
