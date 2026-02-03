---
trigger: always_on
---

Here is the **Finalized System Generation Prompt**.

I have updated **Section 3 (Implementation Directives)** to explicitly allow `async-std` or `smol` while strictly prohibiting `tokio`.

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


3. **`crates/cortex`** (`rusty-genius--cortex`):
* **Role:** Inference Engine.
* **Content:** `llama.cpp` bindings (**Optional**), KV Cache, Token Streaming.


4. **`crates/brainstem`** (`rusty-genius--stem`):
* **Role:** Orchestrator.
* **Content:** Central Event Loop, Asset delegation, Engine Lifecycle (TTL).


5. **`crates/genius`** (`rusty-genius`):
* **Role:** Public Facade.
* **Content:** Re-exports internal crates.


6. **`crates/brainteaser`** (`rusty-genius--teaser`):
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

* **Network Stack:**
* **Strict Requirement:** You must use **`surf`** for all HTTP requests.
* **Runtime:** Configure `surf` to use `smol` or `async-std`.
* **Forbidden:** Do **not** use `reqwest` or `tokio` for the downloader.
* **Streams:** Use **`futures`** for handling download streams.


* **Error Handling:** Use **`anyhow`** internally, mapping to `GeniusError`.
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
* **Embedded Test Data:**
* You must generate physical fixture files in `crates/brainteaser/fixtures/` targeting **low-RAM models (<4GB)**.
* **Target Model 1:** `Qwen/Qwen2.5-1.5B-Instruct` (Quant: `Q4_K_M`) -> Very fast, ~1GB RAM.
* **Target Model 2:** `Qwen/Qwen2.5-3B-Instruct` (Quant: `Q4_K_M`) -> Robust, ~2.5GB RAM.
* **Fixture Content:** The `.md` files should contain simple prompts (e.g., "What is the capital of France?" or "Write a hello world in Rust") to verify the pipeline.



---

## 3. Implementation Directives

1. **Build Configuration:**
* **Git Ignore Strategy:** Ensure `Cargo.lock` is **NOT** ignored (it must be tracked).
* **Library Warning:** Add a comment in `.gitignore` or `README.md` stating: *"NOTE: Cargo.lock is tracked for development stability. Restore to .gitignore before publishing to crates.io."*
* `cortex/Cargo.toml`: `llama-cpp-2` is `optional = true`.
* `genius/Cargo.toml`: Expose `metal` and `cuda` features, forwarding them to `cortex`.
* Default workspace behavior: **Stubbed (Pinky)**.


2. **Dependencies & Runtime:**
* **Async Runtime:** You may use either **`smol`** or **`async-std`** for internal crate logic.
* **Strict Prohibition:** Do **not** use `tokio` as the runtime.
* **HTTP Client:** `facecrab` must use `surf`. Avoid `reqwest`.


3. **Sandbox Isolation (Temp Files):**
* **Requirement:** The system must not rely on the OS-default `/tmp` directory.
* **Implementation:** Configure **all scripts** to create a local temporary directory (e.g., `target/tmp`) and force the process to use it by exporting `TMPDIR`, `TEMP`, and `TMP`.


4. **Scripts:**
* **`scripts/pinky.sh` (Fast Test):**
* Runs tests using the **Stubbed Backend**.
* Command: `cargo test -p rusty-genius--teaser --no-default-features -- --nocapture`


* **`scripts/metal.sh` (Real Test):**
* Runs tests using the **Real Llama.cpp Backend**.
* Must verify `cmake` exists.
* Command: `cargo test -p rusty-genius--teaser --features "rusty-genius--cortex/real-engine rusty-genius--cortex/metal" -- --nocapture`





---

## 4. Order of Battle (Execution Plan)

You must execute the generation in the following strict turns. **At the end of each turn, you must stop and provide a short, concise git commit message.**

**Execution Directive:**

1. **Automate Process:** The Agent (you) must execute the necessary shell commands (creation, compilation, and **testing**) within the turn. Do not ask the user to run commands unless your environment lacks execution permissions.
2. **Seamless Flow:** Continue working seamlessly from turn to turn.
3. **Interrupt Policy:** **Only interrupt the user if there is a blocking problem with the build/test execution that you cannot auto-correct.** Otherwise, proceed through the turns automatically.

* **Turn 1: Workspace Initialization**
* Create directories, `Cargo.toml` (workspace), and the `.gitignore` (tracking `Cargo.lock`).
* **Commit:** `chore: init workspace and gitignore`


* **Turn 2: Protocol (Core)**
* Implement `rusty-genius-core` with all Enums, Manifests, and Error types (`thiserror`).
* **Commit:** `feat(core): implement protocol, manifests, and errors`


* **Turn 3: Assets (Facecrab)**
* Implement `facecrab` using `surf`, `futures`, and either `smol` or `async-std`.
* Implement Registry logic, Asset Manager, and embedded defaults.
* **Commit:** `feat(facecrab): implement asset registry and downloader via surf`


* **Turn 4: Engine (Cortex)**
* Implement `rusty-genius--cortex`.
* Create the `Backend` trait.
* Implement `PinkyBackend` (Stub) and `LlamaBackend` (Real) guarded by feature flags.
* **Commit:** `feat(cortex): implement engine with stub and real backends`


* **Turn 5: Orchestration (Brainstem)**
* Implement `rusty-genius--stem`.
* Connect the Event Loop, implement TTL/Hibernation logic.
* **Commit:** `feat(stem): implement orchestrator event loop`


* **Turn 6: Facade (Genius)**
* Implement `rusty-genius` re-exports and public API surface.
* **Commit:** `feat(genius): implement public facade`


* **Turn 7: Quality Assurance (Brainteaser) & Data**
* Implement `rusty-genius--teaser` fixture scanner and test harness.
* Generate physical fixture files for Qwen models.
* Create `scripts/pinky.sh` and `scripts/metal.sh` (ensure `chmod +x`).
* **Commit:** `test(teaser): implement fixture harness, test data, and scripts`


* **Turn 8: Intermediate Validation (Automated)**
* **Action:** The Agent must execute `./scripts/pinky.sh`.
* **Goal:** Verify the logic and orchestration flow without downloading heavy models.
* **Commit:** `test: verify pinky stub pipeline`


* **Turn 9: Final Validation (Automated Real Engine)**
* **Action:** The Agent must execute `./scripts/metal.sh` (this will download the ~2.5GB model).
* **Goal:** Verify real inference.
* **Recovery:** If this step fails or panics, you must add a "Turn 10" to correct the issues and retry until it passes.



---

## 5. Overall Validation Plan

If the agent environment cannot execute the scripts in Turn 8/9 due to sandbox limitations, output this final block for the user:

* **Objective:** Ensure the system can download a real model, load it into RAM (within 4GB limit), and generate a coherent response.
* **Target:** Use the **Qwen2.5-3B-Instruct** fixture generated in **Turn 7**.
* **Action:**
```bash
./scripts/metal.sh

```


* **Success Criteria:**
1. Facecrab successfully downloads the Qwen GGUF (~2.5GB) via `surf`.
2. The engine boots in `real-engine` mode (with Metal/CUDA if available).
3. The system processes the fixture prompt.
4. The system outputs a coherent text response + usage stats.



---

### **Next Step:**

Begin next turn.

---

## 6. Security & GitHub Actions Review

All GitHub Actions workflows must be reviewed for:

1. **Security Concerns:**
   - No hardcoded secrets or credentials
   - Proper use of GitHub secrets for sensitive data
   - Least-privilege permissions (jobs and steps)
   - Safe use of third-party actions (pinned versions)
   - No dangerous script injection vectors

2. **Best Practices:**
   - Proper artifact handling and retention
   - Reasonable job timeouts
   - Safe git operations (signing commits, atomic operations)
   - Proper error handling and reporting
   - Clear job naming and documentation

3. **Specification Compliance:**
   - Tests use local temp directory isolation (`target/tmp`)
   - No `tokio`/`reqwest` in dependency tree validation
   - Validates spec requirements from this prompt
   - Generates timestamped reports

Review file: `.github/workflows/code-review.yml`
