# Terminology

rusty-genius uses biologically inspired naming for its crates. This document maps each term to the brain structure it references and its role in the project.

| Term | Brain Structure | Role in Project | Crate |
|------|----------------|-----------------|-------|
| **core** | — | Shared types, protocols, traits | `rusty-genius-core` |
| **cortex** | Cerebral cortex | Inference engine layer (llama.cpp, genai) | `rusty-genius-cortex` |
| **brainstem** | Brain stem | Orchestrator, event loop, state management | `rusty-genius-stem` |
| **pfc** | Prefrontal cortex | Short-term / working memory worker | `rusty-genius-pfc` |
| **striatum** | Striatum (basal ganglia) | Redis-backed memory search & indexing | `rusty-genius-striatum` |
| **caudate** | Caudate nucleus | Memory policy engine (reserved) | — |
| **neocortex** | Neocortex | Long-term memory worker (SQLite) | `rusty-genius-neocortex` |
| **gyrus** | Cerebral gyrus | Standalone SQLite memory store | `rusty-genius-gyrus` |
| **hippocampus** | Hippocampus | Browser/WASM memory (IndexedDB + FTS5) | `rusty-genius-hippocampus` |
| **facecrab** | — (HuggingFace pun) | Asset authority, model registry & downloads | `rusty-genius-facecrab` |
| **genius** | — (intelligence) | Public facade / API surface | `rusty-genius` |
| **brainteaser** | — (puzzle) | Integration testing harness | `rusty-genius-teaser` |
