// Copyright 2024-2026 TME
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Brainstem Protocol
//!
//! Defines the internal communication structures (`BrainstemInput`, `BrainstemOutput`)
//! used for orchestrating AI tasks within the `rusty-genius` workspace. These
//! types wrap the public `thinkerv1` protocol messages, adding necessary context
//! for the internal event loop.

use rusty_genius_thinkerv1 as thinkerv1;
use serde::{Deserialize, Serialize};

// --- Input to the Brainstem ---

/// Wraps an inbound command for the `brainstem` orchestrator.
///
/// Every input must have an ID to correlate it with the corresponding output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainstemInput {
    pub id: String,
    pub command: BrainstemCommand,
}

/// The set of commands the `brainstem` can execute.
///
/// These commands closely mirror the `thinkerv1` protocol actions, but are
/// intended for internal use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemCommand {
    /// Command to ensure a model is downloaded and ready.
    EnsureModel(thinkerv1::EnsureRequest),
    /// Command to perform inference.
    Inference(thinkerv1::InferenceRequest),
    /// Command to generate embeddings.
    Embed(thinkerv1::EmbedRequest),
    /// Command to unload all models and reset the engine.
    Reset,
    /// Command to gracefully shut down the orchestrator.
    Shutdown,
}

// --- Output from the Brainstem ---

/// Wraps an outbound event from the `brainstem` orchestrator.
///
/// Every output includes the ID from the originating `BrainstemInput` to allow
/// callers (like the `ogenius` server) to route asynchronous events correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainstemOutput {
    pub id: String,
    pub body: BrainstemBody,
}

/// The content of a `BrainstemOutput` message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemBody {
    /// An event related to the `thinkerv1` protocol.
    Thinker(thinkerv1::Response),
    /// An error that occurred during processing.
    Error(String),
    /// A simple acknowledgement that a task is complete.
    Done,
}

impl BrainstemOutput {
    /// Helper to create a new `BrainstemOutput` with a `thinkerv1::Response`.
    pub fn new_thinker(id: String, response: thinkerv1::Response) -> Self {
        Self {
            id,
            body: BrainstemBody::Thinker(response),
        }
    }

    /// Helper to create a new error output.
    pub fn new_error(id: String, message: impl Into<String>) -> Self {
        Self {
            id,
            body: BrainstemBody::Error(message.into()),
        }
    }

    /// Helper to create a new 'done' acknowledgement.
    pub fn new_done(id: String) -> Self {
        Self {
            id,
            body: BrainstemBody::Done,
        }
    }
}
