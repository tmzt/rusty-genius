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

//! # Rusty Genius Thinkerv1 Protocol
//!
//! This crate defines the serializable message schemas for the `thinkerv1` protocol,
//! which is a JSONL-based protocol for AI orchestration.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// --- Requests ---

/// Represents all possible requests in the `thinkerv1` protocol.
/// Each request is tagged with an `action` field.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "action")]
#[serde(rename_all = "lowercase")]
pub enum Request {
    /// A request to ensure a model is downloaded and ready.
    Ensure(EnsureRequest),
    /// A request to perform inference with a model.
    Inference(InferenceRequest),
    /// A request to generate embeddings for a piece of text.
    Embed(EmbedRequest),
}

/// The payload for an `ensure` action.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct EnsureRequest {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub report_status: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config: Option<ModelConfig>,
}

/// The payload for an `inference` action.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_config: Option<InferenceConfig>,
}

/// Configuration for loading a model.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ModelConfig {
    pub quant: Option<String>,
    pub context_size: Option<u32>,
    pub ttl_seconds: Option<i64>, // -1 represents infinite TTL override
}

/// Configuration options for an inference request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct InferenceConfig {
    #[serde(default)]
    pub show_thinking: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
}

/// The payload for an `embed` action.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct EmbedRequest {
    pub id: String,
    pub text: String,
}

impl Request {
    /// Returns the ID of the request.
    pub fn get_id(&self) -> &str {
        match self {
            Request::Ensure(req) => &req.id,
            Request::Inference(req) => &req.id,
            Request::Embed(req) => &req.id,
        }
    }
}

// --- Responses ---

/// Represents all possible responses in the `thinkerv1` protocol.
/// The `untagged` enum allows for different response structures.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Response {
    /// A response indicating the status of an operation (e.g., for `ensure`).
    Status(StatusResponse),
    /// An event-based response (e.g., for `inference` or `embed`).
    Event(EventResponse),
}

/// For responses from 'ensure' action or general status updates.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct StatusResponse {
    pub id: String,
    /// The status of the operation (e.g., "downloading", "ready", "error").
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// For streaming responses from 'inference' or 'embed' actions.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum EventResponse {
    Thought { id: String, content: String },
    Content { id: String, content: String },
    Complete { id: String },
    Embedding {
        id: String,
        #[serde(rename = "vector")]
        vector_hex: String,
    },
}

// --- Helper functions to create requests with new UUIDs ---

/// Creates a new `ensure` request with a unique ID.
pub fn new_ensure_request(model: String, report_status: bool, config: Option<ModelConfig>) -> Request {
    Request::Ensure(EnsureRequest {
        id: Uuid::new_v4().to_string(),
        model,
        report_status,
        model_config: config,
    })
}

/// Creates a new `inference` request with a unique ID.
pub fn new_inference_request(prompt: String, config: Option<InferenceConfig>) -> Request {
    Request::Inference(InferenceRequest {
        id: Uuid::new_v4().to_string(),
        prompt,
        inference_config: config,
    })
}

/// Creates a new `embed` request with a unique ID.
pub fn new_embed_request(text: String) -> Request {
    Request::Embed(EmbedRequest {
        id: Uuid::new_v4().to_string(),
        text,
    })
}
