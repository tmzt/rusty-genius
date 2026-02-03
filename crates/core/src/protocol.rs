use crate::manifest::InferenceConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEvent {
    ProcessStart,
    Thought(ThoughtEvent),
    Content(String),
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThoughtEvent {
    Start,
    Delta(String),
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemInput {
    LoadModel(String),
    Infer {
        prompt: String,
        config: InferenceConfig,
    },
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainstemOutput {
    Event(InferenceEvent),
    Error(String),
}
