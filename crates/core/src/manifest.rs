use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserManifest {
    pub name: String,
    pub model: String,
    // Add other fields as needed
}

impl Default for UserManifest {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            model: "llama-2-7b".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub repo: String,
    pub filename: String,
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    // Model loading parameters
    pub quant: Option<String>,
    pub context_size: Option<u32>,
    pub ttl_seconds: Option<i64>,

    // Inference generation parameters
    pub show_thinking: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            // Model
            quant: None,
            context_size: Some(2048),
            ttl_seconds: None,
            // Inference
            show_thinking: false,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            repetition_penalty: Some(1.1),
            max_tokens: None,
        }
    }
}

impl UserManifest {
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            name: if !other.name.is_empty() && other.name != "default" {
                other.name.clone()
            } else {
                self.name.clone()
            },
            model: if !other.model.is_empty() {
                other.model.clone()
            } else {
                self.model.clone()
            },
        }
    }
}
