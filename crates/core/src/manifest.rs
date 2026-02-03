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
pub struct InferenceConfig {
    pub temperature: f32,
    pub max_tokens: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: None,
        }
    }
}
