#![cfg(feature = "genai")]

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::{InferenceEvent, ThoughtEvent};
use serde::{Deserialize, Serialize};

// ── API Configuration ──

/// Authentication and endpoint configuration for the Gemini API.
#[derive(Debug, Clone)]
pub enum GeminiApiConfig {
    /// Google AI Studio — authenticated via API key.
    AiStudio { api_key: String },
    /// Vertex AI — authenticated via Bearer token.
    VertexAi {
        project_id: String,
        location: String,
        access_token: String,
    },
}

// ── Request / Response serde types ──

#[derive(Serialize)]
struct GenerateRequest {
    contents: Vec<ContentBlock>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Serialize)]
struct ContentBlock {
    role: String,
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct Part {
    text: String,
}

#[derive(Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
}

#[derive(Serialize)]
struct EmbedRequest {
    content: EmbedContent,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct StreamChunk {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Option<Vec<CandidatePart>>,
}

#[derive(Deserialize)]
struct CandidatePart {
    text: Option<String>,
    thought: Option<bool>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: EmbedValues,
}

#[derive(Deserialize)]
struct EmbedValues {
    values: Vec<f32>,
}

// ── URL construction (public for testing) ──

/// Build the streaming inference URL for the given config and model.
pub fn infer_url(config: &GeminiApiConfig, model: &str) -> String {
    match config {
        GeminiApiConfig::AiStudio { .. } => {
            format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse",
                model
            )
        }
        GeminiApiConfig::VertexAi {
            project_id,
            location,
            ..
        } => {
            format!(
                "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent?alt=sse",
                location = location,
                project = project_id,
                model = model,
            )
        }
    }
}

/// Build the embed URL for the given config and model.
pub fn embed_url(config: &GeminiApiConfig, model: &str) -> String {
    match config {
        GeminiApiConfig::AiStudio { .. } => {
            format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
                model
            )
        }
        GeminiApiConfig::VertexAi {
            project_id,
            location,
            ..
        } => {
            format!(
                "https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:embedContent",
                location = location,
                project = project_id,
                model = model,
            )
        }
    }
}

/// Build the JSON body for a streaming inference request.
pub fn build_infer_body(prompt: &str, config: &InferenceConfig) -> serde_json::Value {
    let request = GenerateRequest {
        contents: vec![ContentBlock {
            role: "user".to_string(),
            parts: vec![Part {
                text: prompt.to_string(),
            }],
        }],
        generation_config: Some(GenerationConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            max_output_tokens: config.max_tokens,
        }),
    };
    serde_json::to_value(request).expect("serialize infer body")
}

/// Build the JSON body for an embed request.
pub fn build_embed_body(input: &str) -> serde_json::Value {
    let request = EmbedRequest {
        content: EmbedContent {
            parts: vec![Part {
                text: input.to_string(),
            }],
        },
    };
    serde_json::to_value(request).expect("serialize embed body")
}

/// Parse a single SSE `data: ...` line into extracted text and optional finish reason.
pub fn parse_sse_line(line: &str) -> Option<(Option<String>, Option<String>, bool)> {
    let json_str = line.strip_prefix("data: ")?;
    let chunk: StreamChunk = serde_json::from_str(json_str).ok()?;
    let candidate = chunk.candidates?.into_iter().next()?;
    let finish_reason = candidate.finish_reason.clone();

    let mut text = None;
    let mut is_thought = false;
    if let Some(content) = candidate.content {
        if let Some(parts) = content.parts {
            if let Some(part) = parts.into_iter().next() {
                text = part.text;
                is_thought = part.thought.unwrap_or(false);
            }
        }
    }
    Some((text, finish_reason, is_thought))
}

// ── GeminiEngine ──

pub struct GeminiEngine {
    config: GeminiApiConfig,
    model: String,
    loaded: bool,
}

impl GeminiEngine {
    pub fn new(config: GeminiApiConfig) -> Self {
        Self {
            config,
            model: "gemini-2.0-flash".to_string(),
            loaded: false,
        }
    }

    /// Apply auth headers to a surf request based on the API config.
    fn apply_auth(&self, mut req: surf::RequestBuilder) -> surf::RequestBuilder {
        match &self.config {
            GeminiApiConfig::AiStudio { api_key } => {
                req = req.header("x-goog-api-key", api_key.as_str());
            }
            GeminiApiConfig::VertexAi { access_token, .. } => {
                req = req.header("Authorization", format!("Bearer {}", access_token));
            }
        }
        req
    }
}

#[async_trait]
impl Engine for GeminiEngine {
    async fn load_model(&mut self, model_name: &str) -> Result<()> {
        self.model = model_name.to_string();
        self.loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn default_model(&self) -> String {
        "gemini-2.0-flash".to_string()
    }

    async fn preload_model(&mut self, _model_path: &str, _purpose: &str) -> Result<()> {
        Ok(()) // genai models are remote, no local preload
    }

    async fn infer(
        &mut self,
        _model: Option<&str>,
        prompt: &str,
        config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.loaded {
            return Err(anyhow!("GeminiEngine: no model loaded"));
        }

        let url = infer_url(&self.config, &self.model);
        let body = build_infer_body(prompt, &config);

        let req = surf::post(&url)
            .header("Content-Type", "application/json")
            .body_json(&body)
            .map_err(|e| anyhow!("Failed to build request body: {}", e))?;

        let req = self.apply_auth(req);

        let mut response = req
            .await
            .map_err(|e| anyhow!("Gemini API request failed: {}", e))?;

        if response.status() != 200 {
            let status = response.status();
            let err_body = response
                .body_string()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(anyhow!("Gemini API error {}: {}", status, err_body));
        }

        let raw = response
            .body_string()
            .await
            .map_err(|e| anyhow!("Failed to read response body: {}", e))?;

        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            let mut in_thought = false;

            for line in raw.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some((text, finish_reason, is_thought)) = parse_sse_line(line) {
                    // Handle thought transitions
                    if is_thought && !in_thought {
                        let _ = tx
                            .send(Ok(InferenceEvent::Thought(ThoughtEvent::Start)))
                            .await;
                        in_thought = true;
                    } else if !is_thought && in_thought {
                        let _ = tx
                            .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                            .await;
                        in_thought = false;
                    }

                    if let Some(text) = text {
                        if is_thought {
                            let _ = tx
                                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Delta(text))))
                                .await;
                        } else {
                            let _ = tx.send(Ok(InferenceEvent::Content(text))).await;
                        }
                    }

                    if finish_reason.as_deref() == Some("STOP") {
                        if in_thought {
                            let _ = tx
                                .send(Ok(InferenceEvent::Thought(ThoughtEvent::Stop)))
                                .await;
                        }
                        break;
                    }
                }
            }

            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        })
        .detach();

        Ok(rx)
    }

    async fn embed(
        &mut self,
        _model: Option<&str>,
        input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        if !self.loaded {
            return Err(anyhow!("GeminiEngine: no model loaded"));
        }

        let url = embed_url(&self.config, &self.model);
        let body = build_embed_body(input);

        let req = surf::post(&url)
            .header("Content-Type", "application/json")
            .body_json(&body)
            .map_err(|e| anyhow!("Failed to build embed request body: {}", e))?;

        let req = self.apply_auth(req);

        let mut response = req
            .await
            .map_err(|e| anyhow!("Gemini embed API request failed: {}", e))?;

        if response.status() != 200 {
            let status = response.status();
            let err_body = response
                .body_string()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(anyhow!("Gemini embed API error {}: {}", status, err_body));
        }

        let raw = response
            .body_string()
            .await
            .map_err(|e| anyhow!("Failed to read embed response body: {}", e))?;

        let embed_resp: EmbedResponse =
            serde_json::from_str(&raw).map_err(|e| anyhow!("Failed to parse embed response: {}", e))?;

        let values = embed_resp.embedding.values;

        let (mut tx, rx) = mpsc::channel(100);

        smol::spawn(async move {
            let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;
            let _ = tx.send(Ok(InferenceEvent::Embedding(values))).await;
            let _ = tx.send(Ok(InferenceEvent::Complete)).await;
        })
        .detach();

        Ok(rx)
    }
}
