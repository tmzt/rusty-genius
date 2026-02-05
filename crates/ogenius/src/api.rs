use async_std::sync::Mutex;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius::core::protocol::{
    BrainstemInput, BrainstemOutput, InferenceConfig, InferenceEvent,
};
use rusty_genius::facecrab::registry::ModelRegistry;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tide::{Request, Response, StatusCode};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub models: Vec<ModelResponse>,
    pub preferred: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub _role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Clone)]
pub struct ApiState {
    pub input_tx: mpsc::Sender<BrainstemInput>,
    pub output_rx: Arc<Mutex<mpsc::Receiver<BrainstemOutput>>>,
    pub ws_addr: String,
}

#[derive(Serialize)]
pub struct ApiConfig {
    pub ws_addr: String,
}

pub async fn list_models(_req: Request<ApiState>) -> tide::Result {
    let registry = ModelRegistry::new().map_err(|e| tide::Error::from_str(500, e))?;
    let models = registry.list_models();

    let data: Vec<ModelResponse> = models
        .into_iter()
        .map(|m| ModelResponse {
            id: m.name,
            object: "model".to_string(),
            created: 1677610602,
            owned_by: "rusty-genius".to_string(),
        })
        .collect();

    // Heuristic for preferred model: first one or one containing "qwen"
    let preferred = data
        .iter()
        .find(|m| m.id.to_lowercase().contains("qwen"))
        .or_else(|| data.first())
        .map(|m| m.id.clone());

    Ok(Response::builder(StatusCode::Ok)
        .body(serde_json::to_string(&ModelListResponse {
            object: "list".to_string(),
            models: data,
            preferred,
        })?)
        .build())
}

pub async fn chat_completions(mut req: Request<ApiState>) -> tide::Result {
    let body: ChatCompletionRequest = req.body_json().await?;
    let state = req.state();

    // For now, only non-streaming is implemented as per simple Web UI request
    if body.stream {
        return Ok(Response::builder(StatusCode::NotImplemented)
            .body("Streaming not yet implemented in REST API")
            .build());
    }

    // Use the first user message as the prompt for now
    let prompt = body
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let mut input_tx = state.input_tx.clone();

    // Send inference request
    input_tx
        .send(BrainstemInput::Infer {
            model: Some(body.model.clone()),
            prompt,
            config: InferenceConfig::default(),
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let mut output_rx = state.output_rx.lock().await;
    let mut full_content = String::new();

    while let Some(output) = output_rx.next().await {
        match output {
            BrainstemOutput::Event(InferenceEvent::Content(c)) => {
                full_content.push_str(&c);
            }
            BrainstemOutput::Event(InferenceEvent::Complete) => {
                break;
            }
            BrainstemOutput::Error(e) => {
                return Err(tide::Error::from_str(500, e));
            }
            _ => {}
        }
    }

    let response = ChatCompletionResponse {
        id: "gen-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1677610602,
        model: body.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content: full_content,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Response::builder(StatusCode::Ok)
        .body(serde_json::to_string(&response)?)
        .build())
}

pub async fn get_config(req: Request<ApiState>) -> tide::Result {
    let state = req.state();
    Ok(Response::builder(StatusCode::Ok)
        .body(serde_json::to_string(&ApiConfig {
            ws_addr: state.ws_addr.clone(),
        })?)
        .build())
}
