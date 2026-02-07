use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, InferenceConfig,
    InferenceEvent,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tide::{Request, Response, StatusCode};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelResponse {
    pub id: String,
    pub object: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelResponse>,
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
}

#[derive(Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: String,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
}

#[derive(Serialize)]
pub struct ApiConfig {
    pub ws_addr: String,
}

use async_std::sync::Mutex;

#[derive(Clone)]
pub struct ApiState {
    pub input_tx: mpsc::Sender<BrainstemInput>,
    pub output_senders: Arc<Mutex<Vec<mpsc::Sender<BrainstemOutput>>>>,
    pub ws_addr: String,
}

pub async fn list_models(_req: Request<ApiState>) -> tide::Result {
    let models = vec![ModelResponse {
        id: "any".to_string(),
        object: "model".to_string(),
    }];
    Ok(serde_json::to_string(&ModelList {
        object: "list".to_string(),
        data: models,
    })?
    .into())
}

pub async fn chat_completions(mut req: Request<ApiState>) -> tide::Result {
    let body: ChatCompletionRequest = req.body_json().await?;
    let state = req.state();

    let prompt = body
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let request_id = format!(
        "api-chat-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    );
    eprintln!(
        "DEBUG: chat_completions [{}] prompt: {}",
        request_id, prompt
    );

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    // Register our sender in the global broadcast list
    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    eprintln!("DEBUG: Sending inference request...");
    input_tx
        .send(BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::Infer {
                model: Some(body.model.clone()),
                prompt,
                config: InferenceConfig::default(),
            },
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let mut full_content = String::new();

    eprintln!("DEBUG: Waiting for brainstem output...");
    let timeout = std::time::Duration::from_secs(30);
    let mut last_event = std::time::Instant::now();

    while let Ok(msg) = async_std::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg {
            if output.id.as_ref() != Some(&request_id) {
                continue;
            }

            eprintln!("DEBUG: Received output body type");
            match output.body {
                BrainstemBody::Event(InferenceEvent::Content(c)) => {
                    full_content.push_str(&c);
                    last_event = std::time::Instant::now();
                }
                BrainstemBody::Event(InferenceEvent::Complete) => {
                    break;
                }
                BrainstemBody::Error(e) => {
                    return Err(tide::Error::from_str(500, e));
                }
                _ => {}
            }
        } else {
            break;
        }

        if last_event.elapsed() > timeout {
            break;
        }
    }

    let response = ChatCompletionResponse {
        id: format!("gen-{}", request_id),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
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

pub async fn embeddings(mut req: Request<ApiState>) -> tide::Result {
    let body: EmbeddingRequest = req.body_json().await?;
    let state = req.state();

    let request_id = format!(
        "api-embed-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    );
    let input = body.input.clone();
    eprintln!("DEBUG: embeddings [{}] request for: {}", request_id, input);

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    eprintln!("DEBUG: Sending embedding request...");
    input_tx
        .send(BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::Embed {
                model: Some(body.model.clone()),
                input: input.clone(),
                config: InferenceConfig::default(),
            },
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let mut embedding_vec: Option<Vec<f32>> = None;

    eprintln!("DEBUG: Waiting for brainstem embedding...");
    let timeout = std::time::Duration::from_secs(60); // Embeddings take longer to cold start

    while let Ok(msg) = async_std::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg {
            if output.id.as_ref() != Some(&request_id) {
                continue;
            }

            eprintln!("DEBUG: Received output body type");
            match output.body {
                BrainstemBody::Event(InferenceEvent::Embedding(emb)) => {
                    embedding_vec = Some(emb);
                }
                BrainstemBody::Event(InferenceEvent::Complete) => {
                    break;
                }
                BrainstemBody::Error(e) => {
                    return Err(tide::Error::from_str(500, e));
                }
                _ => {}
            }
        } else {
            break;
        }
    }

    if let Some(vec) = embedding_vec {
        let response = EmbeddingResponse {
            object: "list".to_string(),
            data: vec![EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec,
                index: 0,
            }],
            model: body.model,
        };
        Ok(serde_json::to_string(&response)?.into())
    } else {
        Err(tide::Error::from_str(500, "No embedding in response"))
    }
}

pub async fn get_config(req: Request<ApiState>) -> tide::Result {
    let state = req.state();
    Ok(Response::builder(StatusCode::Ok)
        .body(serde_json::to_string(&ApiConfig {
            ws_addr: state.ws_addr.clone(),
        })?)
        .build())
}
