use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{
    BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, ContextBody, ContextCommand,
    ContextInput, ContextOutput, InferenceConfig, InferenceEvent,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tide::{Body, Request, Response, StatusCode};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelResponse {
    pub id: String,
    pub object: String,
    pub purpose: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelResponse>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    #[allow(dead_code)]
    pub role: String,
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

#[derive(Deserialize)]
struct ContextMessageContent {
    command: String,
    #[serde(default)]
    key: Option<String>,
    #[serde(default)]
    value: Option<String>,
    #[serde(default)]
    pattern: Option<String>,
}

#[derive(Serialize)]
struct ContextResultContent {
    #[serde(rename = "type")]
    result_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keys: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

use async_std::sync::Mutex;

#[derive(Clone)]
pub struct ApiState {
    pub input_tx: mpsc::Sender<BrainstemInput>,
    pub output_senders: Arc<Mutex<Vec<mpsc::Sender<BrainstemOutput>>>>,
    pub context_tx: mpsc::Sender<ContextInput>,
    pub context_output_senders: Arc<Mutex<Vec<mpsc::Sender<ContextOutput>>>>,
    pub ws_addr: String,
}

pub async fn list_models(req: Request<ApiState>) -> tide::Result {
    eprintln!("DEBUG: list_models entry");
    let state = req.state();

    let request_id = format!(
        "api-list-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    );

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    input_tx
        .send(BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::ListModels,
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let timeout = std::time::Duration::from_secs(10);
    let mut models_vec = Vec::new();

    while let Ok(msg_opt) = async_std::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg_opt {
            if output.id.as_ref() == Some(&request_id) {
                match output.body {
                    BrainstemBody::ModelList(m) => {
                        models_vec = m;
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide::Error::from_str(500, e));
                    }
                    _ => {}
                }
            }
        } else {
            break;
        }
    }

    let models = models_vec
        .into_iter()
        .map(|desc| ModelResponse {
            id: desc.id,
            object: "model".to_string(),
            purpose: desc.purpose,
        })
        .collect();

    let resp = ModelList {
        object: "list".to_string(),
        data: models,
    };
    Ok(Response::builder(StatusCode::Ok)
        .body(Body::from_json(&resp)?)
        .build())
}

pub async fn chat_completions(mut req: Request<ApiState>) -> tide::Result {
    eprintln!("DEBUG: chat_completions entry");
    let body: ChatCompletionRequest = req.body_json().await?;
    eprintln!("DEBUG: chat_completions body parsed");
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

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

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
    let timeout = std::time::Duration::from_secs(30);

    while let Ok(msg_opt) = async_std::future::timeout(timeout, rx.next()).await {
        eprintln!(
            "DEBUG: chat_completions [{}] received result message: {:?}",
            request_id, msg_opt
        );
        if let Some(output) = msg_opt {
            if output.id.as_ref() == Some(&request_id) {
                match output.body {
                    BrainstemBody::Event(InferenceEvent::Content(c)) => {
                        eprintln!("DEBUG: [{}] received Content", request_id);
                        full_content.push_str(&c);
                    }
                    BrainstemBody::Event(InferenceEvent::Complete) => {
                        eprintln!("DEBUG: [{}] received Complete", request_id);
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide::Error::from_str(500, e));
                    }
                    _ => {}
                }
            }
        } else {
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
        .body(Body::from_json(&response)?)
        .build())
}

pub async fn embeddings(mut req: Request<ApiState>) -> tide::Result {
    eprintln!("DEBUG: embeddings entry");
    let body: EmbeddingRequest = req.body_json().await?;
    eprintln!("DEBUG: embeddings body parsed");
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

    input_tx
        .send(BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::Embed {
                model: Some(body.model.clone()),
                input,
                config: InferenceConfig::default(),
            },
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let mut embedding_vec: Option<Vec<f32>> = None;
    let timeout = std::time::Duration::from_secs(60);

    while let Ok(msg_opt) = async_std::future::timeout(timeout, rx.next()).await {
        eprintln!(
            "DEBUG: embeddings [{}] received result message: {:?}",
            request_id, msg_opt
        );
        if let Some(output) = msg_opt {
            if output.id.as_ref() == Some(&request_id) {
                match output.body {
                    BrainstemBody::Event(InferenceEvent::Embedding(emb)) => {
                        eprintln!("DEBUG: [{}] received Embedding", request_id);
                        embedding_vec = Some(emb);
                    }
                    BrainstemBody::Event(InferenceEvent::Complete) => {
                        eprintln!("DEBUG: [{}] received Complete", request_id);
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide::Error::from_str(500, e));
                    }
                    _ => {}
                }
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
        Ok(Response::builder(StatusCode::Ok)
            .body(Body::from_json(&response)?)
            .build())
    } else {
        Ok(Response::builder(StatusCode::InternalServerError)
            .body("No embedding in response")
            .build())
    }
}

pub async fn get_config(req: Request<ApiState>) -> tide::Result {
    let state = req.state();
    let response = ApiConfig {
        ws_addr: state.ws_addr.clone(),
    };
    Ok(Response::builder(StatusCode::Ok)
        .body(Body::from_json(&response)?)
        .build())
}

pub async fn reset_engine(req: Request<ApiState>) -> tide::Result {
    eprintln!("DEBUG: reset_engine entry");
    let state = req.state();

    let request_id = format!(
        "api-reset-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    );

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    input_tx
        .send(BrainstemInput {
            id: Some(request_id.clone()),
            command: BrainstemCommand::Reset,
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let timeout = std::time::Duration::from_secs(10);
    // Wait for acknowledgment
    while let Ok(msg_opt) = async_std::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg_opt {
            if output.id.as_ref() == Some(&request_id) {
                match output.body {
                    BrainstemBody::Event(InferenceEvent::Complete) => {
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide::Error::from_str(500, e));
                    }
                    _ => {}
                }
            }
        } else {
            break;
        }
    }

    Ok(Response::builder(StatusCode::Ok)
        .body("Engine reset")
        .build())
}

pub async fn context_chat(mut req: Request<ApiState>) -> tide::Result {
    eprintln!("DEBUG: context_chat entry");
    let body: ChatCompletionRequest = req.body_json().await?;
    let state = req.state();

    let user_content = body
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let request_id = format!(
        "ctx-api-context-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    );

    let parsed: ContextMessageContent = match serde_json::from_str(&user_content) {
        Ok(c) => c,
        Err(e) => {
            let result = ContextResultContent {
                result_type: "error".to_string(),
                value: None,
                keys: None,
                message: Some(format!("Invalid context command JSON: {}", e)),
            };
            let response = ChatCompletionResponse {
                id: request_id,
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
                        content: serde_json::to_string(&result).unwrap(),
                    },
                    finish_reason: "stop".to_string(),
                }],
            };
            return Ok(Response::builder(StatusCode::Ok)
                .body(Body::from_json(&response)?)
                .build());
        }
    };

    let command = match parsed.command.as_str() {
        "set" => {
            let key = parsed.key.unwrap_or_default();
            let value = parsed.value.unwrap_or_default();
            ContextCommand::Set { key, value }
        }
        "get" => {
            let key = parsed.key.unwrap_or_default();
            ContextCommand::Get { key }
        }
        "delete" => {
            let key = parsed.key.unwrap_or_default();
            ContextCommand::Delete { key }
        }
        "list_keys" => {
            let pattern = parsed.pattern.unwrap_or_else(|| "*".to_string());
            ContextCommand::ListKeys { pattern }
        }
        "flush_all" => ContextCommand::FlushAll,
        other => {
            let result = ContextResultContent {
                result_type: "error".to_string(),
                value: None,
                keys: None,
                message: Some(format!("Unknown command: {}", other)),
            };
            let response = ChatCompletionResponse {
                id: request_id,
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
                        content: serde_json::to_string(&result).unwrap(),
                    },
                    finish_reason: "stop".to_string(),
                }],
            };
            return Ok(Response::builder(StatusCode::Ok)
                .body(Body::from_json(&response)?)
                .build());
        }
    };

    let mut context_tx = state.context_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.context_output_senders.lock().await;
        senders.push(tx);
    }

    context_tx
        .send(ContextInput {
            id: Some(request_id.clone()),
            command,
        })
        .await
        .map_err(|e| tide::Error::from_str(500, e))?;

    let timeout = std::time::Duration::from_secs(10);
    let mut result_body: Option<ContextBody> = None;

    while let Ok(msg_opt) = async_std::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg_opt {
            if output.id.as_ref() == Some(&request_id) {
                result_body = Some(output.body);
                break;
            }
        } else {
            break;
        }
    }

    let result = match result_body {
        Some(ContextBody::Ack) => ContextResultContent {
            result_type: "ack".to_string(),
            value: None,
            keys: None,
            message: None,
        },
        Some(ContextBody::Value(v)) => ContextResultContent {
            result_type: "value".to_string(),
            value: Some(match v {
                Some(s) => serde_json::Value::String(s),
                None => serde_json::Value::Null,
            }),
            keys: None,
            message: None,
        },
        Some(ContextBody::Keys(k)) => ContextResultContent {
            result_type: "keys".to_string(),
            value: None,
            keys: Some(k),
            message: None,
        },
        Some(ContextBody::Error(e)) => ContextResultContent {
            result_type: "error".to_string(),
            value: None,
            keys: None,
            message: Some(e),
        },
        None => ContextResultContent {
            result_type: "error".to_string(),
            value: None,
            keys: None,
            message: Some("Timeout waiting for context response".to_string()),
        },
    };

    let response = ChatCompletionResponse {
        id: request_id,
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
                content: serde_json::to_string(&result).unwrap(),
            },
            finish_reason: "stop".to_string(),
        }],
    };

    Ok(Response::builder(StatusCode::Ok)
        .body(Body::from_json(&response)?)
        .build())
}
