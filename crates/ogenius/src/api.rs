use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::protocol::{BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput};
use rusty_genius_stem::Orchestrator;
use rusty_genius_thinkerv1::{
    EventResponse, InferenceConfig, InferenceRequest, EmbedRequest,
    new_ensure_request, new_inference_request, new_embed_request,
    Response, StatusResponse, Request
};
use serde::{Deserialize, Serialize};
use tide_smol::{Body, Request as tide_smol::Request, Response as tide_smol::Response, StatusCode};

use facecrab::AssetAuthority; // Import AssetAuthority
use facecrab::registry::ModelEntry; // Import ModelEntry

// --- API types ---

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

use smol::lock::Mutex;

#[derive(Clone)]
pub struct ApiState {
    pub input_tx: mpsc::Sender<BrainstemInput>,
    pub output_senders: Arc<Mutex<Vec<mpsc::Sender<BrainstemOutput>>>>,
    pub ws_addr: String,
}

// --- API Handlers ---

pub async fn list_models(req: tide_smol::Request<ApiState>) -> tide_smol::Result {
    eprintln!("DEBUG: list_models entry");
    let asset_authority = facecrab::AssetAuthority::new()?;
    let models_vec = asset_authority.list_models();

    let models = models_vec
        .into_iter()
        .map(|model_entry| ModelResponse {
            id: model_entry.name,
            object: "model".to_string(),
            purpose: model_entry.purpose.to_string(), // Convert ModelPurpose to String
        })
        .collect();

    let resp = ModelList {
        object: "list".to_string(),
        data: models,
    };
    Ok(tide_smol::Response::builder(StatusCode::Ok)
        .body(Body::from_json(&resp)?)
        .build())
}

pub async fn chat_completions(mut req: tide_smol::Request<ApiState>) -> tide_smol::Result {
    eprintln!("DEBUG: chat_completions entry");
    let body: ChatCompletionRequest = req.body_json().await?;
    eprintln!("DEBUG: chat_completions body parsed");
    let state = req.state();

    let prompt = body
        .messages
        .last()
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let request_id = Uuid::new_v4().to_string();
    eprintln!("DEBUG: chat_completions [{}] prompt: {}", request_id, prompt);

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    let infer_req_enum = new_inference_request(prompt.to_string(), Some(InferenceConfig::default()));
    let infer_req = match infer_req_enum {
        Request::Inference(req) => req,
        _ => unreachable!(), // new_inference_request always returns Request::Inference
    };

    input_tx
        .send(BrainstemInput {
            id: request_id.clone(),
            command: BrainstemCommand::Inference(infer_req),
        })
        .await
        .map_err(|e| tide_smol::Error::from_str(500, e))?;

    let mut full_content = String::new();
    let timeout = std::time::Duration::from_secs(30);

    while let Ok(msg_opt) = smol::future::timeout(timeout, rx.next()).await {
        eprintln!(
            "DEBUG: chat_completions [{}] received result message: {:?}",
            request_id, msg_opt
        );
        if let Some(output) = msg_opt {
            if output.id == request_id { // Changed from Some(&request_id)
                match output.body {
                    BrainstemBody::Thinker(Response::Event(EventResponse::Content{ content, .. })) => { // Changed
                        eprintln!("DEBUG: [{}] received Content", request_id);
                        full_content.push_str(&content);
                    }
                    BrainstemBody::Thinker(Response::Event(EventResponse::Complete{ .. })) => { // Changed
                        eprintln!("DEBUG: [{}] received Complete", request_id);
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide_smol::Error::from_str(500, e));
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

    Ok(tide_smol::Response::builder(StatusCode::Ok)
        .body(Body::from_json(&response)?)
        .build())
}

pub async fn embeddings(mut req: tide_smol::Request<ApiState>) -> tide_smol::Result {
    eprintln!("DEBUG: embeddings entry");
    let body: EmbeddingRequest = req.body_json().await?;
    eprintln!("DEBUG: embeddings body parsed");
    let state = req.state();

    let request_id = Uuid::new_v4().to_string();
    let input = body.input.clone();
    eprintln!("DEBUG: embeddings [{}] request for: {}", request_id, input);

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    let embed_req_enum = new_embed_request(input.clone());
    let embed_req = match embed_req_enum {
        Request::Embed(req) => req,
        _ => unreachable!(), // new_embed_request always returns Request::Embed
    };

    input_tx
        .send(BrainstemInput {
            id: request_id.clone(),
            command: BrainstemCommand::Embed(embed_req), // Changed
        })
        .await
        .map_err(|e| tide_smol::Error::from_str(500, e))?;

    let mut embedding_vec: Option<Vec<f32>> = None;
    let timeout = std::time::Duration::from_secs(60);

    while let Ok(msg_opt) = smol::future::timeout(timeout, rx.next()).await {
        eprintln!(
            "DEBUG: embeddings [{}] received result message: {:?}",
            request_id, msg_opt
        );
        if let Some(output) = msg_opt {
            if output.id == request_id { // Changed from Some(&request_id)
                match output.body {
                    BrainstemBody::Thinker(Response::Event(EventResponse::Embedding{ vector_hex, .. })) => { // Changed
                        eprintln!("DEBUG: [{}] received Embedding", request_id);
                        // Convert hex string to f32 vector
                        let mut vec_f32 = Vec::new();
                        for chunk in vector_hex.as_bytes().chunks(8) { // Each f32 is 4 bytes (8 hex chars)
                            if chunk.len() == 8 {
                                let mut bytes = [0u8; 4]; // Use 4 bytes for f32
                                for (i, &byte) in chunk.iter().enumerate() {
                                    bytes[i] = byte;
                                }
                                // Correctly convert u32 to f32 using from_bits
                                let bits = u32::from_be_bytes(bytes);
                                vec_f32.push(f32::from_bits(bits));
                            }
                        }
                        embedding_vec = Some(vec_f32);
                    }
                    BrainstemBody::Thinker(Response::Event(EventResponse::Complete{ .. })) => { // Changed
                        eprintln!("DEBUG: [{}] received Complete", request_id);
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide_smol::Error::from_str(500, e));
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
        Ok(tide_smol::Response::builder(StatusCode::Ok)
            .body(Body::from_json(&response)?)
            .build())
    } else {
        Ok(tide_smol::Response::builder(StatusCode::InternalServerError)
            .body("No embedding in response")
            .build())
    }
}

pub async fn get_config(req: tide_smol::Request<ApiState>) -> tide_smol::Result {
    let state = req.state();
    let response = ApiConfig {
        ws_addr: state.ws_addr.clone(),
    };
    Ok(tide_smol::Response::builder(StatusCode::Ok)
        .body(Body::from_json(&response)?)
        .build())
}

pub async fn reset_engine(req: tide_smol::Request<ApiState>) -> tide_smol::Result {
    eprintln!("DEBUG: reset_engine entry");
    let state = req.state();

    let request_id = Uuid::new_v4().to_string();

    let mut input_tx = state.input_tx.clone();
    let (tx, mut rx) = mpsc::channel(100);

    {
        let mut senders = state.output_senders.lock().await;
        senders.push(tx);
    }

    input_tx
        .send(BrainstemInput {
            id: request_id.clone(),
            command: BrainstemCommand::Reset,
        })
        .await
        .map_err(|e| tide_smol::Error::from_str(500, e))?;

    let timeout = std::time::Duration::from_secs(10);
    // Wait for acknowledgment
    while let Ok(msg_opt) = smol::future::timeout(timeout, rx.next()).await {
        if let Some(output) = msg_opt {
            if output.id == request_id { // Changed from Some(&request_id)
                match output.body {
                    BrainstemBody::Thinker(Response::Event(EventResponse::Complete{ .. })) => { // Changed
                        break;
                    }
                    BrainstemBody::Error(e) => {
                        return Err(tide_smol::Error::from_str(500, e));
                    }
                    _ => {}
                }
            }
        } else {
            break;
        }
    }

    Ok(tide_smol::Response::builder(StatusCode::Ok)
        .body("Engine reset")
        .build())
}
