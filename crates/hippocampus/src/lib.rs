pub mod fts;
pub mod idb;
pub mod schema;
pub mod store;
pub mod wrapper;

pub use store::IdbMemoryStore;

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::memory::{EmbeddingProvider, MemoryStore};
use rusty_genius_core::protocol::{MemoryBody, MemoryCommand, MemoryInput, MemoryOutput};

/// Channel-driven memory worker for the browser, mirroring `NeocortexWorker`.
///
/// Dispatches `MemoryInput` commands to an `IdbMemoryStore` (or any `MemoryStore`)
/// and sends `MemoryOutput` responses back through the output channel.
pub struct HippocampusWorker {
    store: Box<dyn MemoryStore>,
    embedder: Box<dyn EmbeddingProvider>,
}

impl HippocampusWorker {
    pub fn new(store: Box<dyn MemoryStore>, embedder: Box<dyn EmbeddingProvider>) -> Self {
        Self { store, embedder }
    }

    pub async fn run(
        &self,
        mut input_rx: mpsc::Receiver<MemoryInput>,
        mut output_tx: mpsc::Sender<MemoryOutput>,
    ) {
        while let Some(msg) = input_rx.next().await {
            let request_id = msg.id.clone();

            let body = match msg.command {
                MemoryCommand::Store(mut object) => {
                    // Auto-embed if not already provided
                    if object.embedding.is_none() {
                        match self.embedder.embed(&object.content).await {
                            Ok(vec) => object.embedding = Some(vec),
                            Err(e) => {
                                MemoryBody::Error(format!("Embedding failed: {}", e));
                                continue;
                            }
                        }
                    }
                    match self.store.store(object).await {
                        Ok(id) => MemoryBody::Stored(id),
                        Err(e) => MemoryBody::Error(e.to_string()),
                    }
                }

                MemoryCommand::Recall {
                    query,
                    limit,
                    object_type,
                } => {
                    let embedding = match self.embedder.embed(&query).await {
                        Ok(vec) => vec,
                        Err(e) => {
                            MemoryBody::Error(format!("Embedding failed: {}", e));
                            continue;
                        }
                    };
                    match self
                        .store
                        .recall(&query, &embedding, limit, object_type.as_ref())
                        .await
                    {
                        Ok(results) => MemoryBody::Recalled(results),
                        Err(e) => MemoryBody::Error(e.to_string()),
                    }
                }

                MemoryCommand::RecallByVector {
                    embedding,
                    limit,
                    object_type,
                } => match self
                    .store
                    .recall_by_vector(&embedding, limit, object_type.as_ref())
                    .await
                {
                    Ok(results) => MemoryBody::Recalled(results),
                    Err(e) => MemoryBody::Error(e.to_string()),
                },

                MemoryCommand::Get { object_id } => match self.store.get(&object_id).await {
                    Ok(obj) => MemoryBody::Object(obj),
                    Err(e) => MemoryBody::Error(e.to_string()),
                },

                MemoryCommand::Forget { object_id } => {
                    match self.store.forget(&object_id).await {
                        Ok(()) => MemoryBody::Ack,
                        Err(e) => MemoryBody::Error(e.to_string()),
                    }
                }

                MemoryCommand::ListByType { object_type } => {
                    match self.store.list_by_type(&object_type).await {
                        Ok(objects) => MemoryBody::Recalled(objects),
                        Err(e) => MemoryBody::Error(e.to_string()),
                    }
                }

                // Ship is a no-op for Hippocampus (it's the destination, not the source)
                MemoryCommand::Ship => MemoryBody::Ack,

                MemoryCommand::Stop => break,
            };

            if output_tx
                .send(MemoryOutput {
                    id: request_id,
                    body,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    }
}
