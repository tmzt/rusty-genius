// Redis store and bootstrap logic live in the `striatum` crate.
// Re-export for backward compatibility.
pub use rusty_genius_striatum::bootstrap;
pub use rusty_genius_striatum::store;
pub use rusty_genius_striatum::RedisMemoryStore;

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::memory::{EmbeddingProvider, MemoryStore};
use rusty_genius_core::protocol::{MemoryBody, MemoryCommand, MemoryInput, MemoryOutput};

pub struct PfcWorker {
    store: Box<dyn MemoryStore>,
    embedder: Box<dyn EmbeddingProvider>,
    neocortex: Option<Box<dyn MemoryStore>>,
}

impl PfcWorker {
    pub fn new(
        store: Box<dyn MemoryStore>,
        embedder: Box<dyn EmbeddingProvider>,
        neocortex: Option<Box<dyn MemoryStore>>,
    ) -> Self {
        Self {
            store,
            embedder,
            neocortex,
        }
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
                    // Embed if not already provided
                    if object.embedding.is_none() {
                        match self.embedder.embed(&object.content).await {
                            Ok(vec) => object.embedding = Some(vec),
                            Err(e) => {
                                output_tx.send(MemoryOutput { id: request_id, body: MemoryBody::Error(format!("Embedding failed: {}", e)) }).await.ok();
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
                            output_tx.send(MemoryOutput { id: request_id, body: MemoryBody::Error(format!("Embedding failed: {}", e)) }).await.ok();
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

                MemoryCommand::Ship => match self.ship().await {
                    Ok(()) => MemoryBody::Ack,
                    Err(e) => MemoryBody::Error(e.to_string()),
                },

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

    /// Consolidate all PFC objects into Neocortex (long-term memory).
    async fn ship(&self) -> Result<(), rusty_genius_core::error::GeniusError> {
        let neocortex = self.neocortex.as_ref().ok_or_else(|| {
            rusty_genius_core::error::GeniusError::MemoryError(
                "No neocortex store configured for shipping".to_string(),
            )
        })?;

        let objects = self.store.list_all().await?;
        for object in objects {
            let id = object.id.clone();
            neocortex.store(object).await?;
            self.store.forget(&id).await?;
        }

        Ok(())
    }
}
