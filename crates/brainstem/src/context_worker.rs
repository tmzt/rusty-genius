use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::context::ContextStore;
use rusty_genius_core::protocol::{ContextBody, ContextCommand, ContextInput, ContextOutput};

pub struct ContextWorker {
    store: Box<dyn ContextStore>,
}

impl ContextWorker {
    pub fn new(store: Box<dyn ContextStore>) -> Self {
        Self { store }
    }

    pub async fn run(
        &self,
        mut input_rx: mpsc::Receiver<ContextInput>,
        mut output_tx: mpsc::Sender<ContextOutput>,
    ) {
        while let Some(msg) = input_rx.next().await {
            let request_id = msg.id.clone();
            let body = match msg.command {
                ContextCommand::Get { key } => match self.store.get(&key).await {
                    Ok(value) => ContextBody::Value(value),
                    Err(e) => ContextBody::Error(e.to_string()),
                },
                ContextCommand::Set { key, value } => match self.store.set(&key, &value).await {
                    Ok(()) => ContextBody::Ack,
                    Err(e) => ContextBody::Error(e.to_string()),
                },
                ContextCommand::Delete { key } => match self.store.delete(&key).await {
                    Ok(()) => ContextBody::Ack,
                    Err(e) => ContextBody::Error(e.to_string()),
                },
                ContextCommand::ListKeys { pattern } => {
                    match self.store.list_keys(&pattern).await {
                        Ok(keys) => ContextBody::Keys(keys),
                        Err(e) => ContextBody::Error(e.to_string()),
                    }
                }
                ContextCommand::FlushAll => match self.store.flush_all().await {
                    Ok(()) => ContextBody::Ack,
                    Err(e) => ContextBody::Error(e.to_string()),
                },
            };

            if output_tx
                .send(ContextOutput {
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
