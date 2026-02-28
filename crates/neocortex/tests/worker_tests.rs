use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::memory::{
    InMemoryMemoryStore, MemoryObject, MemoryObjectType, MockEmbeddingProvider,
};
use rusty_genius_core::protocol::{MemoryBody, MemoryCommand, MemoryInput, MemoryOutput};
use rusty_genius_neocortex::NeocortexWorker;
use std::time::Duration;

fn make_object(id: &str, short_name: &str, content: &str) -> MemoryObject {
    MemoryObject {
        id: id.to_string(),
        short_name: short_name.to_string(),
        long_name: format!("{} (full)", short_name),
        description: format!("Desc: {}", short_name),
        object_type: MemoryObjectType::Fact,
        content: content.to_string(),
        embedding: None,
        metadata: None,
        created_at: 1700000000,
        updated_at: 1700000000,
        ttl: None,
    }
}

fn spawn_worker() -> (
    mpsc::Sender<MemoryInput>,
    mpsc::Receiver<MemoryOutput>,
    async_std::task::JoinHandle<()>,
) {
    let store = Box::new(InMemoryMemoryStore::new());
    let embedder = Box::new(MockEmbeddingProvider::new(16));
    let worker = NeocortexWorker::new(store, embedder);

    let (input_tx, input_rx) = mpsc::channel::<MemoryInput>(64);
    let (output_tx, output_rx) = mpsc::channel::<MemoryOutput>(64);

    let handle = async_std::task::spawn(async move {
        worker.run(input_rx, output_tx).await;
    });

    (input_tx, output_rx, handle)
}

async fn send_recv(
    tx: &mut mpsc::Sender<MemoryInput>,
    rx: &mut mpsc::Receiver<MemoryOutput>,
    id: &str,
    command: MemoryCommand,
) -> MemoryOutput {
    tx.send(MemoryInput {
        id: Some(id.to_string()),
        command,
    })
    .await
    .expect("send failed");

    async_std::future::timeout(Duration::from_secs(5), rx.next())
        .await
        .expect("timeout")
        .expect("channel closed")
}

#[async_std::test]
async fn test_neocortex_worker_store_and_get() {
    let (mut tx, mut rx, _handle) = spawn_worker();

    let obj = make_object("nw1", "test", "some content");
    let resp = send_recv(&mut tx, &mut rx, "s1", MemoryCommand::Store(obj)).await;
    assert!(matches!(resp.body, MemoryBody::Stored(ref id) if id == "nw1"));

    let get_resp = send_recv(
        &mut tx,
        &mut rx,
        "g1",
        MemoryCommand::Get { object_id: "nw1".to_string() },
    )
    .await;
    match get_resp.body {
        MemoryBody::Object(Some(obj)) => {
            assert_eq!(obj.id, "nw1");
            assert!(obj.embedding.is_some(), "Worker should auto-embed");
        }
        other => panic!("Expected Object(Some), got {:?}", other),
    }
}

#[async_std::test]
async fn test_neocortex_worker_recall() {
    let (mut tx, mut rx, _handle) = spawn_worker();

    send_recv(
        &mut tx,
        &mut rx,
        "s1",
        MemoryCommand::Store(make_object("nr1", "sql", "SELECT * FROM users")),
    )
    .await;
    send_recv(
        &mut tx,
        &mut rx,
        "s2",
        MemoryCommand::Store(make_object("nr2", "shader", "void main() {}")),
    )
    .await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "r1",
        MemoryCommand::Recall {
            query: "SELECT".to_string(),
            limit: 10,
            object_type: None,
        },
    )
    .await;
    match resp.body {
        MemoryBody::Recalled(results) => {
            assert!(!results.is_empty());
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

#[async_std::test]
async fn test_neocortex_worker_forget() {
    let (mut tx, mut rx, _handle) = spawn_worker();

    send_recv(
        &mut tx,
        &mut rx,
        "s1",
        MemoryCommand::Store(make_object("nf1", "temp", "temporary")),
    )
    .await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "f1",
        MemoryCommand::Forget { object_id: "nf1".to_string() },
    )
    .await;
    assert!(matches!(resp.body, MemoryBody::Ack));

    let get_resp = send_recv(
        &mut tx,
        &mut rx,
        "g1",
        MemoryCommand::Get { object_id: "nf1".to_string() },
    )
    .await;
    assert!(matches!(get_resp.body, MemoryBody::Object(None)));
}

#[async_std::test]
async fn test_neocortex_worker_ship_is_noop() {
    let (mut tx, mut rx, _handle) = spawn_worker();

    // Store something
    send_recv(
        &mut tx,
        &mut rx,
        "s1",
        MemoryCommand::Store(make_object("ns1", "item", "content")),
    )
    .await;

    // Ship should be Ack (no-op for neocortex)
    let resp = send_recv(&mut tx, &mut rx, "ship1", MemoryCommand::Ship).await;
    assert!(matches!(resp.body, MemoryBody::Ack));

    // Data should still be there (not evicted)
    let get_resp = send_recv(
        &mut tx,
        &mut rx,
        "g1",
        MemoryCommand::Get { object_id: "ns1".to_string() },
    )
    .await;
    assert!(matches!(get_resp.body, MemoryBody::Object(Some(_))));
}

#[async_std::test]
async fn test_neocortex_worker_stop() {
    let (mut tx, _rx, handle) = spawn_worker();

    tx.send(MemoryInput {
        id: Some("stop".to_string()),
        command: MemoryCommand::Stop,
    })
    .await
    .unwrap();

    async_std::future::timeout(Duration::from_secs(2), handle)
        .await
        .expect("Worker should stop");
}

#[async_std::test]
async fn test_neocortex_worker_request_id_correlation() {
    let (mut tx, mut rx, _handle) = spawn_worker();

    let obj = make_object("nc1", "corr", "content");
    let resp = send_recv(&mut tx, &mut rx, "custom-req-99", MemoryCommand::Store(obj)).await;
    assert_eq!(resp.id, Some("custom-req-99".to_string()));
}
