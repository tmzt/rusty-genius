use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::memory::{
    InMemoryMemoryStore, LogicElement, LogicElementSubtype, MemoryObject, MemoryObjectType,
    MemoryStore, MockEmbeddingProvider,
};
use rusty_genius_core::protocol::{MemoryBody, MemoryCommand, MemoryInput, MemoryOutput};
use rusty_genius_pfc::PfcWorker;
use std::time::Duration;

fn make_object(
    id: &str,
    short_name: &str,
    object_type: MemoryObjectType,
    content: &str,
) -> MemoryObject {
    MemoryObject {
        id: id.to_string(),
        short_name: short_name.to_string(),
        long_name: format!("{} (full)", short_name),
        description: format!("Desc: {}", short_name),
        object_type,
        content: content.to_string(),
        embedding: None, // Worker should auto-embed
        metadata: None,
        created_at: 1700000000,
        updated_at: 1700000000,
        ttl: None,
    }
}

/// Spawn a PfcWorker and return channels for communication.
fn spawn_worker(
    neocortex: Option<Box<dyn MemoryStore>>,
) -> (
    mpsc::Sender<MemoryInput>,
    mpsc::Receiver<MemoryOutput>,
    async_std::task::JoinHandle<()>,
) {
    let store = Box::new(InMemoryMemoryStore::new());
    let embedder = Box::new(MockEmbeddingProvider::new(16));
    let worker = PfcWorker::new(store, embedder, neocortex);

    let (input_tx, input_rx) = mpsc::channel::<MemoryInput>(64);
    let (output_tx, output_rx) = mpsc::channel::<MemoryOutput>(64);

    let handle = async_std::task::spawn(async move {
        worker.run(input_rx, output_tx).await;
    });

    (input_tx, output_rx, handle)
}

/// Send a command and receive one response with a timeout.
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
        .expect("timeout waiting for response")
        .expect("channel closed")
}

// ── Store command with auto-embedding ──

#[async_std::test]
async fn test_worker_store_auto_embeds() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let obj = make_object("ae1", "auto_embed", MemoryObjectType::Fact, "some content");
    let resp = send_recv(
        &mut tx,
        &mut rx,
        "r1",
        MemoryCommand::Store(obj),
    )
    .await;

    assert_eq!(resp.id, Some("r1".to_string()));
    match resp.body {
        MemoryBody::Stored(id) => assert_eq!(id, "ae1"),
        other => panic!("Expected Stored, got {:?}", other),
    }

    // Verify the object was stored with an embedding via Get
    let get_resp = send_recv(&mut tx, &mut rx, "r2", MemoryCommand::Get { object_id: "ae1".to_string() }).await;
    match get_resp.body {
        MemoryBody::Object(Some(obj)) => {
            assert!(obj.embedding.is_some(), "Worker should have auto-embedded");
            assert_eq!(obj.embedding.unwrap().len(), 16);
        }
        other => panic!("Expected Object(Some), got {:?}", other),
    }
}

// ── Store with pre-existing embedding (no auto-embed) ──

#[async_std::test]
async fn test_worker_store_preserves_existing_embedding() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let mut obj = make_object("pe1", "pre_embed", MemoryObjectType::Fact, "content");
    obj.embedding = Some(vec![1.0; 16]); // Pre-set

    let resp = send_recv(&mut tx, &mut rx, "r1", MemoryCommand::Store(obj)).await;
    match resp.body {
        MemoryBody::Stored(id) => assert_eq!(id, "pe1"),
        other => panic!("Expected Stored, got {:?}", other),
    }

    let get_resp = send_recv(&mut tx, &mut rx, "r2", MemoryCommand::Get { object_id: "pe1".to_string() }).await;
    match get_resp.body {
        MemoryBody::Object(Some(obj)) => {
            let emb = obj.embedding.unwrap();
            assert!(emb.iter().all(|v| (*v - 1.0).abs() < 0.001), "Should preserve original embedding");
        }
        other => panic!("Expected Object(Some), got {:?}", other),
    }
}

// ── Recall command ──

#[async_std::test]
async fn test_worker_recall() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    // Store two objects
    let obj1 = make_object("rc1", "sql_example", MemoryObjectType::Fact, "SELECT * FROM users");
    send_recv(&mut tx, &mut rx, "s1", MemoryCommand::Store(obj1)).await;

    let obj2 = make_object("rc2", "shader_example", MemoryObjectType::Fact, "void main() {}");
    send_recv(&mut tx, &mut rx, "s2", MemoryCommand::Store(obj2)).await;

    // Recall with text query
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
            assert!(!results.is_empty(), "Should find at least one result");
            // SQL result should be in there (text match)
            assert!(results.iter().any(|r| r.id == "rc1"));
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

// ── Get command ──

#[async_std::test]
async fn test_worker_get_existing() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let obj = make_object("g1", "getme", MemoryObjectType::Observation, "observable");
    send_recv(&mut tx, &mut rx, "s1", MemoryCommand::Store(obj)).await;

    let resp = send_recv(&mut tx, &mut rx, "r1", MemoryCommand::Get { object_id: "g1".to_string() }).await;
    match resp.body {
        MemoryBody::Object(Some(obj)) => {
            assert_eq!(obj.id, "g1");
            assert_eq!(obj.content, "observable");
        }
        other => panic!("Expected Object(Some), got {:?}", other),
    }
}

#[async_std::test]
async fn test_worker_get_missing() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "r1",
        MemoryCommand::Get { object_id: "nope".to_string() },
    )
    .await;
    match resp.body {
        MemoryBody::Object(None) => {}
        other => panic!("Expected Object(None), got {:?}", other),
    }
}

// ── Forget command ──

#[async_std::test]
async fn test_worker_forget() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let obj = make_object("fg1", "forgettable", MemoryObjectType::Fact, "temp");
    send_recv(&mut tx, &mut rx, "s1", MemoryCommand::Store(obj)).await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "r1",
        MemoryCommand::Forget { object_id: "fg1".to_string() },
    )
    .await;
    match resp.body {
        MemoryBody::Ack => {}
        other => panic!("Expected Ack, got {:?}", other),
    }

    let get_resp = send_recv(
        &mut tx,
        &mut rx,
        "r2",
        MemoryCommand::Get { object_id: "fg1".to_string() },
    )
    .await;
    match get_resp.body {
        MemoryBody::Object(None) => {}
        other => panic!("Expected Object(None) after forget, got {:?}", other),
    }
}

// ── ListByType command ──

#[async_std::test]
async fn test_worker_list_by_type() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let one_shot_type = MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
        LogicElementSubtype::ActiveQuery,
    ));

    send_recv(
        &mut tx,
        &mut rx,
        "s1",
        MemoryCommand::Store(make_object("lbt1", "query1", one_shot_type.clone(), "SELECT 1")),
    )
    .await;
    send_recv(
        &mut tx,
        &mut rx,
        "s2",
        MemoryCommand::Store(make_object("lbt2", "fact1", MemoryObjectType::Fact, "a fact")),
    )
    .await;
    send_recv(
        &mut tx,
        &mut rx,
        "s3",
        MemoryCommand::Store(make_object("lbt3", "query2", one_shot_type.clone(), "SELECT 2")),
    )
    .await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "r1",
        MemoryCommand::ListByType {
            object_type: one_shot_type,
        },
    )
    .await;
    match resp.body {
        MemoryBody::Recalled(results) => {
            assert_eq!(results.len(), 2);
            assert!(results.iter().all(|r| r.id == "lbt1" || r.id == "lbt3"));
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

// ── Ship command (PFC → Neocortex) ──

#[async_std::test]
async fn test_worker_ship_to_neocortex() {
    let neocortex = Box::new(InMemoryMemoryStore::new());
    // Keep a reference to verify neocortex content after ship
    // We need a shared reference — use Arc trick via clone
    // Ship, then verify PFC is empty (shipped objects removed)
    let (mut tx, mut rx, _handle) = spawn_worker(Some(neocortex));

    // Store 3 objects in PFC
    send_recv(
        &mut tx,
        &mut rx,
        "s1",
        MemoryCommand::Store(make_object("sh1", "item1", MemoryObjectType::Fact, "fact 1")),
    )
    .await;
    send_recv(
        &mut tx,
        &mut rx,
        "s2",
        MemoryCommand::Store(make_object("sh2", "item2", MemoryObjectType::Observation, "obs 1")),
    )
    .await;
    send_recv(
        &mut tx,
        &mut rx,
        "s3",
        MemoryCommand::Store(make_object("sh3", "item3", MemoryObjectType::Preference, "pref 1")),
    )
    .await;

    // Verify all 3 exist in PFC before ship
    let get1 = send_recv(&mut tx, &mut rx, "g1", MemoryCommand::Get { object_id: "sh1".to_string() }).await;
    assert!(matches!(get1.body, MemoryBody::Object(Some(_))));

    // Ship!
    let ship_resp = send_recv(&mut tx, &mut rx, "ship1", MemoryCommand::Ship).await;
    match ship_resp.body {
        MemoryBody::Ack => {}
        other => panic!("Expected Ack from Ship, got {:?}", other),
    }

    // After ship, PFC should be empty
    let get_after = send_recv(&mut tx, &mut rx, "g2", MemoryCommand::Get { object_id: "sh1".to_string() }).await;
    match get_after.body {
        MemoryBody::Object(None) => {}
        other => panic!("After Ship, PFC should be empty. Got {:?}", other),
    }

    let get_after2 = send_recv(&mut tx, &mut rx, "g3", MemoryCommand::Get { object_id: "sh2".to_string() }).await;
    assert!(matches!(get_after2.body, MemoryBody::Object(None)));

    let get_after3 = send_recv(&mut tx, &mut rx, "g4", MemoryCommand::Get { object_id: "sh3".to_string() }).await;
    assert!(matches!(get_after3.body, MemoryBody::Object(None)));
}

// ── Ship without neocortex configured ──

#[async_std::test]
async fn test_worker_ship_without_neocortex_errors() {
    let (mut tx, mut rx, _handle) = spawn_worker(None); // No neocortex

    let resp = send_recv(&mut tx, &mut rx, "ship1", MemoryCommand::Ship).await;
    match resp.body {
        MemoryBody::Error(msg) => {
            assert!(msg.contains("neocortex"), "Error should mention neocortex: {}", msg);
        }
        other => panic!("Expected Error from Ship without neocortex, got {:?}", other),
    }
}

// ── Stop command ──

#[async_std::test]
async fn test_worker_stop() {
    let (mut tx, _rx, handle) = spawn_worker(None);

    tx.send(MemoryInput {
        id: Some("stop".to_string()),
        command: MemoryCommand::Stop,
    })
    .await
    .unwrap();

    // Worker should terminate within a reasonable time
    async_std::future::timeout(Duration::from_secs(2), handle)
        .await
        .expect("Worker should stop after Stop command");
}

// ── Request ID correlation ──

#[async_std::test]
async fn test_worker_request_id_preserved() {
    let (mut tx, mut rx, _handle) = spawn_worker(None);

    let obj = make_object("rid1", "corr", MemoryObjectType::Fact, "content");
    let resp = send_recv(&mut tx, &mut rx, "my-custom-id-42", MemoryCommand::Store(obj)).await;

    assert_eq!(resp.id, Some("my-custom-id-42".to_string()));
}
