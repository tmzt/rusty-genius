use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use rusty_genius_core::memory::{MemoryObject, MemoryObjectType, MemoryStore};
use rusty_genius_hippocampus::store::IdbMemoryStore;

fn make_object(
    id: &str,
    short_name: &str,
    object_type: MemoryObjectType,
    content: &str,
    embedding: Option<Vec<f32>>,
) -> MemoryObject {
    MemoryObject {
        id: id.to_string(),
        short_name: short_name.to_string(),
        long_name: format!("{} (full)", short_name),
        description: format!("Test object: {}", short_name),
        object_type,
        content: content.to_string(),
        embedding,
        metadata: None,
        created_at: 1000,
        updated_at: 1000,
        ttl: None,
    }
}

fn mock_embedding(text: &str) -> Vec<f32> {
    // Simple deterministic mock embedding
    let mut vec = vec![0.0f32; 8];
    let bytes = text.as_bytes();
    let mut hash: u64 = 5381;
    for (i, slot) in vec.iter_mut().enumerate() {
        for (j, &b) in bytes.iter().enumerate() {
            hash = hash.wrapping_mul(33).wrapping_add(b as u64);
            hash = hash.wrapping_add((i as u64).wrapping_mul(7));
            hash = hash.wrapping_add((j as u64).wrapping_mul(13));
        }
        *slot = ((hash % 20000) as f32 / 10000.0) - 1.0;
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for slot in vec.iter_mut() {
            *slot /= norm;
        }
    }
    vec
}

#[wasm_bindgen_test]
async fn test_store_and_get_roundtrip() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    let obj = make_object("rt-1", "roundtrip", MemoryObjectType::Fact, "some content", None);
    let id = store.store(obj).await.expect("store");
    assert_eq!(id, "rt-1");

    let retrieved = store.get("rt-1").await.expect("get");
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.short_name, "roundtrip");
    assert_eq!(retrieved.content, "some content");

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_fts5_recall() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    let emb1 = mock_embedding("SELECT * FROM users");
    let obj1 = make_object(
        "fts-1",
        "sql_query",
        MemoryObjectType::Fact,
        "SELECT * FROM users WHERE active = true",
        Some(emb1.clone()),
    );
    store.store(obj1).await.expect("store sql");

    let emb2 = mock_embedding("void main fragment shader");
    let obj2 = make_object(
        "fts-2",
        "shader_code",
        MemoryObjectType::Fact,
        "void main() { gl_FragColor = vec4(1.0); }",
        Some(emb2),
    );
    store.store(obj2).await.expect("store shader");

    // FTS search for "SELECT" should find the SQL object
    let results = store.recall("SELECT", &emb1, 10, None).await.expect("recall");
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "fts-1");

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_vector_recall() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    let emb = mock_embedding("neural networks");
    let obj = make_object(
        "vec-1",
        "nn_topic",
        MemoryObjectType::Fact,
        "neural network architecture",
        Some(emb.clone()),
    );
    store.store(obj).await.expect("store");

    let results = store.recall_by_vector(&emb, 10, None).await.expect("recall_by_vector");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "vec-1");

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_forget() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    let obj = make_object("fg-1", "forget_me", MemoryObjectType::Fact, "temp data", None);
    store.store(obj).await.expect("store");

    store.forget("fg-1").await.expect("forget");
    let retrieved = store.get("fg-1").await.expect("get");
    assert!(retrieved.is_none());

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_list_all() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    store
        .store(make_object("la-1", "obj_a", MemoryObjectType::Fact, "fact a", None))
        .await
        .expect("store a");
    store
        .store(make_object("la-2", "obj_b", MemoryObjectType::Observation, "obs b", None))
        .await
        .expect("store b");

    let all = store.list_all().await.expect("list_all");
    assert_eq!(all.len(), 2);

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_list_by_type() {
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    store
        .store(make_object("lt-1", "fact_a", MemoryObjectType::Fact, "fact", None))
        .await
        .expect("store");
    store
        .store(make_object("lt-2", "obs_b", MemoryObjectType::Observation, "obs", None))
        .await
        .expect("store");
    store
        .store(make_object("lt-3", "fact_c", MemoryObjectType::Fact, "another fact", None))
        .await
        .expect("store");

    let facts = store.list_by_type(&MemoryObjectType::Fact).await.expect("list_by_type");
    assert_eq!(facts.len(), 2);

    store.flush_all().await.expect("cleanup");
}

#[wasm_bindgen_test]
async fn test_flush_all() {
    let store = IdbMemoryStore::open().await.expect("open store");

    store
        .store(make_object("fl-1", "a", MemoryObjectType::Fact, "a", None))
        .await
        .expect("store");
    store
        .store(make_object("fl-2", "b", MemoryObjectType::Fact, "b", None))
        .await
        .expect("store");

    store.flush_all().await.expect("flush_all");
    let all = store.list_all().await.expect("list_all");
    assert_eq!(all.len(), 0);
}

#[wasm_bindgen_test]
async fn test_persistence_across_reopens() {
    // Store an object
    let store = IdbMemoryStore::open().await.expect("open store");
    store.flush_all().await.expect("flush");

    let obj = make_object("persist-1", "persistent", MemoryObjectType::Fact, "I persist", None);
    store.store(obj).await.expect("store");
    drop(store);

    // Reopen and verify it's still there
    let store2 = IdbMemoryStore::open().await.expect("reopen store");
    let retrieved = store2.get("persist-1").await.expect("get");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().content, "I persist");

    store2.flush_all().await.expect("cleanup");
}
