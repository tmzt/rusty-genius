use rusty_genius_core::memory::{
    LogicElement, LogicElementSubtype, MemoryObject, MemoryObjectType, MemoryStore,
    MockEmbeddingProvider,
};
use rusty_genius_neocortex::SqliteMemoryStore;

/// Create a fresh in-memory SQLite store for each test.
async fn fresh_store() -> SqliteMemoryStore {
    SqliteMemoryStore::new("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite store")
}

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
        long_name: format!("{} (detailed)", short_name),
        description: format!("Description for {}", short_name),
        object_type,
        content: content.to_string(),
        embedding,
        metadata: None,
        created_at: 1700000000,
        updated_at: 1700000000,
        ttl: None,
    }
}

// ── Basic CRUD ──

#[async_std::test]
async fn test_store_and_get() {
    let store = fresh_store().await;
    let obj = make_object("obj1", "test", MemoryObjectType::Fact, "Hello world", None);

    let id = store.store(obj).await.unwrap();
    assert_eq!(id, "obj1");

    let retrieved = store.get("obj1").await.unwrap();
    assert!(retrieved.is_some());
    let r = retrieved.unwrap();
    assert_eq!(r.id, "obj1");
    assert_eq!(r.content, "Hello world");
    assert_eq!(r.short_name, "test");
}

#[async_std::test]
async fn test_get_nonexistent() {
    let store = fresh_store().await;
    assert!(store.get("no-such-id").await.unwrap().is_none());
}

#[async_std::test]
async fn test_store_with_embedding_roundtrip() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);
    let emb = embedder.embed_sync("test content");

    let obj = make_object(
        "emb1",
        "with_embedding",
        MemoryObjectType::Fact,
        "test content",
        Some(emb.clone()),
    );
    store.store(obj).await.unwrap();

    let retrieved = store.get("emb1").await.unwrap().unwrap();
    assert!(retrieved.embedding.is_some());
    let stored_emb = retrieved.embedding.unwrap();
    assert_eq!(stored_emb.len(), 16);
    // Verify values are close (floating point serialization roundtrip)
    for (a, b) in emb.iter().zip(stored_emb.iter()) {
        assert!((a - b).abs() < 0.0001, "Embedding mismatch: {} vs {}", a, b);
    }
}

#[async_std::test]
async fn test_forget() {
    let store = fresh_store().await;
    let obj = make_object("del1", "deleteme", MemoryObjectType::Fact, "ephemeral", None);
    store.store(obj).await.unwrap();
    assert!(store.get("del1").await.unwrap().is_some());

    store.forget("del1").await.unwrap();
    assert!(store.get("del1").await.unwrap().is_none());
}

#[async_std::test]
async fn test_forget_with_embedding() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);
    let emb = embedder.embed_sync("will be deleted");
    let obj = make_object(
        "del2",
        "deleteme",
        MemoryObjectType::Fact,
        "will be deleted",
        Some(emb),
    );
    store.store(obj).await.unwrap();
    store.forget("del2").await.unwrap();
    assert!(store.get("del2").await.unwrap().is_none());
}

#[async_std::test]
async fn test_store_replace() {
    let store = fresh_store().await;
    let obj1 = make_object("dup", "v1", MemoryObjectType::Fact, "first version", None);
    store.store(obj1).await.unwrap();

    let obj2 = make_object("dup", "v2", MemoryObjectType::Fact, "second version", None);
    store.store(obj2).await.unwrap();

    let retrieved = store.get("dup").await.unwrap().unwrap();
    assert_eq!(retrieved.short_name, "v2");
    assert_eq!(retrieved.content, "second version");
}

// ── Listing ──

#[async_std::test]
async fn test_list_all() {
    let store = fresh_store().await;
    store
        .store(make_object("a", "a", MemoryObjectType::Fact, "fact a", None))
        .await
        .unwrap();
    store
        .store(make_object(
            "b",
            "b",
            MemoryObjectType::Observation,
            "obs b",
            None,
        ))
        .await
        .unwrap();
    store
        .store(make_object(
            "c",
            "c",
            MemoryObjectType::Preference,
            "pref c",
            None,
        ))
        .await
        .unwrap();

    let all = store.list_all().await.unwrap();
    assert_eq!(all.len(), 3);
}

#[async_std::test]
async fn test_list_by_type() {
    let store = fresh_store().await;
    store
        .store(make_object("f1", "fact1", MemoryObjectType::Fact, "fact one", None))
        .await
        .unwrap();
    store
        .store(make_object(
            "o1",
            "obs1",
            MemoryObjectType::Observation,
            "obs one",
            None,
        ))
        .await
        .unwrap();
    store
        .store(make_object("f2", "fact2", MemoryObjectType::Fact, "fact two", None))
        .await
        .unwrap();

    let facts = store.list_by_type(&MemoryObjectType::Fact).await.unwrap();
    assert_eq!(facts.len(), 2);

    let observations = store
        .list_by_type(&MemoryObjectType::Observation)
        .await
        .unwrap();
    assert_eq!(observations.len(), 1);

    let prefs = store
        .list_by_type(&MemoryObjectType::Preference)
        .await
        .unwrap();
    assert_eq!(prefs.len(), 0);
}

#[async_std::test]
async fn test_list_by_logic_element_type() {
    let store = fresh_store().await;

    let one_shot_query = MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
        LogicElementSubtype::ActiveQuery,
    ));
    let few_shot_card = MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
        LogicElementSubtype::UICard,
    ));

    store
        .store(make_object("lq1", "query_example", one_shot_query.clone(), "SELECT 1", None))
        .await
        .unwrap();
    store
        .store(make_object("lc1", "card_example", few_shot_card.clone(), "<Card/>", None))
        .await
        .unwrap();
    store
        .store(make_object("lq2", "query_example2", one_shot_query.clone(), "SELECT 2", None))
        .await
        .unwrap();

    let queries = store.list_by_type(&one_shot_query).await.unwrap();
    assert_eq!(queries.len(), 2);

    let cards = store.list_by_type(&few_shot_card).await.unwrap();
    assert_eq!(cards.len(), 1);
}

// ── Flush ──

#[async_std::test]
async fn test_flush_all() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);

    store
        .store(make_object(
            "a",
            "a",
            MemoryObjectType::Fact,
            "a",
            Some(embedder.embed_sync("a")),
        ))
        .await
        .unwrap();
    store
        .store(make_object(
            "b",
            "b",
            MemoryObjectType::Fact,
            "b",
            Some(embedder.embed_sync("b")),
        ))
        .await
        .unwrap();

    store.flush_all().await.unwrap();

    assert_eq!(store.list_all().await.unwrap().len(), 0);
    assert!(store.get("a").await.unwrap().is_none());
}

// ── FTS5 text search via recall ──

#[async_std::test]
async fn test_recall_fts5_basic() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);

    // Store objects with text content
    store
        .store(make_object(
            "sql1",
            "users_query",
            MemoryObjectType::Fact,
            "SELECT id, name FROM users WHERE active = true",
            Some(embedder.embed_sync("SELECT id, name FROM users WHERE active = true")),
        ))
        .await
        .unwrap();

    store
        .store(make_object(
            "shader1",
            "frag_shader",
            MemoryObjectType::Fact,
            "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
            Some(embedder.embed_sync(
                "void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }",
            )),
        ))
        .await
        .unwrap();

    store
        .store(make_object(
            "sql2",
            "orders_query",
            MemoryObjectType::Fact,
            "SELECT order_id, total FROM orders WHERE status = 'pending'",
            Some(embedder.embed_sync(
                "SELECT order_id, total FROM orders WHERE status = 'pending'",
            )),
        ))
        .await
        .unwrap();

    // Search for "SELECT" — should find the two SQL objects
    let query_vec = embedder.embed_sync("SELECT");
    let results = store.recall("SELECT", &query_vec, 10, None).await.unwrap();
    assert!(results.len() >= 2, "Expected at least 2 SQL results, got {}", results.len());

    // Both SQL objects should be in the results
    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
    assert!(ids.contains(&"sql1"));
    assert!(ids.contains(&"sql2"));
}

#[async_std::test]
async fn test_recall_fts5_with_type_filter() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);

    store
        .store(make_object(
            "f1",
            "sql_fact",
            MemoryObjectType::Fact,
            "SQL is a query language",
            Some(embedder.embed_sync("SQL is a query language")),
        ))
        .await
        .unwrap();

    store
        .store(make_object(
            "o1",
            "sql_obs",
            MemoryObjectType::Observation,
            "SQL performance was slow today",
            Some(embedder.embed_sync("SQL performance was slow today")),
        ))
        .await
        .unwrap();

    let query_vec = embedder.embed_sync("SQL");
    let facts_only = store
        .recall("SQL", &query_vec, 10, Some(&MemoryObjectType::Fact))
        .await
        .unwrap();
    assert_eq!(facts_only.len(), 1);
    assert_eq!(facts_only[0].id, "f1");
}

// ── Vector similarity search ──

#[async_std::test]
async fn test_recall_by_vector_cosine() {
    let store = fresh_store().await;
    let embedder = MockEmbeddingProvider::new(16);

    // Store three objects with embeddings
    let content_a = "Machine learning model training";
    let content_b = "Deep neural network architecture";
    let content_c = "Cooking pasta with tomato sauce";

    store
        .store(make_object(
            "ml",
            "ml_training",
            MemoryObjectType::Fact,
            content_a,
            Some(embedder.embed_sync(content_a)),
        ))
        .await
        .unwrap();
    store
        .store(make_object(
            "nn",
            "neural_nets",
            MemoryObjectType::Fact,
            content_b,
            Some(embedder.embed_sync(content_b)),
        ))
        .await
        .unwrap();
    store
        .store(make_object(
            "cook",
            "cooking",
            MemoryObjectType::Fact,
            content_c,
            Some(embedder.embed_sync(content_c)),
        ))
        .await
        .unwrap();

    // Query with the exact embedding of content_a — should return it first
    let query_vec = embedder.embed_sync(content_a);
    let results = store.recall_by_vector(&query_vec, 3, None).await.unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].id, "ml", "Exact match should be first result");
}

// ── Metadata preservation ──

#[async_std::test]
async fn test_metadata_roundtrip() {
    let store = fresh_store().await;
    let mut obj = make_object("meta1", "with_meta", MemoryObjectType::Fact, "content", None);
    obj.metadata = Some(r#"{"source": "test", "version": 2}"#.to_string());

    store.store(obj).await.unwrap();
    let retrieved = store.get("meta1").await.unwrap().unwrap();
    assert_eq!(
        retrieved.metadata,
        Some(r#"{"source": "test", "version": 2}"#.to_string())
    );
}

// ── Timestamps ──

#[async_std::test]
async fn test_timestamps_preserved() {
    let store = fresh_store().await;
    let mut obj = make_object("ts1", "timed", MemoryObjectType::Fact, "content", None);
    obj.created_at = 1700000000;
    obj.updated_at = 1700000999;

    store.store(obj).await.unwrap();
    let retrieved = store.get("ts1").await.unwrap().unwrap();
    assert_eq!(retrieved.created_at, 1700000000);
    assert_eq!(retrieved.updated_at, 1700000999);
}
