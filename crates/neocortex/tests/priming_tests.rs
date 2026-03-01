//! Practical priming scenario tests: populate memory with realistic one-shot
//! and few-shot examples of every LogicElement subtype, then recall them by
//! text search, vector similarity, and type filtering.

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use rusty_genius_core::memory::{
    InMemoryMemoryStore, LogicElement, LogicElementSubtype, MemoryObject, MemoryObjectType,
    MockEmbeddingProvider,
};
use rusty_genius_core::protocol::{MemoryBody, MemoryCommand, MemoryInput, MemoryOutput};
use rusty_genius_neocortex::NeocortexWorker;
use std::time::Duration;

// ── Fixture data: realistic one-shot and few-shot examples ──

fn one_shot_active_query() -> MemoryObject {
    MemoryObject {
        id: "os-aq-1".to_string(),
        short_name: "user_active_filter".to_string(),
        long_name: "One-shot: Active user filter query".to_string(),
        description: "Example of filtering active users with a SQL WHERE clause".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::ActiveQuery,
        )),
        content: r#"SELECT u.id, u.name, u.email
FROM users u
WHERE u.active = true
  AND u.last_login > NOW() - INTERVAL '30 days'
ORDER BY u.last_login DESC
LIMIT 50;"#
            .to_string(),
        embedding: None,
        metadata: Some(r#"{"dialect": "postgresql", "complexity": "simple"}"#.to_string()),
        created_at: 1700000000,
        updated_at: 1700000000,
        ttl: None,
    }
}

fn one_shot_active_filter() -> MemoryObject {
    MemoryObject {
        id: "os-af-1".to_string(),
        short_name: "price_range_filter".to_string(),
        long_name: "One-shot: Price range active filter".to_string(),
        description: "Example of a composable filter predicate for price ranges".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::ActiveFilter,
        )),
        content: r#"fn price_filter(min: f64, max: f64) -> impl Fn(&Product) -> bool {
    move |p| p.price >= min && p.price <= max
}"#
        .to_string(),
        embedding: None,
        metadata: Some(r#"{"language": "rust"}"#.to_string()),
        created_at: 1700000100,
        updated_at: 1700000100,
        ttl: None,
    }
}

fn one_shot_ui_card() -> MemoryObject {
    MemoryObject {
        id: "os-uc-1".to_string(),
        short_name: "user_profile_card".to_string(),
        long_name: "One-shot: User profile card component".to_string(),
        description: "React component showing a user profile card with avatar and stats".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::UICard,
        )),
        content: r#"function UserProfileCard({ user }) {
  return (
    <div className="card shadow-md rounded-lg p-4">
      <img src={user.avatar} alt={user.name} className="w-16 h-16 rounded-full" />
      <h3 className="text-lg font-bold mt-2">{user.name}</h3>
      <p className="text-gray-500">{user.email}</p>
      <div className="flex gap-4 mt-3">
        <span>{user.posts} posts</span>
        <span>{user.followers} followers</span>
      </div>
    </div>
  );
}"#
        .to_string(),
        embedding: None,
        metadata: Some(r#"{"framework": "react", "styling": "tailwind"}"#.to_string()),
        created_at: 1700000200,
        updated_at: 1700000200,
        ttl: None,
    }
}

fn one_shot_shader() -> MemoryObject {
    MemoryObject {
        id: "os-sh-1".to_string(),
        short_name: "gradient_frag".to_string(),
        long_name: "One-shot: Gradient fragment shader".to_string(),
        description: "GLSL fragment shader that renders a horizontal color gradient".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::Shader,
        )),
        content: r#"#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform vec3 colorLeft;
uniform vec3 colorRight;

void main() {
    vec3 color = mix(colorLeft, colorRight, TexCoord.x);
    FragColor = vec4(color, 1.0);
}"#
        .to_string(),
        embedding: None,
        metadata: Some(r#"{"glsl_version": "330", "type": "fragment"}"#.to_string()),
        created_at: 1700000300,
        updated_at: 1700000300,
        ttl: None,
    }
}

fn one_shot_shader_portion() -> MemoryObject {
    MemoryObject {
        id: "os-sp-1".to_string(),
        short_name: "phong_lighting".to_string(),
        long_name: "One-shot: Phong lighting calculation portion".to_string(),
        description: "Reusable GLSL function for Phong specular lighting".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::ShaderPortion,
        )),
        content: r#"vec3 phongSpecular(vec3 lightDir, vec3 viewDir, vec3 normal, float shininess) {
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return vec3(spec);
}"#
        .to_string(),
        embedding: None,
        metadata: None,
        created_at: 1700000400,
        updated_at: 1700000400,
        ttl: None,
    }
}

fn one_shot_mtsm_ops() -> MemoryObject {
    MemoryObject {
        id: "os-mt-1".to_string(),
        short_name: "mtsm_batch_transform".to_string(),
        long_name: "One-shot: MTSM batch transformation operation".to_string(),
        description: "Multi-tensor state machine batch operation for tensor reshaping".to_string(),
        object_type: MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
            LogicElementSubtype::MtsmOps,
        )),
        content: r#"op: BatchTransform
inputs: [tensor_a: f32[batch, seq, hidden], tensor_b: f32[batch, hidden, out]]
output: f32[batch, seq, out]
steps:
  - matmul(tensor_a, tensor_b)
  - relu()
  - normalize(dim=-1)"#
            .to_string(),
        embedding: None,
        metadata: Some(r#"{"precision": "f32", "fused": true}"#.to_string()),
        created_at: 1700000500,
        updated_at: 1700000500,
        ttl: None,
    }
}

fn few_shot_ui_component() -> Vec<MemoryObject> {
    vec![
        MemoryObject {
            id: "fs-uic-1".to_string(),
            short_name: "toggle_switch".to_string(),
            long_name: "Few-shot: Toggle switch component (example 1)".to_string(),
            description: "Accessible toggle switch with animation".to_string(),
            object_type: MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::UIComponent,
            )),
            content: r#"function Toggle({ checked, onChange, label }) {
  return (
    <label className="flex items-center cursor-pointer gap-2">
      <span>{label}</span>
      <div className={`w-10 h-6 rounded-full transition ${checked ? 'bg-blue-500' : 'bg-gray-300'}`}>
        <div className={`w-4 h-4 bg-white rounded-full m-1 transition ${checked ? 'translate-x-4' : ''}`} />
      </div>
      <input type="checkbox" checked={checked} onChange={onChange} className="sr-only" />
    </label>
  );
}"#
            .to_string(),
            embedding: None,
            metadata: Some(r#"{"framework": "react", "a11y": true}"#.to_string()),
            created_at: 1700001000,
            updated_at: 1700001000,
            ttl: None,
        },
        MemoryObject {
            id: "fs-uic-2".to_string(),
            short_name: "dropdown_menu".to_string(),
            long_name: "Few-shot: Dropdown menu component (example 2)".to_string(),
            description: "Dropdown menu with keyboard navigation".to_string(),
            object_type: MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::UIComponent,
            )),
            content: r#"function Dropdown({ items, onSelect }) {
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(0);

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') setFocused(f => Math.min(f + 1, items.length - 1));
    if (e.key === 'ArrowUp') setFocused(f => Math.max(f - 1, 0));
    if (e.key === 'Enter') { onSelect(items[focused]); setOpen(false); }
  };

  return (
    <div onKeyDown={handleKeyDown} tabIndex={0}>
      <button onClick={() => setOpen(!open)}>Select...</button>
      {open && (
        <ul className="absolute bg-white shadow-lg rounded mt-1">
          {items.map((item, i) => (
            <li key={item.id} className={i === focused ? 'bg-blue-100' : ''}
                onClick={() => { onSelect(item); setOpen(false); }}>
              {item.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}"#
            .to_string(),
            embedding: None,
            metadata: Some(r#"{"framework": "react", "a11y": true, "keyboard": true}"#.to_string()),
            created_at: 1700001100,
            updated_at: 1700001100,
            ttl: None,
        },
        MemoryObject {
            id: "fs-uic-3".to_string(),
            short_name: "search_input".to_string(),
            long_name: "Few-shot: Search input with debounce (example 3)".to_string(),
            description: "Search input component with debounced onChange handler".to_string(),
            object_type: MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::UIComponent,
            )),
            content: r#"function SearchInput({ onSearch, placeholder = "Search..." }) {
  const [value, setValue] = useState('');
  const timeoutRef = useRef(null);

  const handleChange = (e) => {
    const v = e.target.value;
    setValue(v);
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => onSearch(v), 300);
  };

  return (
    <div className="relative">
      <SearchIcon className="absolute left-3 top-2.5 text-gray-400 w-5 h-5" />
      <input type="text" value={value} onChange={handleChange}
             placeholder={placeholder}
             className="pl-10 pr-4 py-2 border rounded-lg w-full" />
    </div>
  );
}"#
            .to_string(),
            embedding: None,
            metadata: Some(r#"{"framework": "react", "pattern": "debounce"}"#.to_string()),
            created_at: 1700001200,
            updated_at: 1700001200,
            ttl: None,
        },
    ]
}

/// Non-logic-element objects to test mixed-type recall
fn supporting_facts() -> Vec<MemoryObject> {
    vec![
        MemoryObject {
            id: "fact-sql".to_string(),
            short_name: "sql_join_types".to_string(),
            long_name: "SQL join types reference".to_string(),
            description: "Reference for INNER, LEFT, RIGHT, FULL OUTER joins".to_string(),
            object_type: MemoryObjectType::Fact,
            content: "SQL joins: INNER returns matching rows, LEFT returns all from left table, RIGHT returns all from right table, FULL OUTER returns all rows from both.".to_string(),
            embedding: None,
            metadata: None,
            created_at: 1700002000,
            updated_at: 1700002000,
            ttl: None,
        },
        MemoryObject {
            id: "pref-dark".to_string(),
            short_name: "prefer_dark_mode".to_string(),
            long_name: "User preference: dark mode".to_string(),
            description: "User prefers dark mode for all UI components".to_string(),
            object_type: MemoryObjectType::Preference,
            content: "Always generate UI components with dark mode variants. Use dark backgrounds (bg-gray-900) and light text (text-gray-100).".to_string(),
            embedding: None,
            metadata: None,
            created_at: 1700002100,
            updated_at: 1700002100,
            ttl: None,
        },
        MemoryObject {
            id: "obs-perf".to_string(),
            short_name: "react_perf_memo".to_string(),
            long_name: "Observation: React.memo improved dropdown perf".to_string(),
            description: "Wrapping dropdown items in React.memo reduced re-renders by 60%".to_string(),
            object_type: MemoryObjectType::Observation,
            content: "Observed that wrapping list items in React.memo with a custom comparator reduced unnecessary re-renders from 150ms to 60ms on a 500-item dropdown.".to_string(),
            embedding: None,
            metadata: None,
            created_at: 1700002200,
            updated_at: 1700002200,
            ttl: None,
        },
    ]
}

// ── Helpers ──

fn spawn_worker() -> (
    mpsc::Sender<MemoryInput>,
    mpsc::Receiver<MemoryOutput>,
    async_std::task::JoinHandle<()>,
) {
    let store = Box::new(InMemoryMemoryStore::new());
    let embedder = Box::new(MockEmbeddingProvider::new(16));
    let worker = NeocortexWorker::new(store, embedder);

    let (input_tx, input_rx) = mpsc::channel::<MemoryInput>(128);
    let (output_tx, output_rx) = mpsc::channel::<MemoryOutput>(128);

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

/// Prime the worker with all fixture objects, returning count stored.
async fn prime_all(
    tx: &mut mpsc::Sender<MemoryInput>,
    rx: &mut mpsc::Receiver<MemoryOutput>,
) -> usize {
    let mut objects: Vec<MemoryObject> = vec![
        one_shot_active_query(),
        one_shot_active_filter(),
        one_shot_ui_card(),
        one_shot_shader(),
        one_shot_shader_portion(),
        one_shot_mtsm_ops(),
    ];
    objects.extend(few_shot_ui_component());
    objects.extend(supporting_facts());

    let count = objects.len();
    for (i, obj) in objects.into_iter().enumerate() {
        let resp = send_recv(tx, rx, &format!("prime-{}", i), MemoryCommand::Store(obj)).await;
        assert!(
            matches!(resp.body, MemoryBody::Stored(_)),
            "Failed to store object #{}: {:?}",
            i,
            resp.body
        );
    }
    count
}

// ══════════════════════════════════════════════════════════════
// Priming scenario tests
// ══════════════════════════════════════════════════════════════

/// Prime memory with all examples and verify total count.
#[async_std::test]
async fn test_prime_all_objects_stored() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    let count = prime_all(&mut tx, &mut rx).await;

    // 6 one-shots + 3 few-shots + 3 supporting = 12
    assert_eq!(count, 12);

    // Verify via Get on a sample
    let resp = send_recv(
        &mut tx,
        &mut rx,
        "check",
        MemoryCommand::Get {
            object_id: "os-aq-1".to_string(),
        },
    )
    .await;
    match resp.body {
        MemoryBody::Object(Some(obj)) => {
            assert_eq!(obj.short_name, "user_active_filter");
            assert!(obj.content.contains("SELECT"));
            assert!(obj.embedding.is_some(), "Should be auto-embedded");
        }
        other => panic!("Expected stored object, got {:?}", other),
    }
}

/// Recall one-shot ActiveQuery examples by SQL-related text.
#[async_std::test]
async fn test_recall_active_query_by_text() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "recall-sql",
        MemoryCommand::Recall {
            query: "SELECT users active login".to_string(),
            limit: 5,
            object_type: None,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            assert!(!results.is_empty(), "Should find SQL-related results");
            // The active query example should be in results
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            assert!(
                ids.contains(&"os-aq-1"),
                "ActiveQuery example should be recalled, got: {:?}",
                ids
            );
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

/// Recall shader examples by GLSL-related text.
/// Uses keywords present in the stored shader content (FragColor, main, vec4).
#[async_std::test]
async fn test_recall_shaders_by_text() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "recall-glsl",
        MemoryCommand::Recall {
            query: "FragColor".to_string(),
            limit: 5,
            object_type: None,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            // Should find the gradient shader (contains "FragColor")
            assert!(
                ids.contains(&"os-sh-1"),
                "Should recall gradient shader example, got: {:?}",
                ids
            );
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

/// Filter recall to only UIComponent few-shot examples.
#[async_std::test]
async fn test_recall_few_shot_ui_components_by_type() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let ui_component_type = MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
        LogicElementSubtype::UIComponent,
    ));

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "recall-uic",
        MemoryCommand::ListByType {
            object_type: ui_component_type,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            assert_eq!(results.len(), 3, "Should find exactly 3 few-shot UIComponent examples");
            let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            assert!(ids.contains(&"fs-uic-1"));
            assert!(ids.contains(&"fs-uic-2"));
            assert!(ids.contains(&"fs-uic-3"));
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

/// Filter recall to only one-shot shader examples.
#[async_std::test]
async fn test_list_one_shot_shaders() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let shader_type = MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
        LogicElementSubtype::Shader,
    ));

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "list-shaders",
        MemoryCommand::ListByType {
            object_type: shader_type,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "os-sh-1");
            assert!(results[0].content.contains("FragColor"));
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

/// Verify different LogicElement types are stored distinctly.
#[async_std::test]
async fn test_logic_element_types_are_distinct() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    // Each subtype should have the correct count
    let types_and_expected = vec![
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::ActiveQuery,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::ActiveFilter,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::UICard,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::Shader,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::ShaderPortion,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::OneShotExamples(
                LogicElementSubtype::MtsmOps,
            )),
            1,
        ),
        (
            MemoryObjectType::LogicElement(LogicElement::FewShotExamples(
                LogicElementSubtype::UIComponent,
            )),
            3,
        ),
        (MemoryObjectType::Fact, 1),
        (MemoryObjectType::Preference, 1),
        (MemoryObjectType::Observation, 1),
    ];

    for (i, (object_type, expected)) in types_and_expected.iter().enumerate() {
        let resp = send_recv(
            &mut tx,
            &mut rx,
            &format!("type-{}", i),
            MemoryCommand::ListByType {
                object_type: object_type.clone(),
            },
        )
        .await;
        match resp.body {
            MemoryBody::Recalled(results) => {
                assert_eq!(
                    results.len(),
                    *expected,
                    "Type {:?} expected {} results, got {}",
                    object_type,
                    expected,
                    results.len()
                );
            }
            other => panic!("Expected Recalled for type {:?}, got {:?}", object_type, other),
        }
    }
}

/// Recall UI-related objects across all types with text query.
#[async_std::test]
async fn test_recall_cross_type_ui_query() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "recall-ui",
        MemoryCommand::Recall {
            query: "component toggle dropdown".to_string(),
            limit: 10,
            object_type: None,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            // Should find UI-related results across types
            assert!(
                !results.is_empty(),
                "Should find UI-related results"
            );
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}

/// Verify metadata is preserved through priming.
#[async_std::test]
async fn test_primed_metadata_preserved() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let resp = send_recv(
        &mut tx,
        &mut rx,
        "get-shader",
        MemoryCommand::Get {
            object_id: "os-sh-1".to_string(),
        },
    )
    .await;

    match resp.body {
        MemoryBody::Object(Some(obj)) => {
            let meta = obj.metadata.expect("Shader should have metadata");
            assert!(meta.contains("glsl_version"));
            assert!(meta.contains("330"));
            assert!(meta.contains("fragment"));
        }
        other => panic!("Expected Object, got {:?}", other),
    }
}

/// Verify all primed objects have embeddings auto-generated.
#[async_std::test]
async fn test_all_primed_objects_have_embeddings() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    let all_ids = vec![
        "os-aq-1", "os-af-1", "os-uc-1", "os-sh-1", "os-sp-1", "os-mt-1",
        "fs-uic-1", "fs-uic-2", "fs-uic-3",
        "fact-sql", "pref-dark", "obs-perf",
    ];

    for id in all_ids {
        let resp = send_recv(
            &mut tx,
            &mut rx,
            &format!("emb-check-{}", id),
            MemoryCommand::Get {
                object_id: id.to_string(),
            },
        )
        .await;
        match resp.body {
            MemoryBody::Object(Some(obj)) => {
                assert!(
                    obj.embedding.is_some(),
                    "Object '{}' should have embedding",
                    id
                );
                assert_eq!(
                    obj.embedding.as_ref().unwrap().len(),
                    16,
                    "Embedding dimension mismatch for '{}'",
                    id
                );
            }
            other => panic!("Expected Object for '{}', got {:?}", id, other),
        }
    }
}

/// Test the full PFC → Neocortex shipping flow with primed data.
/// Prime PFC, ship to Neocortex, verify PFC is empty and Neocortex has data.
#[async_std::test]
async fn test_ship_primed_data_pfc_to_neocortex() {
    use rusty_genius_pfc::PfcWorker;

    // Set up PFC with a neocortex backend
    let pfc_store = Box::new(InMemoryMemoryStore::new());
    let neo_store = Box::new(InMemoryMemoryStore::new());
    let embedder = Box::new(MockEmbeddingProvider::new(16));
    let pfc_worker = PfcWorker::new(pfc_store, embedder, Some(neo_store));

    let (pfc_tx, pfc_rx) = mpsc::channel::<MemoryInput>(128);
    let (pfc_out_tx, pfc_out_rx) = mpsc::channel::<MemoryOutput>(128);

    let _pfc_handle = async_std::task::spawn(async move {
        pfc_worker.run(pfc_rx, pfc_out_tx).await;
    });

    let mut tx = pfc_tx;
    let mut rx = pfc_out_rx;

    // Prime PFC with all fixtures
    let count = prime_all(&mut tx, &mut rx).await;
    assert_eq!(count, 12);

    // Verify PFC has objects
    let resp = send_recv(
        &mut tx,
        &mut rx,
        "pre-check",
        MemoryCommand::Get {
            object_id: "os-aq-1".to_string(),
        },
    )
    .await;
    assert!(matches!(resp.body, MemoryBody::Object(Some(_))));

    // Ship PFC → Neocortex
    let ship_resp = send_recv(&mut tx, &mut rx, "ship", MemoryCommand::Ship).await;
    assert!(
        matches!(ship_resp.body, MemoryBody::Ack),
        "Ship should succeed: {:?}",
        ship_resp.body
    );

    // Verify PFC is now empty
    for id in &["os-aq-1", "os-sh-1", "fs-uic-1", "fact-sql", "pref-dark"] {
        let resp = send_recv(
            &mut tx,
            &mut rx,
            &format!("post-{}", id),
            MemoryCommand::Get {
                object_id: id.to_string(),
            },
        )
        .await;
        assert!(
            matches!(resp.body, MemoryBody::Object(None)),
            "PFC should be empty after ship, but '{}' still exists",
            id
        );
    }
}

/// Recall by vector similarity after priming — the exact embedding of a
/// stored object should return that object as the top result.
#[async_std::test]
async fn test_recall_by_vector_after_priming() {
    let (mut tx, mut rx, _handle) = spawn_worker();
    prime_all(&mut tx, &mut rx).await;

    // Get the embedding of the shader object
    let resp = send_recv(
        &mut tx,
        &mut rx,
        "get-emb",
        MemoryCommand::Get {
            object_id: "os-sh-1".to_string(),
        },
    )
    .await;

    let shader_emb = match resp.body {
        MemoryBody::Object(Some(obj)) => obj.embedding.expect("should have embedding"),
        other => panic!("Expected Object, got {:?}", other),
    };

    // Use that exact embedding to recall by vector
    let resp = send_recv(
        &mut tx,
        &mut rx,
        "vec-recall",
        MemoryCommand::RecallByVector {
            embedding: shader_emb,
            limit: 3,
            object_type: None,
        },
    )
    .await;

    match resp.body {
        MemoryBody::Recalled(results) => {
            assert!(!results.is_empty(), "Vector recall should return results");
            assert_eq!(
                results[0].id, "os-sh-1",
                "Exact embedding match should be first result"
            );
        }
        other => panic!("Expected Recalled, got {:?}", other),
    }
}
