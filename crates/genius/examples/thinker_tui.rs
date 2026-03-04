//! TUI example for rusty-genius with tool-use memory integration.
//!
//! Two screens toggled by Ctrl-M:
//!   - Chat screen: send messages, see tool calls and results
//!   - Memory browser: view/delete stored memories
//!
//! Run (macOS with Metal):
//!   cargo run -p rusty-genius --features metal --example thinker_tui -- [MODEL]
//!
//! Run (Linux with CUDA):
//!   cargo run -p rusty-genius --features cuda --example thinker_tui -- [MODEL]
//!
//! Run (CPU only):
//!   cargo run -p rusty-genius --features real-engine --example thinker_tui -- [MODEL]
//!
//! MODEL defaults to "qwen-2.5-7b-instruct" (from facecrab registry).

#[cfg(not(feature = "real-engine"))]
compile_error!(
    "thinker_tui requires a real inference engine. Build with one of:\n  \
     --features metal       (macOS, GPU)\n  \
     --features cuda        (Linux, GPU)\n  \
     --features vulkan      (cross-platform, GPU)\n  \
     --features real-engine (CPU only)"
);

use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use futures::channel::mpsc as fmpsc;
use futures::sink::SinkExt;
use futures::StreamExt;
use ratatui::prelude::*;
use ratatui::widgets::*;

use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::memory::{InMemoryMemoryStore, MemoryObject, MemoryObjectType, MockEmbeddingProvider};
use rusty_genius_core::protocol::{
    AssetEvent, BrainstemBody, BrainstemCommand, BrainstemInput, BrainstemOutput, ChatContent,
    ChatMessage, ChatRole, InferenceEvent,
};
use rusty_genius_core::tools::ToolExecutor;
use rusty_genius_stem::{MemoryToolExecutor, Orchestrator};

// ── App types ──

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Chat,
    MemoryBrowser,
}

#[derive(Clone)]
enum EntryKind {
    Message,
    Thought,
    ToolCall,
    ToolResult,
}

#[derive(Clone, Copy, PartialEq)]
enum EntryRole {
    User,
    Assistant,
    System,
}

#[derive(Clone)]
struct ChatEntry {
    role: EntryRole,
    content: String,
    kind: EntryKind,
}

struct App {
    screen: Screen,
    should_quit: bool,
    chat_log: Vec<ChatEntry>,
    input: String,
    chat_scroll: u16,
    // Memory store for the browser
    store: Arc<InMemoryMemoryStore>,
    memory_items: Vec<MemoryObject>,
    memory_list_state: ListState,
    // Orchestrator channels
    inference_tx: fmpsc::Sender<BrainstemInput>,
    // Sync receiver for main loop
    sync_rx: std::sync::mpsc::Receiver<BrainstemOutput>,
    is_inferring: bool,
    thinking_since: Option<Instant>,
    show_debug: bool,
    // Tool definitions for InferWithTools
    tool_defs: Vec<rusty_genius_core::protocol::ToolDefinition>,
    // Conversation messages for multi-turn
    conversation: Vec<ChatMessage>,
    // Model loading state
    model_status: ModelStatus,
}

#[derive(Clone)]
enum ModelStatus {
    NotLoaded,
    Downloading { name: String, current: u64, total: u64 },
    Loading(String),
    Ready(String),
    Error(String),
}

impl App {
    fn scroll_to_bottom(&mut self, visible_height: u16) {
        let total_lines: u16 = self
            .chat_log
            .iter()
            .map(|e| e.content.lines().count().max(1) as u16)
            .sum();
        if total_lines > visible_height {
            self.chat_scroll = total_lines.saturating_sub(visible_height);
        } else {
            self.chat_scroll = 0;
        }
    }
}

fn main() -> io::Result<()> {
    smol::block_on(async {
        run_app().await
    })
}

async fn run_app() -> io::Result<()> {
    // Set up memory store + embedder
    let store = Arc::new(InMemoryMemoryStore::new());
    let embedder = Arc::new(MockEmbeddingProvider::new(384));

    // Seed data
    seed_memories(&store, &embedder).await;

    // Create tool executor
    let mem_executor = MemoryToolExecutor::new(store.clone(), embedder.clone());
    let tool_defs = mem_executor.tool_definitions();

    // Create orchestrator with Pinky engine + tool executor
    let engine = rusty_genius_cortex::create_engine().await;
    let orchestrator = Orchestrator::with_engine(engine)
        .with_tool_executor(Box::new(mem_executor));

    // Channels
    let (input_tx, input_rx) = fmpsc::channel(100);
    let (output_tx, mut output_rx) = fmpsc::channel(100);

    // Spawn orchestrator
    smol::spawn(async move {
        let mut orch = orchestrator;
        if let Err(e) = orch.run(input_rx, output_tx).await {
            eprintln!("Orchestrator error: {}", e);
        }
    })
    .detach();

    // Bridge: async output_rx → sync std::sync::mpsc
    let (sync_tx, sync_rx) = std::sync::mpsc::channel();
    smol::spawn(async move {
        while let Some(msg) = output_rx.next().await {
            if sync_tx.send(msg).is_err() {
                break;
            }
        }
    })
    .detach();

    // Terminal setup
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App {
        screen: Screen::Chat,
        should_quit: false,
        chat_log: vec![ChatEntry {
            role: EntryRole::System,
            content: "Welcome to rusty-genius Thinker TUI! Type a message and press Enter.".to_string(),
            kind: EntryKind::Message,
        }],
        input: String::new(),
        chat_scroll: 0,
        store: store.clone(),
        memory_items: Vec::new(),
        memory_list_state: ListState::default(),
        inference_tx: input_tx,
        sync_rx,
        is_inferring: false,
        thinking_since: None,
        show_debug: false,
        tool_defs,
        conversation: Vec::new(),
        model_status: ModelStatus::NotLoaded,
    };

    // Send LoadModel to pre-load and trigger download if needed
    {
        let model_name = "qwen-2.5-7b-instruct".to_string();
        let mut tx = app.inference_tx.clone();
        smol::block_on(async {
            let _ = tx
                .send(BrainstemInput {
                    id: Some("model-load".to_string()),
                    command: BrainstemCommand::LoadModel(model_name.clone()),
                })
                .await;
        });
        app.model_status = ModelStatus::Loading("qwen-2.5-7b-instruct".to_string());
    }

    // Main loop
    while !app.should_quit {
        terminal.draw(|f| draw_ui(f, &mut app))?;

        // Drain orchestrator events (non-blocking)
        drain_events(&mut app);

        // Poll for keyboard events (50ms timeout)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                handle_key(&mut app, key);
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

async fn seed_memories(store: &Arc<InMemoryMemoryStore>, embedder: &Arc<MockEmbeddingProvider>) {
    use rusty_genius_core::memory::MemoryStore;

    let seeds = vec![
        ("mem-seed-1", "rust-ownership", MemoryObjectType::Fact,
         "Rust uses ownership and borrowing to guarantee memory safety without a garbage collector."),
        ("mem-seed-2", "pinky-brain", MemoryObjectType::Entity,
         "Pinky is the stub inference engine in rusty-genius, used for testing without a real LLM."),
        ("mem-seed-3", "pattern-match", MemoryObjectType::Skill,
         "Use Rust match expressions for exhaustive pattern matching on enums and values."),
    ];

    for (id, name, obj_type, content) in seeds {
        let embedding = embedder.embed_sync(content);
        let obj = MemoryObject {
            id: id.to_string(),
            short_name: name.to_string(),
            long_name: name.to_string(),
            description: format!("Seed memory: {}", name),
            object_type: obj_type,
            content: content.to_string(),
            embedding: Some(embedding),
            metadata: None,
            created_at: 1709443200,
            updated_at: 1709443200,
            ttl: None,
        };
        let _ = store.store(obj).await;
    }
}

fn drain_events(app: &mut App) {
    loop {
        match app.sync_rx.try_recv() {
            Ok(msg) => handle_brainstem_output(app, msg),
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.is_inferring = false;
                break;
            }
        }
    }
}

fn handle_brainstem_output(app: &mut App, msg: BrainstemOutput) {
    match msg.body {
        BrainstemBody::Event(event) => match event {
            InferenceEvent::ProcessStart => {}
            InferenceEvent::Thought(thought) => match thought {
                rusty_genius_core::protocol::ThoughtEvent::Start => {}
                rusty_genius_core::protocol::ThoughtEvent::Delta(text) => {
                    app.chat_log.push(ChatEntry {
                        role: EntryRole::Assistant,
                        content: text,
                        kind: EntryKind::Thought,
                    });
                }
                rusty_genius_core::protocol::ThoughtEvent::Stop => {}
            },
            InferenceEvent::Content(text) => {
                if text.starts_with("[tool_result:") {
                    app.chat_log.push(ChatEntry {
                        role: EntryRole::System,
                        content: text,
                        kind: EntryKind::ToolResult,
                    });
                } else {
                    app.chat_log.push(ChatEntry {
                        role: EntryRole::Assistant,
                        content: text,
                        kind: EntryKind::Message,
                    });
                }
            }
            InferenceEvent::ToolUse(calls) => {
                for call in &calls {
                    let args_str = serde_json::to_string(&call.arguments).unwrap_or_default();
                    app.chat_log.push(ChatEntry {
                        role: EntryRole::System,
                        content: format!("{}({})", call.name, args_str),
                        kind: EntryKind::ToolCall,
                    });
                }
            }
            InferenceEvent::Complete => {
                app.is_inferring = false;
                app.thinking_since = None;
            }
            InferenceEvent::Embedding(_) => {}
        },
        BrainstemBody::Asset(event) => match event {
            AssetEvent::Started(name) => {
                app.model_status = ModelStatus::Downloading {
                    name,
                    current: 0,
                    total: 0,
                };
            }
            AssetEvent::Progress(current, total) => {
                if let ModelStatus::Downloading { ref name, .. } = app.model_status {
                    app.model_status = ModelStatus::Downloading {
                        name: name.clone(),
                        current,
                        total,
                    };
                }
            }
            AssetEvent::Complete(path) => {
                let short = path.rsplit('/').next().unwrap_or(&path).to_string();
                app.model_status = ModelStatus::Ready(short);
            }
            AssetEvent::Error(e) => {
                app.model_status = ModelStatus::Error(e);
            }
        },
        BrainstemBody::Error(e) => {
            app.chat_log.push(ChatEntry {
                role: EntryRole::System,
                content: format!("Error: {}", e),
                kind: EntryKind::Message,
            });
            app.is_inferring = false;
            app.thinking_since = None;
        }
        _ => {}
    }
}

fn handle_key(app: &mut App, key: KeyEvent) {
    match app.screen {
        Screen::Chat => handle_chat_key(app, key),
        Screen::MemoryBrowser => handle_memory_key(app, key),
    }
}

fn handle_chat_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('m') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            // Refresh memory list and switch screen
            refresh_memory_list(app);
            app.screen = Screen::MemoryBrowser;
        }
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.show_debug = !app.show_debug;
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        KeyCode::Char('q') if app.input.is_empty() && !app.is_inferring => {
            app.should_quit = true;
        }
        KeyCode::Char(c) => {
            app.input.push(c);
        }
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Enter => {
            let model_ready = matches!(app.model_status, ModelStatus::Ready(_));
            if !app.input.is_empty() && !app.is_inferring && model_ready {
                send_message(app);
            }
        }
        KeyCode::Up => {
            app.chat_scroll = app.chat_scroll.saturating_sub(1);
        }
        KeyCode::Down => {
            app.chat_scroll = app.chat_scroll.saturating_add(1);
        }
        _ => {}
    }
}

fn handle_memory_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => {
            app.screen = Screen::Chat;
        }
        KeyCode::Char('q') => {
            app.screen = Screen::Chat;
        }
        KeyCode::Char('j') | KeyCode::Down => {
            let len = app.memory_items.len();
            if len > 0 {
                let i = app.memory_list_state.selected().unwrap_or(0);
                app.memory_list_state.select(Some((i + 1).min(len - 1)));
            }
        }
        KeyCode::Char('k') | KeyCode::Up => {
            let i = app.memory_list_state.selected().unwrap_or(0);
            app.memory_list_state.select(Some(i.saturating_sub(1)));
        }
        KeyCode::Char('x') => {
            // Delete selected memory
            if let Some(idx) = app.memory_list_state.selected() {
                if idx < app.memory_items.len() {
                    let id = app.memory_items[idx].id.clone();
                    let store = app.store.clone();
                    smol::block_on(async {
                        use rusty_genius_core::memory::MemoryStore;
                        let _ = store.forget(&id).await;
                    });
                    refresh_memory_list(app);
                    // Adjust selection
                    if !app.memory_items.is_empty() {
                        let new_idx = idx.min(app.memory_items.len() - 1);
                        app.memory_list_state.select(Some(new_idx));
                    } else {
                        app.memory_list_state.select(None);
                    }
                }
            }
        }
        _ => {}
    }
}

fn refresh_memory_list(app: &mut App) {
    let store = app.store.clone();
    app.memory_items = smol::block_on(async {
        use rusty_genius_core::memory::MemoryStore;
        store.list_all().await.unwrap_or_default()
    });
    // Sort by id for stable ordering
    app.memory_items.sort_by(|a, b| a.id.cmp(&b.id));
    if !app.memory_items.is_empty() && app.memory_list_state.selected().is_none() {
        app.memory_list_state.select(Some(0));
    }
}

fn send_message(app: &mut App) {
    let text = std::mem::take(&mut app.input);

    // Add to chat log
    app.chat_log.push(ChatEntry {
        role: EntryRole::User,
        content: text.clone(),
        kind: EntryKind::Message,
    });

    // Add to conversation
    app.conversation.push(ChatMessage {
        role: ChatRole::User,
        content: ChatContent::Text(text),
    });

    app.is_inferring = true;
    app.thinking_since = Some(Instant::now());

    // Send InferWithTools command
    let cmd = BrainstemCommand::InferWithTools {
        model: Some("qwen-2.5-7b-instruct".to_string()),
        messages: app.conversation.clone(),
        tools: app.tool_defs.clone(),
        config: InferenceConfig::default(),
    };

    let mut tx = app.inference_tx.clone();
    smol::block_on(async {
        let _ = tx
            .send(BrainstemInput {
                id: Some("tui-chat".to_string()),
                command: cmd,
            })
            .await;
    });
}

// ── Drawing ──

fn draw_ui(f: &mut Frame, app: &mut App) {
    match app.screen {
        Screen::Chat => draw_chat(f, app),
        Screen::MemoryBrowser => draw_memory_browser(f, app),
    }
}

fn draw_chat(f: &mut Frame, app: &mut App) {
    let area = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(5),    // Chat log
            Constraint::Length(3), // Input
            Constraint::Length(1), // Status bar
        ])
        .split(area);

    // Header
    let engine_name = if cfg!(feature = "real-engine") { "llama.cpp" } else { "pinky" };
    let header = Paragraph::new(format!(
        " rusty-genius  |  {}  |  ^M: Memory  ^D: Debug  ^C: Quit",
        engine_name,
    ))
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Chat log
    let chat_area = chunks[1];
    let inner_height = chat_area.height.saturating_sub(2); // borders

    // Auto-scroll to bottom
    app.scroll_to_bottom(inner_height);

    let show_debug = app.show_debug;
    let lines: Vec<Line> = app
        .chat_log
        .iter()
        .filter(|entry| {
            match entry.kind {
                EntryKind::Thought | EntryKind::ToolCall | EntryKind::ToolResult => show_debug,
                EntryKind::Message => true,
            }
        })
        .flat_map(|entry| {
            let (prefix, style) = match (&entry.role, &entry.kind) {
                (EntryRole::User, _) => (
                    "[User] ",
                    Style::default().fg(Color::Green),
                ),
                (EntryRole::Assistant, EntryKind::Thought) => (
                    "  [thought] ",
                    Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                ),
                (EntryRole::Assistant, EntryKind::Message) => (
                    "[Assistant] ",
                    Style::default().fg(Color::Yellow),
                ),
                (EntryRole::System, EntryKind::ToolCall) => (
                    "  [tool] ",
                    Style::default().fg(Color::Magenta),
                ),
                (EntryRole::System, EntryKind::ToolResult) => (
                    "  [result] ",
                    Style::default().fg(Color::Blue),
                ),
                (_, _) => (
                    "[System] ",
                    Style::default().fg(Color::Gray),
                ),
            };
            // Split content into lines for proper wrapping
            entry
                .content
                .lines()
                .enumerate()
                .map(|(i, line)| {
                    if i == 0 {
                        Line::from(Span::styled(format!("{}{}", prefix, line), style))
                    } else {
                        Line::from(Span::styled(
                            format!("{}{}", " ".repeat(prefix.len()), line),
                            style,
                        ))
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let chat_widget = Paragraph::new(lines)
        .block(
            Block::default()
                .title(" Chat ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        )
        .scroll((app.chat_scroll, 0))
        .wrap(Wrap { trim: false });
    f.render_widget(chat_widget, chat_area);

    // Input box
    let model_ready = matches!(app.model_status, ModelStatus::Ready(_));
    let input_display = if !model_ready {
        " (loading model...)".to_string()
    } else if app.is_inferring {
        " (waiting for response...)".to_string()
    } else {
        format!(" > {}", app.input)
    };
    let input_widget = Paragraph::new(input_display)
        .block(
            Block::default()
                .title(" Input [Enter: Send] ")
                .borders(Borders::ALL)
                .border_style(if app.is_inferring || !model_ready {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().fg(Color::Cyan)
                }),
        );
    f.render_widget(input_widget, chunks[2]);

    // Show cursor position in input box
    if !app.is_inferring && model_ready {
        f.set_cursor_position((
            chunks[2].x + app.input.len() as u16 + 4, // " > " + cursor
            chunks[2].y + 1,
        ));
    }

    // Status bar (bottom, like Claude Code)
    let model_part = match &app.model_status {
        ModelStatus::NotLoaded => "model: not loaded".to_string(),
        ModelStatus::Downloading { name, current, total } => {
            if *total > 0 {
                let pct = (*current as f64 / *total as f64) * 100.0;
                let mb_cur = *current as f64 / 1_048_576.0;
                let mb_tot = *total as f64 / 1_048_576.0;
                format!("downloading {} {:.0}% ({:.0}/{:.0} MB)", name, pct, mb_cur, mb_tot)
            } else {
                format!("downloading {}...", name)
            }
        }
        ModelStatus::Loading(name) => format!("loading {}...", name),
        ModelStatus::Ready(name) => format!("model: {}", name),
        ModelStatus::Error(e) => format!("model error: {}", e),
    };

    let activity_part = if let Some(since) = app.thinking_since {
        format!("  Thinking... {:.1}s", since.elapsed().as_secs_f64())
    } else {
        String::new()
    };

    let debug_tag = if app.show_debug { "  [DBG]" } else { "" };

    let status_text = format!(" {}{}{}",  model_part, activity_part, debug_tag);
    let status_bar = Paragraph::new(status_text)
        .style(Style::default().fg(Color::DarkGray).bg(Color::Black));
    f.render_widget(status_bar, chunks[3]);
}

fn draw_memory_browser(f: &mut Frame, app: &mut App) {
    let area = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(5),   // Content
        ])
        .split(area);

    // Header
    let header = Paragraph::new(
        " Memory Browser  |  j/k: Navigate  |  x: Delete  |  Esc: Back to Chat",
    )
    .style(Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(header, chunks[0]);

    // Content: split into list + detail
    let content_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Percentage(60),
        ])
        .split(chunks[1]);

    // Memory list
    let items: Vec<ListItem> = app
        .memory_items
        .iter()
        .map(|obj| {
            let type_tag = match &obj.object_type {
                MemoryObjectType::Fact => "Fact",
                MemoryObjectType::Entity => "Entity",
                MemoryObjectType::Skill => "Skill",
                MemoryObjectType::Observation => "Obs",
                MemoryObjectType::Preference => "Pref",
                MemoryObjectType::Relationship => "Rel",
                MemoryObjectType::Custom(s) => s.as_str(),
                MemoryObjectType::LogicElement(_) => "Logic",
            };
            ListItem::new(format!("[{}] {}", type_tag, obj.short_name))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .title(" Items ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Yellow),
        )
        .highlight_symbol("> ");

    f.render_stateful_widget(list, content_chunks[0], &mut app.memory_list_state);

    // Detail panel
    let detail_text = if let Some(idx) = app.memory_list_state.selected() {
        if let Some(obj) = app.memory_items.get(idx) {
            format!(
                "ID: {}\nType: {:?}\nName: {}\nDescription: {}\n\nContent:\n{}",
                obj.id, obj.object_type, obj.short_name, obj.description, obj.content
            )
        } else {
            "No item selected.".to_string()
        }
    } else {
        "No memories stored.".to_string()
    };

    let detail = Paragraph::new(detail_text)
        .block(
            Block::default()
                .title(" Detail ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::White)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(detail, content_chunks[1]);
}
