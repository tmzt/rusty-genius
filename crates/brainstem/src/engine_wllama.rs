use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::channel::mpsc;
use futures::sink::SinkExt;
use rusty_genius_core::engine::Engine;
use rusty_genius_core::manifest::InferenceConfig;
use rusty_genius_core::protocol::InferenceEvent;
use wasmtime::*;

struct HostState {
    wasi: wasmtime_wasi::p1::WasiP1Ctx,
    token_sender: Option<mpsc::Sender<Result<InferenceEvent>>>,
    embedding_buffer: Vec<f32>,
}

pub struct WllamaEngine {
    #[allow(dead_code)]
    wasm_engine: wasmtime::Engine,
    #[allow(dead_code)]
    module: Module,
    store: Store<HostState>,
    instance: Option<Instance>,
    loaded: bool,
}

// Safety: WllamaEngine is only ever accessed through &mut self (Engine trait),
// so no concurrent access is possible. The Sync bound is required by the Engine
// trait but all methods take &mut self.
unsafe impl Sync for WllamaEngine {}

impl WllamaEngine {
    pub fn new() -> Self {
        let wasm_engine = wasmtime::Engine::default();
        let module = Module::new(&wasm_engine, "(module)").expect("failed to create empty module");
        let host_state = HostState {
            wasi: wasmtime_wasi::WasiCtxBuilder::new().build_p1(),
            token_sender: None,
            embedding_buffer: Vec::new(),
        };
        let store = Store::new(&wasm_engine, host_state);
        Self {
            wasm_engine,
            module,
            store,
            instance: None,
            loaded: false,
        }
    }

    pub fn from_wasm_bytes(bytes: &[u8]) -> Result<Self> {
        let wasm_engine = wasmtime::Engine::default();
        let module = Module::new(&wasm_engine, bytes)?;
        Self::build(wasm_engine, module)
    }

    pub fn from_wasm_file(path: &str) -> Result<Self> {
        let wasm_engine = wasmtime::Engine::default();
        let module = Module::from_file(&wasm_engine, path)?;
        Self::build(wasm_engine, module)
    }

    fn build(wasm_engine: wasmtime::Engine, module: Module) -> Result<Self> {
        let mut linker = Linker::<HostState>::new(&wasm_engine);

        // Add WASI p1 imports
        wasmtime_wasi::p1::add_to_linker_sync(&mut linker, |state: &mut HostState| {
            &mut state.wasi
        })?;

        // Host import: emit_token(ptr: i32, len: i32)
        linker.func_wrap(
            "env",
            "emit_token",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                    Some(m) => m,
                    None => return,
                };
                let data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;
                if end > data.len() {
                    return;
                }
                let token = String::from_utf8_lossy(&data[start..end]).to_string();
                if let Some(sender) = caller.data_mut().token_sender.as_mut() {
                    let _ = sender.try_send(Ok(InferenceEvent::Content(token)));
                }
            },
        )?;

        // Host import: emit_embedding(ptr: i32, len: i32)
        // ptr = byte offset in wasm memory, len = number of f32 elements
        linker.func_wrap(
            "env",
            "emit_embedding",
            |mut caller: Caller<'_, HostState>, ptr: i32, len: i32| {
                let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                    Some(m) => m,
                    None => return,
                };
                let data = memory.data(&caller);
                let byte_offset = ptr as usize;
                let float_count = len as usize;
                let byte_len = float_count * 4;
                let end = byte_offset + byte_len;
                if end > data.len() {
                    return;
                }
                let floats: Vec<f32> = data[byte_offset..end]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                caller.data_mut().embedding_buffer = floats;
            },
        )?;

        // Create store and instantiate
        let host_state = HostState {
            wasi: wasmtime_wasi::WasiCtxBuilder::new().build_p1(),
            token_sender: None,
            embedding_buffer: Vec::new(),
        };
        let mut store = Store::new(&wasm_engine, host_state);
        let instance = linker.instantiate(&mut store, &module)?;

        Ok(Self {
            wasm_engine,
            module,
            store,
            instance: Some(instance),
            loaded: false,
        })
    }

    /// Write bytes into guest memory via guest_alloc, returns the guest pointer.
    fn write_to_guest(
        store: &mut Store<HostState>,
        instance: &Instance,
        data: &[u8],
    ) -> Result<i32> {
        let guest_alloc = instance
            .get_typed_func::<i32, i32>(&mut *store, "guest_alloc")
            .map_err(|e| anyhow!("guest_alloc not found: {}", e))?;

        let ptr = guest_alloc
            .call(&mut *store, data.len() as i32)
            .map_err(|e| anyhow!("guest_alloc call failed: {}", e))?;

        let memory = instance
            .get_memory(&mut *store, "memory")
            .ok_or_else(|| anyhow!("no memory export"))?;

        let mem_data = memory.data_mut(&mut *store);
        let start = ptr as usize;
        let end = start + data.len();
        if end > mem_data.len() {
            return Err(anyhow!("write_to_guest: out of bounds"));
        }
        mem_data[start..end].copy_from_slice(data);

        Ok(ptr)
    }

    /// Free guest memory via guest_dealloc.
    fn free_guest(
        store: &mut Store<HostState>,
        instance: &Instance,
        ptr: i32,
        size: i32,
    ) -> Result<()> {
        let guest_dealloc = instance
            .get_typed_func::<(i32, i32), ()>(&mut *store, "guest_dealloc")
            .map_err(|e| anyhow!("guest_dealloc not found: {}", e))?;

        guest_dealloc
            .call(&mut *store, (ptr, size))
            .map_err(|e| anyhow!("guest_dealloc call failed: {}", e))?;

        Ok(())
    }
}

#[async_trait]
impl Engine for WllamaEngine {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no wasm instance"))?
            .clone();

        let path_bytes = model_path.as_bytes();
        let ptr = Self::write_to_guest(&mut self.store, &instance, path_bytes)?;

        let load_model_fn = instance
            .get_typed_func::<(i32, i32), i32>(&mut self.store, "load_model")
            .map_err(|e| anyhow!("load_model export not found: {}", e))?;

        let result = load_model_fn
            .call(&mut self.store, (ptr, path_bytes.len() as i32))
            .map_err(|e| anyhow!("load_model call failed: {}", e))?;

        let _ = Self::free_guest(&mut self.store, &instance, ptr, path_bytes.len() as i32);

        if result != 0 {
            return Err(anyhow!("guest load_model returned error code {}", result));
        }

        self.loaded = true;
        Ok(())
    }

    async fn unload_model(&mut self) -> Result<()> {
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no wasm instance"))?
            .clone();

        let unload_fn = instance
            .get_typed_func::<(), i32>(&mut self.store, "unload_model")
            .map_err(|e| anyhow!("unload_model export not found: {}", e))?;

        let result = unload_fn
            .call(&mut self.store, ())
            .map_err(|e| anyhow!("unload_model call failed: {}", e))?;

        if result != 0 {
            return Err(anyhow!(
                "guest unload_model returned error code {}",
                result
            ));
        }

        self.loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn default_model(&self) -> String {
        "wllama-default".to_string()
    }

    async fn preload_model(&mut self, _model_path: &str, _purpose: &str) -> Result<()> {
        Ok(())
    }

    async fn infer(
        &mut self,
        _model: Option<&str>,
        prompt: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let (mut tx, rx) = mpsc::channel(256);

        // Send ProcessStart
        let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;

        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no wasm instance"))?
            .clone();

        // Set up token sender in host state
        self.store.data_mut().token_sender = Some(tx.clone());

        let prompt_bytes = prompt.as_bytes();
        let ptr = Self::write_to_guest(&mut self.store, &instance, prompt_bytes)?;

        let infer_fn = instance
            .get_typed_func::<(i32, i32, i32), i32>(&mut self.store, "infer")
            .map_err(|e| anyhow!("infer export not found: {}", e))?;

        let result = infer_fn
            .call(
                &mut self.store,
                (ptr, prompt_bytes.len() as i32, 0), // mode 0 = chat
            )
            .map_err(|e| anyhow!("infer call failed: {}", e))?;

        let _ = Self::free_guest(&mut self.store, &instance, ptr, prompt_bytes.len() as i32);

        // Clear token sender
        self.store.data_mut().token_sender = None;

        if result < 0 {
            let _ = tx
                .send(Err(anyhow!("guest infer returned error code {}", result)))
                .await;
        }

        // Send Complete
        let _ = tx.send(Ok(InferenceEvent::Complete)).await;

        Ok(rx)
    }

    async fn embed(
        &mut self,
        _model: Option<&str>,
        input: &str,
        _config: InferenceConfig,
    ) -> Result<mpsc::Receiver<Result<InferenceEvent>>> {
        let (mut tx, rx) = mpsc::channel(256);

        // Send ProcessStart
        let _ = tx.send(Ok(InferenceEvent::ProcessStart)).await;

        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no wasm instance"))?
            .clone();

        // Clear embedding buffer
        self.store.data_mut().embedding_buffer.clear();

        let input_bytes = input.as_bytes();
        let ptr = Self::write_to_guest(&mut self.store, &instance, input_bytes)?;

        let infer_fn = instance
            .get_typed_func::<(i32, i32, i32), i32>(&mut self.store, "infer")
            .map_err(|e| anyhow!("infer export not found: {}", e))?;

        let result = infer_fn
            .call(
                &mut self.store,
                (ptr, input_bytes.len() as i32, 1), // mode 1 = embed
            )
            .map_err(|e| anyhow!("infer (embed mode) call failed: {}", e))?;

        let _ = Self::free_guest(&mut self.store, &instance, ptr, input_bytes.len() as i32);

        if result < 0 {
            let _ = tx
                .send(Err(anyhow!("guest embed returned error code {}", result)))
                .await;
        } else {
            // Collect embeddings from host state
            let embeddings = std::mem::take(&mut self.store.data_mut().embedding_buffer);
            let _ = tx.send(Ok(InferenceEvent::Embedding(embeddings))).await;
        }

        // Send Complete
        let _ = tx.send(Ok(InferenceEvent::Complete)).await;

        Ok(rx)
    }
}
