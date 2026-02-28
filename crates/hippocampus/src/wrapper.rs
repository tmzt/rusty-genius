/// Wrapper that asserts Send + Sync for types that are !Send/!Sync on wasm32.
///
/// # Safety
/// wasm32-unknown-unknown is single-threaded. Browser JS types (`JsValue`, rexie
/// handles, sqlite-wasm-rs pointers) are `!Send + !Sync` only because they hold
/// JS heap references, but there is no other thread to race with. This wrapper
/// allows us to satisfy the `Send + Sync` supertrait bounds on `MemoryStore`
/// without removing those bounds for native targets.
pub struct WasmSendSync<T>(pub T);

unsafe impl<T> Send for WasmSendSync<T> {}
unsafe impl<T> Sync for WasmSendSync<T> {}

impl<T> std::ops::Deref for WasmSendSync<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for WasmSendSync<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
