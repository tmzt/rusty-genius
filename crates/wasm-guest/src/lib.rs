#![allow(clippy::missing_safety_doc)]

use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicBool, Ordering};

static MODEL_LOADED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn emit_token(ptr: i32, len: i32);
    fn emit_embedding(ptr: i32, len: i32);
}

#[no_mangle]
pub extern "C" fn guest_alloc(size: i32) -> i32 {
    let layout = Layout::from_size_align(size as usize, 1).unwrap();
    unsafe { alloc(layout) as i32 }
}

#[no_mangle]
pub extern "C" fn guest_dealloc(ptr: i32, size: i32) {
    let layout = Layout::from_size_align(size as usize, 1).unwrap();
    unsafe { dealloc(ptr as *mut u8, layout) }
}

#[no_mangle]
pub extern "C" fn load_model(name_ptr: i32, name_len: i32) -> i32 {
    // Read the model name (validates it's valid UTF-8)
    let _name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    MODEL_LOADED.store(true, Ordering::SeqCst);
    0 // success
}

#[no_mangle]
pub extern "C" fn unload_model() -> i32 {
    MODEL_LOADED.store(false, Ordering::SeqCst);
    0
}

#[no_mangle]
pub extern "C" fn is_loaded() -> i32 {
    if MODEL_LOADED.load(Ordering::SeqCst) {
        1
    } else {
        0
    }
}

/// Run inference on the given prompt.
/// mode: 0 = chat (emits tokens via emit_token), 1 = embed (emits floats via emit_embedding)
/// Returns token/embedding count on success, -1 on error.
#[no_mangle]
pub extern "C" fn infer(prompt_ptr: i32, prompt_len: i32, mode: i32) -> i32 {
    if !MODEL_LOADED.load(Ordering::SeqCst) {
        return -1;
    }

    let prompt = unsafe {
        let slice = std::slice::from_raw_parts(prompt_ptr as *const u8, prompt_len as usize);
        std::str::from_utf8_unchecked(slice)
    };

    if mode == 1 {
        // Embedding mode: generate deterministic mock floats
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
        unsafe {
            emit_embedding(embedding.as_ptr() as i32, embedding.len() as i32);
        }
        return embedding.len() as i32;
    }

    // Chat mode: split prompt into words, emit each as a token
    let words: Vec<&str> = prompt.split_whitespace().collect();
    let mut count: i32 = 0;
    for word in &words {
        let bytes = word.as_bytes();
        unsafe {
            emit_token(bytes.as_ptr() as i32, bytes.len() as i32);
        }
        count += 1;
    }

    // Emit end token
    let end_token = b"<|end|>";
    unsafe {
        emit_token(end_token.as_ptr() as i32, end_token.len() as i32);
    }
    count += 1;

    count
}
