//! Integration tests for the `load-dynamic` cargo feature.
//!
//! Runs in its own integration-test binary under
//! `cargo test --no-default-features --features load-dynamic --test load_dynamic_tests`,
//! verifying `laya::init_onnx_runtime_from` pre-validation and repeated-failure safety without
//! requiring a pre-installed ONNX Runtime library.

#![cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]

use laya::{LayaError, init_onnx_runtime_from};
use std::fs;

#[test]
fn init_onnx_runtime_from_rejects_missing_invalid_and_symbol_less_binaries_safely() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does_not_exist_onnxruntime.so");

    // 1. Missing path is rejected before `ort`'s `Once`, leaving retry available.
    let err = init_onnx_runtime_from(&missing)
        .expect_err("nonexistent dylib path must return LayaError::Onnx");
    match err {
        LayaError::Onnx(msg) => {
            assert!(
                msg.contains("does_not_exist_onnxruntime.so"),
                "expected dylib path in error message, got: {msg}"
            );
        }
        other => panic!("expected LayaError::Onnx, got {other:?}"),
    }

    // 2. Non-library bytes on disk are rejected before `ort`'s `Once` as well.
    let invalid = dir.path().join("not_a_shared_library.bin");
    fs::write(&invalid, b"not an ELF, Mach-O, or PE image").expect("write dummy file");
    let err2 = init_onnx_runtime_from(&invalid)
        .expect_err("invalid dylib image must return LayaError::Onnx");
    match err2 {
        LayaError::Onnx(msg) => {
            assert!(
                msg.contains("not_a_shared_library.bin"),
                "expected invalid file path in error message, got: {msg}"
            );
        }
        other => panic!("expected LayaError::Onnx, got {other:?}"),
    }

    // 3. A real native binary lacking `OrtGetApiBase` fails inside `ort::init_from`, and a
    //    subsequent call still returns `LayaError::Onnx` cleanly rather than dereferencing an
    //    uninitialized `OnceLock` in `ort`.
    let exe = std::env::current_exe().expect("current_exe");
    let err3 = init_onnx_runtime_from(&exe)
        .expect_err("test binary lacks OrtGetApiBase and must return LayaError::Onnx");
    assert!(matches!(err3, LayaError::Onnx(_)));

    let err4 = init_onnx_runtime_from(&exe)
        .expect_err("repeat call after failed dlopen/dlsym must return LayaError::Onnx safely");
    assert!(matches!(err4, LayaError::Onnx(_)));
}
