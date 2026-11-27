//! Runtime loading and environments are process-wide. The preload-only case starts a fresh
//! process so it cannot inherit the environment created by the caller-owned case.
#![cfg(feature = "load-dynamic")]

mod common;

use common::{golden_data, result_compare, test_artifacts};
use laya::{LayaCheckpoint, LayaEngine, LayaError, LayaOptions};
use ort::environment::GlobalThreadPoolOptions;
use std::{env, path::PathBuf, process::Command};

fn english_options() -> Option<LayaOptions> {
    let checkpoint = LayaCheckpoint::English;
    if !test_artifacts::is_selected(checkpoint) {
        return None;
    }
    let Some(directory) = test_artifacts::locate(checkpoint) else {
        eprintln!("skip: set LAYA_ONNX_ROOT to the parent of the English ONNX artifact");
        return None;
    };
    Some(LayaOptions {
        checkpoint,
        model_directory: Some(directory),
        intra_op_threads: Some(2),
        inter_op_threads: Some(1),
        ..Default::default()
    })
}

fn check_prediction(options: LayaOptions) {
    let engine = LayaEngine::create(options).unwrap();
    let case =
        golden_data::CheckpointGoldenData::for_checkpoint("english").load("case_quickstart.json");
    let questions = golden_data::build_questions(&case["questions"]).unwrap();
    let result = engine.predict(case["state"].clone(), &questions).unwrap();
    result_compare::assert_result_matches("environment/quickstart", &case["result"], &result);
}

fn runtime_path() -> PathBuf {
    env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            format!(
                "{}onnxruntime{}",
                env::consts::DLL_PREFIX,
                env::consts::DLL_SUFFIX
            )
            .into()
        })
}

#[test]
fn caller_environment_survives_engine_recreation() {
    let Some(options) = english_options() else {
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    assert!(ort::init_from(temp.path().join("missing-runtime")).is_err());
    let builder = ort::init_from(runtime_path()).unwrap();
    let pool = GlobalThreadPoolOptions::default()
        .with_intra_threads(1)
        .unwrap();
    let environment = builder.with_global_thread_pool(pool).build().unwrap();

    // A default build fails until Laya receives the caller's existing environment.
    assert!(matches!(LayaEngine::create(options.clone()),
        Err(LayaError::Onnx(message)) if message.contains("LayaOptions::environment")));
    for _ in 0..2 {
        check_prediction(LayaOptions {
            environment: Some(environment.clone()),
            ..options.clone()
        });
    }
    drop(environment);
    check_prediction(options);
}

#[test]
fn default_environment_uses_the_preloaded_library() {
    let Some(options) = english_options() else {
        return;
    };
    const PRELOAD_PATH: &str = "LAYA_TEST_PRELOAD_PATH";
    if let Some(path) = env::var_os(PRELOAD_PATH) {
        drop(ort::init_from(path).unwrap());
        check_prediction(options.clone());
        check_prediction(options);
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let child = Command::new(env::current_exe().unwrap())
        .args([
            "--exact",
            "default_environment_uses_the_preloaded_library",
            "--nocapture",
        ])
        .env(PRELOAD_PATH, runtime_path())
        .env("ORT_DYLIB_PATH", temp.path().join("must-not-be-loaded"))
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "preloaded-library child exited with {}\n{}\n{}",
        child.status,
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
}
