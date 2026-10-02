//! Benchmark the Laya Rust SDK across one checkpoint.
//!
//! Measures:
//!   cold latency : engine creation + first predict (ms)
//!   warm latency : median and p95 over N_WARM timed calls (ms), after N_WARMUP warmups
//!   peak working set (MB), via the Win32 `K32GetProcessMemoryInfo` API
//!   answer snapshot for each bench case
//!
//! Outputs one JSON object to stdout.
//!
//! Usage:
//!   cargo run --release -p laya-benchmark -- --checkpoint multilingual
//!   cargo run --release -p laya-benchmark -- --checkpoint english
//!   cargo run --release -p laya-benchmark -- --checkpoint typed-decisions
//!
//! Mirrors the methodology of `laya-rust/tools/bench_python.py` (arguments, output shape, exit
//! codes) so its output can be tabulated alongside the Python SDK's, with `[bench_rust]` on the
//! progress lines.

use laya::{Answer, LayaCheckpoint, LayaEngine, LayaError, LayaOptions, Question, QuestionSet};
use serde_json::{Map, Value, json};
use std::path::PathBuf;
use std::time::Instant;

const N_WARMUP: usize = 5;
const N_WARM: usize = 30;

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // ── parse args ────────────────────────────────────────────────────────────
    let mut checkpoint = LayaCheckpoint::Multilingual;
    let mut model_directory: Option<PathBuf> = None;
    let onnx_root = std::env::var("LAYA_ONNX_ROOT").ok();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" => {
                if i + 1 >= args.len() {
                    return die("--checkpoint needs a value.");
                }
                i += 1;
                checkpoint = match args[i].as_str() {
                    "multilingual" => LayaCheckpoint::Multilingual,
                    "english" => LayaCheckpoint::English,
                    "typed-decisions" => LayaCheckpoint::TypedDecisions,
                    other => return die(&format!("Unknown checkpoint: {other}")),
                };
            }
            "--model-dir" => {
                if i + 1 >= args.len() {
                    return die("--model-dir needs a value.");
                }
                i += 1;
                model_directory = Some(PathBuf::from(&args[i]));
            }
            other => return die(&format!("Unrecognised argument: {other}")),
        }
        i += 1;
    }

    let checkpoint_name = checkpoint.subdir();

    // If LAYA_ONNX_ROOT is set and no explicit --model-dir was given, derive the dir.
    if model_directory.is_none()
        && let Some(root) = &onnx_root
        && !root.trim().is_empty()
    {
        model_directory = Some(PathBuf::from(root).join(checkpoint_name));
    }

    // ── bench cases ───────────────────────────────────────────────────────────
    // Mirrors tools/bench_python.py: the same two states and question set.
    let english_state = json!({
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "Hi, we were billed twice for March. Please refund the duplicate today \
                 or we will cancel our plan.",
    });
    let hindi_state = json!({
        "body": "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।",
    });
    let questions = match build_questions() {
        Ok(q) => q,
        Err(e) => return die(&e.to_string()),
    };
    let bench_cases: [(&str, Value); 2] = [
        ("sample_app_english", english_state),
        ("sample_app_hindi", hindi_state),
    ];

    // ── cold measurement ──────────────────────────────────────────────────────
    let cold_clock = Instant::now();
    let mut options = LayaOptions::new();
    options.checkpoint = checkpoint;
    options.model_directory = model_directory;
    let engine = match LayaEngine::create(options) {
        Ok(engine) => engine,
        Err(
            e @ (LayaError::ArtifactsNotFound { .. }
            | LayaError::DirectoryNotFound { .. }
            | LayaError::MissingFile { .. }),
        ) => {
            eprintln!("{e}");
            return 1;
        }
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    // First call (still part of cold, as ORT JITs on the first run).
    let (_, first_state) = &bench_cases[0];
    if let Err(e) = engine.predict(first_state.clone(), &questions) {
        eprintln!("{e}");
        return 1;
    }
    let cold_ms = cold_clock.elapsed().as_secs_f64() * 1000.0;

    eprintln!("[bench_rust] cold={cold_ms:.0} ms, running warm-up...");

    // ── warmup ────────────────────────────────────────────────────────────────
    let mut call_idx: usize = 0;
    let mut next_call = || {
        let (_, state) = &bench_cases[call_idx % bench_cases.len()];
        call_idx += 1;
        state.clone()
    };

    for _ in 0..N_WARMUP {
        let state = next_call();
        engine
            .predict(state, &questions)
            .expect("predict failed during warm-up");
    }

    // ── warm measurement ──────────────────────────────────────────────────────
    let mut times_ms = Vec::with_capacity(N_WARM);
    for _ in 0..N_WARM {
        let state = next_call();
        let t0 = Instant::now();
        engine
            .predict(state, &questions)
            .expect("predict failed during warm run");
        times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times_ms.sort_by(|a, b| a.partial_cmp(b).expect("timings are never NaN"));
    let median_ms = times_ms[times_ms.len() / 2];
    let p95_ms = times_ms[(times_ms.len() as f64 * 0.95) as usize];

    // ── peak working set ──────────────────────────────────────────────────────
    let peak_mb = win_mem::peak_working_set_bytes()
        .map(|bytes| bytes as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    eprintln!(
        "[bench_rust] warm median={median_ms:.1} ms  p95={p95_ms:.1} ms  peak={peak_mb:.0} MB"
    );

    // ── answer snapshot ───────────────────────────────────────────────────────
    let mut answers = Map::new();
    for (name, state) in &bench_cases {
        let result = engine
            .predict(state.clone(), &questions)
            .expect("predict failed while snapshotting answers");
        let mut case_answers = Map::new();
        for id in result.ids() {
            case_answers.insert(id.clone(), answer_to_json(&result[id.as_str()]));
        }
        answers.insert((*name).to_string(), Value::Object(case_answers));
    }

    drop(engine);

    // ── emit JSON ─────────────────────────────────────────────────────────────
    let output = json!({
        "checkpoint": checkpoint_name,
        "runtime": "Rust + ONNX Runtime",
        "threads": "ORT default",
        "cold_ms": round_to(cold_ms, 1),
        "warm_times_ms": times_ms.iter().map(|t| round_to(*t, 2)).collect::<Vec<_>>(),
        "warm_median_ms": round_to(median_ms, 1),
        "warm_p95_ms": round_to(p95_ms, 1),
        "n_warmup": N_WARMUP,
        "n_warm": N_WARM,
        "peak_wset_mb": round_to(peak_mb, 1),
        "answers": Value::Object(answers),
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("benchmark output is always valid JSON")
    );
    0
}

fn build_questions() -> laya::Result<QuestionSet> {
    QuestionSet::new()
        .with(
            "department",
            Question::choice(
                "Which department should handle this request?",
                [
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("sales", "pricing, new contracts"),
                    ("other", "everything else"),
                ],
            )?,
        )?
        .with(
            "urgency",
            Question::score(
                "How urgent is this request?",
                ["not urgent", "soon", "critical deadline or blocking issue"],
            )?,
        )?
        .with(
            "churn_risk",
            Question::noul(
                "Does the user threaten to cancel or leave?",
                Value::Null,
                Value::Null,
            ),
        )?
        .with(
            "refund_requested",
            Question::noul(
                "Does the user explicitly request a refund?",
                Value::Null,
                Value::Null,
            ),
        )
}

fn answer_to_json(answer: &Answer) -> Value {
    match answer {
        Answer::Choice(choice) => {
            let mut probabilities = Map::new();
            for (label, prob) in &choice.probabilities {
                probabilities.insert(label.clone(), json!(prob));
            }
            json!({
                "type": "choice",
                "choice": choice.choice,
                "probabilities": Value::Object(probabilities),
                "confidence": answer.confidence(),
                "act_probability": answer.action().act_probability,
            })
        }
        Answer::Score(score) => {
            let mut probabilities = Map::new();
            for (index, prob) in score.probabilities.iter().enumerate() {
                probabilities.insert(index.to_string(), json!(prob));
            }
            json!({
                "type": "score",
                "score": score.score,
                "probabilities": Value::Object(probabilities),
                "confidence": answer.confidence(),
                "act_probability": answer.action().act_probability,
            })
        }
        Answer::Noul(noul) => {
            json!({
                "type": "noul",
                "noul": noul.probability,
                "confidence": answer.confidence(),
                "act_probability": answer.action().act_probability,
            })
        }
    }
}

fn round_to(value: f64, decimals: i32) -> f64 {
    let factor = 10f64.powi(decimals);
    (value * factor).round() / factor
}

fn die(message: &str) -> i32 {
    eprintln!("{message}");
    2
}

#[cfg(windows)]
mod win_mem {
    #![allow(non_snake_case)]

    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }

    #[link(name = "psapi")]
    unsafe extern "system" {
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCountersEx,
            cb: u32,
        ) -> i32;
    }

    /// The process's peak working set, in bytes, via `K32GetProcessMemoryInfo`. `None` if the
    /// call fails (should not happen on a real process handle).
    pub fn peak_working_set_bytes() -> Option<u64> {
        let mut counters: ProcessMemoryCountersEx = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<ProcessMemoryCountersEx>() as u32;
        // SAFETY: `GetCurrentProcess` returns a valid pseudo-handle that needs no closing;
        // `counters` is a correctly-sized, zeroed buffer whose first field (`cb`) is set to that
        // size before the call, exactly as the Win32 API requires.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (ok != 0).then_some(counters.peak_working_set_size as u64)
    }
}

#[cfg(not(windows))]
mod win_mem {
    /// Best-effort on non-Windows: read `VmHWM` (peak resident set) from `/proc/self/status`.
    /// `None` when that file's format doesn't match (e.g. a non-Linux Unix).
    pub fn peak_working_set_bytes() -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }
}
