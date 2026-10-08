# laya (Rust)

Rust port of the [Laya Python SDK](https://github.com/NandhaKishorM/laya/blob/main/README.md) (the `laya` package on PyPI, from this
repository): typed-decision inference over a local ONNX model. It reproduces the Python inference
path (the same tokenization, sequence building, calibration and answers), verified against golden
outputs recorded from the Python code (`tests/golden`), but runs on
[ONNX Runtime](https://onnxruntime.ai/) instead of PyTorch, so no Python process runs at runtime.

You ask typed questions (`choice`, `score`, `noul`) about a piece of state, and they are answered
in **one non-autoregressive forward pass** with calibrated probabilities. Nothing is generated, so
there is nothing to parse and nothing to hallucinate.

Mirrors the [Laya Python SDK](https://github.com/NandhaKishorM/laya/blob/main/README.md) module for module — see the
["Module map"](#module-map) below for the correspondence.

| | |
|---|---|
| Crate | [`laya-onnx`](https://crates.io/crates/laya-onnx) 0.1.0 (imported as `laya`), versioned separately from the Python package |
| Checkpoints | **multilingual** (`mmBERT-base`, 322M, 1024-token, 100+ languages, default), **english** (`ModernBERT-large`, 421M, 512-token, English-optimized), **typed-decisions** (`ModernBERT-large`, 421M, 1024-token, fine-tuned on four typed-decisions workflows) |
| Model format | **ONNX** (opset 18, weights in an external-data sidecar), exported from the official PyTorch checkpoints with [`tools/export_onnx.py`](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/tools/export_onnx.py). See [MODELS.md](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/MODELS.md). |
| Runtime | [ONNX Runtime](https://onnxruntime.ai/) through [`ort`](https://docs.rs/ort) 2.x (CPU by default; `cuda` / `directml` cargo features). No Python process runs at runtime. |
| Tokenizer | [`tokenizers`](https://docs.rs/tokenizers) — the same HuggingFace Rust `tokenizers` library the Python SDK uses |
| Download | opt-in, via [`ureq`](https://docs.rs/ureq) (blocking, `rustls`, feature `download`, on by default) |

---

## Contents

- [Installation](#installation)
- [Getting the model](#getting-the-model)
- [Quickstart](#quickstart)
- [Choosing a checkpoint](#choosing-a-checkpoint)
- [Writing questions](#writing-questions)
- [Built-in presets](#built-in-presets)
- [Reading results](#reading-results)
- [Thread safety](#thread-safety)
- [Execution providers and cargo features](#execution-providers-and-cargo-features)
- [Async usage](#async-usage)
- [Configuration reference](#configuration-reference-layaoptions)
- [Routing, language detection, email and shortlist](#routing-language-detection-email-and-shortlist)
- [Samples](#samples)
- [Testing](#testing)
- [Module map](#module-map)
- [Differences from the Python SDK](#differences-from-the-python-sdk)
- [Known gaps](#known-gaps)

---

## Installation

The crate is published on crates.io as **`laya-onnx`**:

```bash
cargo add laya-onnx
```

or in `Cargo.toml`:

```toml
[dependencies]
laya-onnx = "0.1.0"
# or, straight from the repository (cargo finds the `laya-onnx` package in laya-rust/):
laya-onnx = { git = "https://github.com/NandhaKishorM/laya" }
```

The package is named `laya-onnx`, but the library is imported as **`laya`**: everything lives at
the crate root (`use laya::...`), and tokenizer types are under `laya::tokenization`.

---

## Getting the model

This crate runs **ONNX** models (opset 18) and auto-detects which of two export layouts a
checkpoint directory holds:

- **Fused** — a single `model.onnx` graph with its weights in a `model.onnx.data` sidecar beside
  it, plus `rl_agent_config.json` and a nested `tokenizer/` directory
  (`tokenizer/tokenizer.json` + `tokenizer/tokenizer_config.json`). This is what
  [`tools/export_onnx.py`](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/tools/export_onnx.py) produces.
- **Split** — `encoder.onnx` (text in, hidden states out) and `head.onnx` (hidden states + marker
  metadata in, logits out) as two separate graphs, plus `rl_agent_config.json` and a root-level
  `tokenizer.json` (no sibling `tokenizer_config.json`). This is the layout laya-ts's exporter
  produces.

No Python is needed at runtime either way; Python (or, for the split layout, laya-ts) is used once,
to export the model from the published PyTorch checkpoint. Point the engine at a checkpoint
directory with `LayaEngine::from_directory(...)`, `LayaOptions::model_directory` or
`LAYA_ONNX_ROOT`, as in the [Quickstart](#quickstart) — the layout present on disk is detected
automatically (fused wins if, unusually, a directory has both); `ModelArtifacts::layout()` reports
which one was found.

See [MODELS.md](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/MODELS.md) for exporting the ONNX model step by step, where the engine looks for it
(including the split layout), and downloading.

---

## Quickstart

```rust,no_run
use laya::{LayaEngine, LayaOptions, LayaResult, Question, QuestionSet};
use serde_json::json;

let engine = LayaEngine::from_directory("C:/models/laya-multilingual")?;

// 1. State: a string, or anything serde_json::Value can represent. Insertion order is preserved
//    for objects (the crate enables serde_json's `preserve_order` feature).
let state = json!({
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today \
             or we will cancel our plan.",
});

// 2. Typed questions, keyed by an id you choose. All three question types in one set:
let questions = QuestionSet::new()
    .with("department", Question::choice(
        "Which department should handle this request?",
        [
            ("billing", "invoices, payments, refunds"),
            ("technical", "bugs, outages, system errors"),
            ("sales", "pricing, new contracts"),
            ("other", "everything else"),
        ],
    )?)?
    .with("urgency", Question::score(
        "How urgent is this request?",
        ["not urgent", "soon", "critical deadline or blocking issue"],
    )?)?
    .with("churn_risk", Question::noul(
        "Does the user threaten to cancel or leave?", serde_json::Value::Null, serde_json::Value::Null,
    ))?;

// 3. One batch, one forward pass, every question answered.
let result: LayaResult = engine.predict(state, &questions)?;
// (`predict` also accepts a state that only implements `serde::Serialize` via
// `engine.predict_serialize(&value, &questions)`.)

let department = result["department"].as_choice().unwrap();
println!("{}", department.choice);                 // billing
let urgency = result["urgency"].as_score().unwrap();
println!("{}", urgency.score);                      // 1.8974
let churn = result["churn_risk"].as_noul().unwrap();
println!("{}", churn.value());                      // false

// 4. The same questions work in any language, with no router.
let hindi = engine.predict(
    json!({ "body": "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।" }),
    &questions,
)?;
println!("{:?}", hindi["department"].as_choice().unwrap().choice);   // billing
# Ok::<(), laya::LayaError>(())
```

`result["department"]` panics on an unknown id (`Index<&str>`); use `LayaResult::get` for a
fallible lookup. `Question::choice` / `Question::score` return `Result`, so the `?` above surfaces
`LayaError::InvalidQuestion` for a duplicate label, an empty option list, or an empty level list.

---

## Choosing a checkpoint

```rust,no_run
use laya::{LayaCheckpoint, LayaEngine, LayaOptions};

let engine = LayaEngine::create(LayaOptions {
    checkpoint: LayaCheckpoint::English,
    ..Default::default()
})?;
# Ok::<(), laya::LayaError>(())
```

| Checkpoint | Encoder | Parameters | `max_len` | `head_max_len` | Best for |
|---|---|---|---|---|---|
| `Multilingual` (default) | `jhu-clsp/mmBERT-base` | 322M | 1024 | 256 | Mixed-language or unknown-language traffic |
| `English` | `answerdotai/ModernBERT-large` | 421M | 512 | 192 | English-only workloads (accuracy collapses on non-Latin scripts) |
| `TypedDecisions` | `answerdotai/ModernBERT-large` | 421M | 1024 | 256 | Four specific typed-decisions workflows only |

`max_len` and `head_max_len` come from each checkpoint's `rl_agent_config.json`
(`LayaConfig::max_len` / `LayaConfig::head_max_len`); the table above lists the shipped defaults.

---

## Writing questions

Three primitives, each built with an associated function on `Question`:

| Primitive | Constructor | Answer type | Output |
|---|---|---|---|
| **choice** | `Question::choice(instructions, options)?` / `Question::choice_labels(instructions, labels)?` | `ChoiceAnswer` | top label, probability per option, confidence |
| **score** | `Question::score(instructions, levels)?` | `ScoreAnswer` | expected level on an ordinal scale, distribution, confidence |
| **noul** | `Question::noul(instructions, if_false, if_true)` | `NoulAnswer` | calibrated P(true) in `[0, 1]` |

```rust,no_run
use laya::Question;
use serde_json::Value;

// Labels with descriptions: an ordered sequence of (label, description) pairs.
Question::choice(
    "Which team should handle this?",
    [("billing", "invoices, payments, refunds"), ("technical", "bugs, outages"), ("other", "")],
)?;

// Bare labels, no descriptions.
Question::choice_labels("What is `prompt` about?", ["coding", "writing", "other"])?;

// Score: levels from lowest to highest. The answer is a fractional expected index.
Question::score(
    "How frustrated does the customer sound?",
    ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry"],
)?;

// Noul (yes/no), with optional custom wording for each side.
Question::noul("Does the user explicitly request a refund?", Value::Null, Value::Null);
Question::noul("Is this email a phishing or scam attempt?", "a legitimate email", "phishing, scam, or fraud");
# Ok::<(), laya::LayaError>(())
```

- **Labels are positional.** The model scores one marker token per option, in the order you
  declared them — every collection here (`QuestionSet`, `ChoiceQuestion::options`) preserves
  insertion order for this reason.
- `Question::choice` rejects a duplicate label or an empty option list with
  `LayaError::InvalidQuestion`, rather than silently merging duplicates (which Python's `dict`
  would do, changing every subsequent marker position).
- `instructions`, option descriptions, and score levels are all `impl Into<serde_json::Value>`. A
  string is used as written; anything else is serialized the way `laya/presets.py`'s non-string
  instructions are, with non-ASCII characters kept literal (`ensure_ascii=False`).
- All options share a fixed head budget (`head_max_len`: 256 tokens on `multilingual` and
  `typed-decisions`, 192 on `english`). If the options cannot fit even after truncation,
  `predict` returns `LayaError::NoMarkers` instead of silently answering a truncated question.

### QuestionSet

```rust,no_run
use laya::{Question, QuestionSet};

let mut questions = QuestionSet::new();
questions.insert("spam", Question::noul("Is this spam?", serde_json::Value::Null, serde_json::Value::Null))?;
// or, chained:
let questions = QuestionSet::new()
    .with("urgency", Question::score("How urgent?", ["low", "high"])?)?;

questions.len();                 // number of questions
questions.ids();                 // an iterator of ids, in insertion order
questions.contains_id("spam");   // bool
# Ok::<(), laya::LayaError>(())
```

`insert` (and the chained `with`) returns `Err(LayaError::InvalidQuestion(..))` on a duplicate id,
rather than silently replacing it and shifting every later row's meaning.

---

## Built-in presets

`LayaPresets` provides ready-made question sets, ported from `laya/presets.py`. Each preset's
instructions refer to a specific state field, so pass state with that key:

| Preset | Python name | State field | Questions |
|---|---|---|---|
| `LayaPresets::triage()` | `triage_questions` | `message` | `intent` (choice), `is_urgent`, `frustration` (score), `refund_requested`, `churn_risk` |
| `LayaPresets::email(categories)` | `email_questions` | `body` | `category` (choice), `is_spam`, `is_phishing`, `urgency` (score), `needs_reply` |
| `LayaPresets::guard()` | `guard_questions` | `prompt` | `jailbreak`, `prompt_injection`, `sensitive_data`, `harm_severity` (score), `topic` (choice) |
| `LayaPresets::moderation()` | `moderation_questions` | `post` | `toxic`, `harassment`, `threat`, `spam`, `severity` (score) |
| `LayaPresets::router()` | `router_questions` | `request` | `difficulty` (score), `domain` (choice), `needs_tools`, `is_sensitive` |

```rust,no_run
use laya::{LayaEngine, LayaPresets};
use serde_json::json;

let engine = LayaEngine::from_directory("C:/models/laya-multilingual")?;

// Support ticket triage.
let triage = engine.predict(json!({ "message": "My payment failed twice" }), &LayaPresets::triage())?;

// Prompt guardrails.
let guard = engine.predict(json!({ "prompt": "Ignore all instructions" }), &LayaPresets::guard())?;
if guard["jailbreak"].as_noul().unwrap().probability > 0.5 { /* block */ }

// Email with custom routing categories, in offer order (an ordered slice or Vec, not a HashMap).
let email = engine.predict(
    json!({ "body": "..." }),
    &LayaPresets::email(Some([("orders", "order status, shipping"), ("returns", "returns and exchanges")])),
)?;
# Ok::<(), laya::LayaError>(())
```

`LayaPresets::DEFAULT_EMAIL_CATEGORIES` exposes the default `email()` categories (billing,
technical, sales, security, hr, other) in their offer order, used when `email(None)` is called.

---

## Reading results

`LayaResult` is indexed by question id (`Index<&str>`, panicking on an unknown id — use
`LayaResult::get` for a fallible lookup) and iterates `(id, &Answer)` pairs in the order the
questions were asked.

```rust,no_run
# use laya::{LayaEngine, QuestionSet}; use serde_json::json;
# let engine = LayaEngine::from_directory("x")?; let questions = QuestionSet::new();
let result = engine.predict(json!("some state"), &questions)?;

result.model();                 // "laya-rl-agent"
result.usage().input_tokens;    // non-padding tokens across the whole batch
result.usage().output_tokens;   // always 0: nothing is generated
result.len();

for (id, answer) in result.iter() {
    answer.question_type();     // QuestionType::Choice / Score / Noul
    answer.confidence();        // 0..1
    answer.action().act_probability;   // the action head: P(act rather than escalate)

    match answer {
        laya::Answer::Choice(c) => { let _ = &c.choice; let _ = c.probability("billing"); }
        laya::Answer::Score(s) => { let _ = s.score; let _ = s.most_likely_level(); }
        laya::Answer::Noul(n) => { let _ = n.probability; let _ = n.value(); }
    }
}

// Narrow without a match: `as_choice()` / `as_score()` / `as_noul()` return `Option<&_>`.
let department = result["department"].as_choice().unwrap();
if let Some(refund) = result.get("refund_requested") { /* optional id */ }
# Ok::<(), laya::LayaError>(())
```

**Confidence.** For `choice` and `score` it is normalized Shannon entropy, `1 - H(p) / log k`. A
single-option question gets `1.0`. For `noul` it is `max(p, 1 - p)`. Every probability, score and
confidence is rounded to 4 decimal places (`Calibration::round4`), as in Python.

---

## Thread safety

`LayaEngine` holds a multi-hundred-megabyte ONNX Runtime session. **Create one and share it**,
typically behind an `Arc`:

```rust,no_run
use laya::{LayaEngine, LayaOptions};
use std::sync::Arc;

let engine = Arc::new(LayaEngine::create(LayaOptions::default())?);

let handles: Vec<_> = (0..4).map(|_| {
    let engine = Arc::clone(&engine);
    std::thread::spawn(move || {
        // engine.predict(..) from multiple threads concurrently.
    })
}).collect();
for h in handles { h.join().unwrap(); }
# Ok::<(), laya::LayaError>(())
```

`LayaEngine` is `Send + Sync` (asserted at compile time in `src/laya_engine.rs`), so `Arc<LayaEngine>`
works across threads. Internally: `LayaTokenizer::encode` takes `&self` (the `tokenizers` crate's
`Tokenizer` needs no lock), but `ort` 2.x's `Session::run` takes `&mut self` — a `Session` is not
`Sync` on its own, even though ONNX Runtime's underlying `InferenceSession::Run` is itself
documented as thread-safe. `LayaEngine` therefore holds the ONNX session in an internal `Mutex`, so `predict` calls from
multiple threads serialize on the forward pass itself but never block on tokenization or
calibration.

---

## Execution providers and cargo features

| Feature | Default | Effect |
|---|---|---|
| `download` | on | Enables `ModelArtifacts::resolve`'s opt-in HTTP download via `ureq` (TLS through `ureq`'s default `rustls` backend — no system OpenSSL needed). Without it, `allow_download = true` returns a descriptive `LayaError::Download` instead of failing to compile a download path. |
| `cuda` | off | Forwards to `ort`'s `cuda` feature. Requires `LayaOptions::execution_provider = LayaExecutionProvider::Cuda` and a working CUDA/cuDNN install at runtime. |
| `directml` | off | Forwards to `ort`'s `directml` feature (Windows only). Requires `LayaOptions::execution_provider = LayaExecutionProvider::DirectMl`. |

```toml
laya = { path = "...", features = ["cuda"] }
```

`ort`'s default `download-binaries` feature fetches a prebuilt CPU ONNX Runtime at build time, so
CPU inference needs no system install. `LayaOptions::execution_provider` defaults to
`LayaExecutionProvider::Cpu`; requesting `Cuda` or `DirectMl` without the matching cargo feature
returns `LayaError::Onnx` naming the missing feature, rather than a `Session::builder` failure with
no context.

---

## Async usage

`predict` is CPU-bound and blocking, like the underlying ONNX Runtime call and (when
`allow_download` triggers a fetch) like `ureq`. There is no async `predict` (see
[Differences from the Python SDK](#differences-from-the-python-sdk)). From an async runtime, offload
it:

```rust,no_run
# use laya::{LayaEngine, LayaOptions, QuestionSet}; use std::sync::Arc; use serde_json::json;
# async fn example() -> laya::Result<()> {
let engine = Arc::new(LayaEngine::create(LayaOptions::default())?);
let questions = QuestionSet::new();

let engine2 = Arc::clone(&engine);
let result = tokio::task::spawn_blocking(move || {
    engine2.predict(json!("state"), &questions)
})
.await
.expect("predict panicked")?;
# let _ = result;
# Ok(())
# }
```

The same pattern applies to `ModelArtifacts::resolve` / `LayaEngine::create` with
`allow_download = true`: run it inside `spawn_blocking` if it must not stall an async executor.

---

## Configuration reference (`LayaOptions`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `model_directory` | `Option<PathBuf>` | `None` | Highest-priority artifact location. When set, all other resolution steps are skipped. |
| `checkpoint` | `LayaCheckpoint` | `Multilingual` | Derived subdirectory, HF subfolder and cache path all come from this unless overridden by setting `hugging_face_subfolder` directly, or by `model_directory`. |
| `execution_provider` | `LayaExecutionProvider` | `Cpu` | `Cuda` / `DirectMl` need the matching cargo feature (see above). |
| `intra_op_threads` | `Option<u32>` | ORT default | Threads within one operator. |
| `inter_op_threads` | `Option<u32>` | ORT default | Threads across independent operators. |
| `allow_download` | `bool` | `false` | Network access is never implicit. |
| `hugging_face_repo` | `String` | `"convaiinnovations/laya"` | Repository to download from. |
| `hugging_face_subfolder` | `Option<String>` | `None` (derived from `checkpoint`) | Subfolder override in the repository and local cache, for a custom export that doesn't follow the standard bundle layout. Prefer setting `checkpoint` instead. The computed getters `hugging_face_subfolder()` and `hugging_face_download_subfolder()` fall back to `checkpoint`'s subdirectory when this is `None`; English's repository subfolder differs from its local cache directory name, which is why the two getters exist separately. |
| `hugging_face_token` | `Option<String>` | `None` → `HF_TOKEN` | Bearer token for gated or private repos. `Some(String::new())` disables auth. |
| `cache_directory` | `Option<PathBuf>` | per-user cache | Root that downloads (and cache lookups) land under. |
| `download_progress` | `Option<Arc<dyn Fn(ArtifactDownloadProgress) + Send + Sync>>` | `None` | Per-file progress sink. |

Every field is `pub`, including `hugging_face_subfolder`, so `LayaOptions` can be built with
ordinary struct-update syntax as well as `LayaOptions::new()` plus field assignment:

```rust,no_run
use laya::LayaOptions;

let options = LayaOptions {
    allow_download: true,
    ..Default::default()
};
```

---

## Routing, language detection, email and shortlist

Runnable end-to-end example: [`samples/laya-routing`](https://github.com/NandhaKishorM/laya/tree/main/laya-rust/samples/laya-routing).

**Language detection** — no model, no network:

```rust,no_run
use laya::language_detection::LanguageDetection;
use serde_json::json;

let analysis = LanguageDetection::analyse(&json!("Mein Konto wurde zweimal belastet"));
assert_eq!(analysis.script, "latin");
assert_eq!(analysis.language.as_deref(), Some("de"));
assert!(!analysis.is_english);
```

**`LayaRouter`** picks a checkpoint per request from script/language detection (or an explicit
override), loading checkpoints lazily under an LRU cache (`Arc<dyn LayaPredictor>` per slot, so an
evicted checkpoint stays alive until every in-flight `predict` on it finishes):

```rust,no_run
use laya::laya_router::{LayaPredictor, LayaRouter, LayaRouterOptions};
use serde_json::json;
# use laya::QuestionSet;
# let questions = QuestionSet::new();

let router = LayaRouter::new(LayaRouterOptions { max_loaded: 2, ..Default::default() })?;
let result = router.predict(json!({ "message": "I was charged twice" }), &questions, None, None, None)?; // -> english
let de = router.predict(json!({ "message": "Mein Konto wurde zweimal belastet" }), &questions, None, None, None)?; // -> multilingual
println!("{}", result.routing().unwrap().reason);   // "English Latin text"
# Ok::<(), laya::LayaError>(())
```

`router.route(state, Some(&questions), None, None, None)` decides without loading or running
anything. `router.preload(None)` builds every checkpoint up front for a server or demo.

**Email cleaning** strips quoted history, signatures and disclaimers before you ask any questions:

```rust,no_run
use laya::laya_email::LayaEmail;

let cleaned = LayaEmail::clean_body(raw_body, laya::laya_email::DEFAULT_MAX_CHARS);
let state = LayaEmail::state(subject, raw_body, Some("user@acme.com"), true, std::iter::empty());
```

**Embedding shortlist** narrows a high-cardinality choice question before the model sees it, using
a caller-supplied embedding function (bring your own bi-encoder — this SDK does not include one):

```rust,no_run
use laya::laya_shortlist::LayaShortlist;
# use laya::{laya_router::LayaPredictor, QuestionSet}; use serde_json::json;
# fn embed(_texts: &[String]) -> laya::Result<Vec<Vec<f64>>> { Ok(vec![]) }
# let router: &dyn LayaPredictor = todo!(); let state = json!(""); let questions = QuestionSet::new();

let result = LayaShortlist::predict(router, state, &questions, embed, 20)?;
let kept = &result.shortlist().unwrap()["banking_intent"]; // kept.labels, kept.scores, kept.k, kept.n
# Ok::<(), laya::LayaError>(())
```

---

## Samples

`laya-rust` is a Cargo workspace (`[workspace] members` in its `Cargo.toml` lists the three
`samples/` crates, with the `laya-onnx` library itself as the workspace's root package) so the sample crates, the library, and their tests all share one `target/` directory and `Cargo.lock`. Plain
`cargo build` / `cargo test` at the workspace root still only touch the `laya-onnx` library crate (the
default member); build or test everything, including the samples, with `--workspace`:

```bash
cargo build --workspace
cargo test --workspace
```

| Sample | What it shows |
|---|---|
| [`samples/laya-sample`](https://github.com/NandhaKishorM/laya/tree/main/laya-rust/samples/laya-sample) | The quickstart above as a runnable console app: one engine, four typed questions, answered in English then Hindi. Also a parity fixture — see its README's "What it is testing". |
| [`samples/laya-benchmark`](https://github.com/NandhaKishorM/laya/tree/main/laya-rust/samples/laya-benchmark) | Cold/warm latency, p95, peak memory and an answer snapshot for one checkpoint, as one JSON object — pairs with the Python benchmark. |
| [`samples/laya-routing`](https://github.com/NandhaKishorM/laya/tree/main/laya-rust/samples/laya-routing) | Language detection, `LayaRouter` (routing decisions and multi-checkpoint predict), email cleaning and the embedding shortlist, in five sections. |

```bash
# from laya-rust/, with the exports in the repository-root onnx/ folder
cargo run --release -p laya-sample -- --model-dir ../onnx/multilingual
cargo run --release -p laya-benchmark -- --checkpoint multilingual --model-dir ../onnx/multilingual
cargo run --release -p laya-routing -- --model-root ../onnx
```

---

## Testing

```bash
cargo build
cargo build --no-default-features    # download feature off
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The suite has tiers, so it stays useful without the ~1.3 GB artifact:

| Tier | Needs | Covers |
|---|---|---|
| 1 | nothing | `PythonJson` dialects, calibration math, sequence building against a stub tokenizer, collation, presets, `ModelArtifacts` resolution/validation with temp directories |
| tokenizer | `tokenizer/tokenizer.json` per checkpoint | token ids and marker positions against recorded Python output |
| 2 | the full artifact per checkpoint | `predict` against recorded Python answers, plus a shape sweep and a concurrency test |

Golden vectors are read from `tests/golden/<checkpoint>/` (recorded from Python by
[`tools/dump_golden.py`](https://github.com/NandhaKishorM/laya/blob/main/laya-rust/tools/dump_golden.py)), overridable with `LAYA_GOLDEN_DIR`. ONNX artifacts are found via `LAYA_ONNX_DIR` (multilingual only, legacy) or
`LAYA_ONNX_ROOT` (parent of all three checkpoint subdirectories), or by walking up from the crate
root looking for `onnx/<checkpoint>/`. Tests whose artifacts are missing print a `skip:` line and
return, rather than failing — a machine with no exported checkpoints still gets a green Tier-1 run.

The advisory [Rust parity lane](../.github/workflows/rust.yml) runs on **every push and PR**,
with no path filter. It regenerates fixtures from the Python `laya/` at that commit using
[`tools/regen_golden.py`](tools/regen_golden.py), then runs the whole Cargo workspace.
Goldens are never cached. A `skip:` diagnostic or ignored test makes the lane red, even when
Cargo counts a missing-model early return as a pass. Job-level `continue-on-error` keeps Python
improvements unblocked; Rust drift needs a follow-up port. This workflow never publishes crates.

To reproduce it, install CPU torch and [`tools/requirements-regen.txt`](tools/requirements-regen.txt),
then run from the repository root:

```bash
python laya-rust/tools/test_ci_tools.py
python laya-rust/tools/regen_golden.py
cd laya-rust
cargo test --workspace --locked -- --nocapture --test-threads=1 2>&1 | tee cargo-test.log
python tools/check_test_skips.py cargo-test.log --min-passed 200
```

Use `--artifacts-root` for exports outside the checkout, with `LAYA_ONNX_ROOT=<root>/onnx`
and `LAYA_ONNX_SPLIT_ROOT=<root>/onnx-split`. Regeneration reuses only complete exports whose
stamp matches the pinned checkpoint, exporters, toolchain and Python model definition.
The fused exporter is shared with `laya-dotnet/tools/export_onnx.py`; split exports use
`laya-ts/scripts/export_onnx.py`.

Model loads are ~1.3–1.7 GB each; run model-backed tests serially (optionally with `--release`):

```bash
LAYA_ONNX_ROOT=/path/to/onnx cargo test --release -- --test-threads=1
```

`ModelArtifacts`'s own download-path unit tests (404 / 401 / 403 handling, the `.part`-then-rename
write) run against a local `TcpListener`, not the real network, so they run in every `cargo test`
invocation that has the `download` feature enabled (the default) with no external dependency.

---

## Module map

| Rust module | Purpose | Rust type(s) |
|---|---|---|
| `src/python_json.rs` | Byte-for-byte `json.dumps` dialects | `PythonJson` |
| `src/questions.rs` | Typed question definitions | `QuestionType`, `Question`, `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, `QuestionSet` |
| `src/answers.rs` | Typed answer results | `Answer`, `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `ActionInfo`, `LayaResult`, `Usage` |
| `src/calibration.rs` | Temperature scaling and confidence math | `Calibration` |
| `src/laya_config.rs` | Parsed `rl_agent_config.json` | `LayaConfig` |
| `src/laya_options.rs` | Engine construction options | `LayaOptions`, `LayaCheckpoint`, `LayaExecutionProvider`, `ArtifactDownloadProgress` |
| `src/model_artifacts.rs` | Locating/downloading model files | `ModelArtifacts`, `ModelLayout`, `ModelPaths` (the fused/split layout enum and its resolved paths — see [Getting the model](#getting-the-model)) |
| `src/laya_presets.rs` | Built-in question presets | `LayaPresets` |
| `src/sequence_builder.rs` | Token sequence construction | `SequenceBuilder` |
| `src/collator.rs` | Batches sequences for one forward pass | `Collator`, `CollatedBatch`, `SequenceItem` |
| `src/tokenization/laya_tokenizer.rs` | Tokenizer abstraction | `LayaTokenizer` (trait) |
| `src/tokenization/hf_tokenizer.rs` | HuggingFace tokenizer implementation | `HfTokenizer` |
| `src/laya_engine.rs` | Loads the ONNX model and answers questions | `LayaEngine` |
| `src/error.rs` | Error type | `LayaError`, `Result<T>` — every fallible call in this crate returns `Result<T, LayaError>` |

Where the equivalent Python code is a module of related functions (`common.py`, `agent.py`), the
Rust port groups the corresponding functions into a unit struct with associated functions
(`PythonJson::state(...)`, `Calibration::round4(...)`, ...) so call sites read the same way.

---

## Differences from the Python SDK

- **`LayaError` instead of exceptions.** One `thiserror` enum covers every failure mode Python
  expresses by raising (missing files, bad config, invalid questions, no markers left after
  truncation). Library code never panics on bad input.
- **No async `predict`.** `ModelArtifacts::resolve`'s download step and `LayaEngine::predict` are
  both synchronous, blocking calls (`ureq` for HTTP, blocking `ort::Session::run` for inference).
  An async caller wraps the call in `spawn_blocking` (see [Async usage](#async-usage)).
- **`serde_json::Value` instead of a dynamically-typed object.** State, instructions, option
  descriptions and score levels are `serde_json::Value` (with the `preserve_order` feature, so
  objects keep insertion order), the same JSON model Python's `dict`/`list`/scalar values map onto.
  Public constructors take `impl Into<Value>`, so `&str`, `String`, numbers and
  `serde_json::json!{..}` all work directly; a type that implements `serde::Serialize` but isn't
  already a `Value` goes through `serde_json::to_value` first.
- **`PythonJson::criterion`'s `default=str` fallback is not reachable.** Python's
  `render_criterion` falls back to `str(value)` for values `json.dumps` cannot serialize on its
  own; the three dialect functions here take `&serde_json::Value`, which can only ever hold
  JSON-representable data, so that fallback path has nothing to reach. The blank-criterion rule and
  the float/escaping rules are otherwise byte-for-byte identical to Python.
- **No log-level warning for clamped temperatures.** `LayaConfig` still records
  `clamped_temperatures`; emitting a warning from library code is left to the caller (this crate
  has no logging dependency).
- **`QuestionType` carries no methods.** The Python type name lives on
  `SequenceBuilder::type_name` rather than on the enum, and the `qtype` tensor's integer encoding
  is a plain `qtype as i64` cast (the enum's discriminants already match `QTYPES`). The reverse
  conversion uses `TryFrom<i64>`, since Rust has no implicit enum-from-integer cast the way
  Python's dynamic typing allows.
- **`Agent::new_with_defaults()`, reused across all five files in one download.** Matches Python's
  single reused HTTP session, but there's no async cancellation token — a download that must be
  cancellable should run on its own thread and be dropped/joined by the caller.

---

## Known gaps

- **Python 0.4.0's routing defaults, script counting and fused-line disclaimer fixes are ported**
  and checked against freshly regenerated probes. New mixed-field language heuristics remain
  a follow-up; these probes are not a claim of full coverage of every new Python feature.

- **Opt-in features added in Python 0.3.21 are not ported yet:** abstention (`min_confidence`),
  `predict_long`, and per-language temperatures (`lang_temperatures`). They are off by default in
  Python, so default predictions are unaffected. They are planned as follow-ups.
