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
- [Long prediction](#long-prediction)
- [Batch prediction](#batch-prediction)
- [Abstention (`min_confidence`)](#abstention-min_confidence)
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

## Long prediction

`predict` reads one sequence: a state longer than the room the questions leave is truncated (the
start is kept, or the end for a conversation-turn list). `LayaEngine::predict_long` scans the whole
state instead, in overlapping token windows, and keeps one answer per question. It ports
`ONNXAgent.predict_long` with `aggregate="auto"`:

```rust,no_run
# use laya::{LayaEngine, LongOptions, QuestionSet};
# let engine = LayaEngine::from_directory("x")?; let questions = QuestionSet::new();
# let long_transcript = "";
let options = LongOptions::new().window(96).stride(48).batch_size(8);
let result = engine.predict_long(long_transcript, &questions, &options)?;

result.scan_usage().unwrap().windows;      // windows scored (1 = no scan was needed)
let answer = &result["refund_requested"];
if let Some(window) = answer.window() {    // None for a single-pass state
    println!("decided by window {}/{} (tokens {}..{})",
             window.index + 1, window.count, window.token_start, window.token_end);
}
# Ok::<(), laya::LayaError>(())
```

- **Aggregation.** `noul` takes the window with the greatest P(true). `choice` and `score` take the
  window with the greatest `answer_confidence`. A tie goes to the first window. The probabilities
  are that window's, **not** a calibrated number for the whole document.
- **One pass.** A state that fits the room the questions leave (with an explicit
  `LongOptions::window`, a state that fits that window) is scored by the ordinary `predict` pass:
  one window, no `Answer::window()`.
- **Options.** `window`, `stride` and `batch_size` are optional and `0` counts as unset. The default
  window is `max(64, max_len - head_max_len - 8)`, capped at the smallest state room any question
  leaves; the default stride is half the effective window. A stride that would skip tokens between
  windows is `LayaError::InvalidWindow`, as is a question set whose options leave no room for the
  state. `aggregate` accepts only `"auto"` (`LayaError::UnsupportedAggregate` otherwise).
- **Usage.** `result.usage().input_tokens` is summed over windows. `result.state_usage()` sums
  `state_tokens` and `state_tokens_dropped` over windows and lists the last window's truncated
  questions; `result.scan_usage()` gives the window count and how many windows lost state tokens.
- **Re-encoding.** Each window is decoded back to text with special tokens kept (Python's default
  `decode`) and scored as an ordinary state, so its tokens can differ slightly from the original
  after re-encoding, exactly as in Python. A tokenizer that does not override `LayaTokenizer::decode`
  makes `predict_long` fail with `LayaError::Tokenizer`.
- **Router.** `LayaRouter::predict_long(state, &questions, &options, model, task, lang)` routes once,
  on the original state, then scans on the chosen checkpoint. A predictor without a scan capability
  (see `LayaPredictor::as_long_predictor`) gives `LayaError::NotSupported` rather than truncating.
- **Warnings.** The two `warnings.warn` notices Python emits when it caps the window or reduces the
  stride are sent, with Python's exact text, to the [`log`](https://docs.rs/log) facade at `warn`
  level with target `laya`. Install any `log` backend (for example `env_logger`) to see them; with
  none installed they are dropped.
- **Not ported.** Per-call hooks and per-language temperatures are not part of `predict_long`
  here. As in Python, it takes no `min_confidence`.

---

## Batch prediction

`LayaEngine::predict_batch` answers the same questions over many states and shares forward passes
between them: N states cost one pass, or `ceil(N / batch_size)` of them, instead of N. It ports
`ONNXAgent.predict_batch` and gives each state the result `predict` would, in input order:

```rust,no_run
# use laya::{BatchOptions, LayaEngine, QuestionSet};
# let engine = LayaEngine::from_directory("x")?; let questions = QuestionSet::new();
# let states: Vec<serde_json::Value> = vec![];
let options = BatchOptions::new().batch_size(8).sort_by_length(true);
let results = engine.predict_batch(&states, &questions, &options)?;   // results[i] answers states[i]
# Ok::<(), laya::LayaError>(())
```

- **`batch_size`.** The most states per forward pass. `None` or `0` sends them all in one pass, as
  Python does for `None`, `0` and negative values.
- **`sort_by_length`.** Groups similarly sized states inside windows of eight batches, so each batch
  pads to a shorter maximum. It takes effect only with a `batch_size` above one and below the number
  of states, and results keep input order. Different batch shapes can change floating-point
  predictions slightly near a decision threshold.
- **Edges.** No states give an empty list. No questions give one empty result per state, with no
  state usage. A `null` state is `LayaError::InvalidState` (Python's `TypeError`); any other error
  fails the whole call.
- **Router.** `LayaRouter::predict_batch(&requests, &options)` takes `BatchRequest`s (state,
  questions and the optional `model`, `task`, `lang` and `lang_guess` overrides), routes each,
  groups them by checkpoint and then by question schema (order-sensitive, so option order is
  positional), runs each group through `predict_batch` and returns the results in request order,
  each with its `routing`. `LayaRouter::route_batch(&requests)` returns only the decisions and loads
  nothing. A predictor without a batch capability (see `LayaPredictor::as_batch_predictor`) gives
  `LayaError::NotSupported`.
- **`min_confidence`.** `BatchOptions::min_confidence` gates every result of the call; see
  [Abstention](#abstention-min_confidence).
- **Not ported.** Per-call hooks, `lang`, per-request `max_len` / `head_max_len` and the router's
  `diagnostics` report. `BatchOptions` and `BatchRequest` are `#[non_exhaustive]`
  so those can be added without breaking callers.

---

## Abstention (`min_confidence`)

An opt-in gate on `answer_confidence` (`max(p)`, the quantity the calibration figures describe). It
ports `min_confidence` from Python's `system_one`, `predict_batch` and `Router.predict`. A threshold
is a `MinConfidence`: one number for every answer, or a map from option-count bucket to threshold,
because one level does not transfer across option counts.

```rust,no_run
# use laya::{LayaEngine, MinConfidence, PredictOptions, QuestionSet};
# let engine = LayaEngine::from_directory("x")?; let questions = QuestionSet::new();
let gate = MinConfidence::map([("choice:2", 0.8), ("choice:3-5", 0.6), ("default", 0.5)])?;
let options = PredictOptions::new().min_confidence(gate);
let result = engine.predict_with_options("I want a refund", &questions, &options)?;
for (_id, answer) in result.iter() {
    // "passed", "abstained" or "unevaluated"; None when no threshold was set.
    println!("{:?} below {:?}: {}", answer.abstention(), answer.abstention_threshold(), answer.low_confidence());
}
# Ok::<(), laya::LayaError>(())
```

- **Buckets.** Keys are `choice`, `score` or `noul` plus `:2`, `:3-5`, `:6-10` or `:11+`
  (`laya::bucket_keys()`), and `"default"` for any bucket the map does not name. A bucket the map
  does not name, with no `"default"`, is gated at `0.0`, which gates nothing.
- **Verdict.** With a threshold set every answer reports `abstention` (`laya::GATE_STATES`) and
  `abstention_threshold`; `low_confidence()` is true only for `"abstained"`. An answer with no
  usable confidence is `"unevaluated"`, not a pass. A threshold of `0.0` was set, so it still
  reports states. With no threshold nothing is reported (`abstention()` is `None`).
- **Validation.** `MinConfidence::scalar` and `MinConfidence::map` refuse what Python's
  `check_min_confidence` refuses (out of `[0.0, 1.0]`, NaN, an empty map, an unknown bucket key)
  with Python's `ValueError` text as `LayaError::InvalidMinConfidence`. The variants are public, so
  the engine and the router validate again where Python does: the router before it routes or loads
  anything, the engine before it looks at the states.
- **Router.** `LayaRouter::predict_with_options` and `BatchOptions::min_confidence` on
  `LayaRouter::predict_batch` (one threshold for the whole batch, not part of the grouping). The
  router applies the gate itself, so an attached predictor needs no support for it.
- **`predict_long`.** Takes no threshold, as in Python.
- **`Answer::jev_confidence()`.** The `confidence` Jev would report for the answer's probabilities
  (choice and score only; `None` for noul or malformed probabilities). It is a different formula
  from Laya's normalized entropy, so a threshold carried over from Jev selects about twice as much
  on Laya.
- **Python 3.12 arithmetic.** `jev_confidence` sums with Python's compensated `sum()` (CPython
  3.12 and later), so its last digits match a 3.12+ Python, not 3.11 or earlier.

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

A state that reads English as a whole is still checked line by line and field by field, so a
Portuguese message under an English stack trace goes to the multilingual checkpoint
(`LanguageAnalysis::mixed_segment` names the line). An explicit `lang` that names no language
(blank, `C`, `POSIX`, `C.UTF-8`, `und`, `zxx`, `mul`) falls through to detection instead of pinning a
checkpoint. If you already run a language-identification model, hand its answer to the router as a
`LangGuess` -- a code or a callback, checked after an explicit `lang` and before detection, and
abstaining (`None`, blank, `C`, ...) falls through:

```rust,no_run
use laya::laya_router::{LangGuess, LayaRouter, LayaRouterOptions};
use serde_json::json;

// one hint for every request ...
let router = LayaRouter::new(LayaRouterOptions::default())?.with_lang_guess("ro");
// ... or per call, as a function of the state
let hint = LangGuess::callback(|state| my_language_id_model(state));
let decision = router.route_with_lang_guess(&json!("Care este ora in Tokyo?"), None, None, None, None, Some(&hint))?;
# fn my_language_id_model(_: laya::laya_router::GuessInput<'_>) -> Option<String> { None }
# Ok::<(), laya::LayaError>(())
```

**Email cleaning** strips quoted history, signatures and disclaimers before you ask any questions:

```rust,no_run
use laya::laya_email::LayaEmail;

let cleaned = LayaEmail::clean_body(raw_body, laya::laya_email::DEFAULT_MAX_CHARS);
let state = LayaEmail::state(subject, raw_body, Some("user@acme.com"), true, std::iter::empty());
```

It reads English, Portuguese, Spanish and French mail (Gmail's `Em ... escreveu:` / `Le ... a écrit :`,
Outlook's `-----Mensagem original-----`, `Atenciosamente` / `Cordialement`, `Envoyé depuis mon
iPhone`, confidentiality footers). `max_chars` is a slice budget: raise it for a long message
(`LayaEmail::state_with_max_chars`, `clean_body_signed`). `0` gives an empty body and a negative value
trims from the end, exactly as in Python, so "no limit" means a budget larger than the body.

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
The skip guard runs only after successful Cargo tests; regeneration/build failures retain their
own diagnostics instead of producing a secondary missing-log error.

The regeneration self-check compares option logits within `1e-3` and every **unrounded action
softmax probability** within `1e-4`, matching the signal Python and Rust predict consume.
Action classes are not padded option slots and none are excluded. Raw action logits remain
recorded (and their max difference reported) for diagnostics, but large saturated logits can
vary across CPU kernels without changing action probabilities. Nonfinite or mismatched-shaped
outputs are rejected. The recorded-graph Rust test uses the same action probability bound.

Engine results expose Python 0.4.0's state-budget diagnostics via `LayaResult::state_usage()`:
original state tokens, worst-case dropped tokens, whether truncation occurred, and the affected
question ids. The existing `Usage` totals and result constructor remain source-compatible.
`state_usage().options` additionally identifies questions whose clipped option spans collide.

To reproduce it, install CPU torch and [`tools/requirements-regen.txt`](tools/requirements-regen.txt),
then run from the repository root:

```bash
python laya-rust/tools/test_ci_tools.py
python laya-rust/tools/regen_golden.py
cd laya-rust
cargo test --workspace --locked -- --nocapture --test-threads=1 2>&1 | tee cargo-test.log
python tools/check_test_skips.py cargo-test.log --min-passed 200
```

Float-formatting probes can also be regenerated without models:
`python laya-rust/tools/regen_golden.py --checkpoint repr`. Their IEEE-754 bit patterns cover
half-even ties, subnormals, notation boundaries and 2,000 deterministic random finite values.

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
| `src/answers.rs` | Typed answer results | `Answer`, `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `ActionInfo`, `LayaResult`, `Usage`, `ScanUsage`, `StateUsage`, `WindowInfo` |
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
| `src/long_prediction.rs` | Window plan and aggregation for `predict_long` | `LongOptions` |
| `src/batch_prediction.rs` | Chunking, sort windows and order restoration for `predict_batch` | `BatchOptions` |
| `src/confidence.rs` | The abstention gate and Jev's `confidence` | `MinConfidence`, `GATE_STATES`, `bucket_keys` |
| `src/predict_options.rs` | Per-call options for `predict_with_options` | `PredictOptions` |
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
- **Warnings use the `log` facade.** The clamped-temperature warning (`LayaConfig::parse`) and the
  two `predict_long` window warnings are sent, with Python's text, to `log` at `warn` level with
  target `laya`. `LayaConfig::clamped_temperatures` still records the clamped entries.
- **`QuestionType` carries no methods.** The Python type name lives on
  `SequenceBuilder::type_name` rather than on the enum, and the `qtype` tensor's integer encoding
  is a plain `qtype as i64` cast (the enum's discriminants already match `QTYPES`). The reverse
  conversion uses `TryFrom<i64>`, since Rust has no implicit enum-from-integer cast the way
  Python's dynamic typing allows.
- **`Agent::new_with_defaults()`, reused across all five files in one download.** Matches Python's
  single reused HTTP session, but there's no async cancellation token — a download that must be
  cancellable should run on its own thread and be dropped/joined by the caller.
- **Config values Python reads loosely are refused at load.** `max_len` and `head_max_len` accept a
  non-negative JSON integer or a bool (`true` is 1, `false` is 0), as Python does. A float, string,
  `null`, list or object is refused when the engine is created, with the `TypeError` text Python
  raises at its first prediction (`head_max_len` is met first); a negative value is refused with
  `max_len=-5: must be a non-negative integer`, since Python slices from the end for it. Python
  also tolerates a small fractional `head_max_len` (roughly up to 8 plus 4 per option); that is
  refused here too.
- **Lone surrogate escapes in config JSON are refused.** `"\ud800"` loads in Python and is an error
  here, because a Rust `String` cannot hold a lone surrogate.
- **JSON nesting is cut off at 1000 levels.** Python's limit depends on the interpreter's
  recursion limit.
- **Config files are decoded as strict UTF-8.** Python opens them with the locale's encoding (cp1252
  on a default Windows install), which only differs for non-ASCII bytes; a BOM is rejected, as
  `json.loads` does.

---

## Known gaps

- **Language detection, routing and email cleaning follow Python 0.4.14** and are checked against
  routing probes and a seeded differential corpus regenerated at that version
  (`tools/regen_golden.py --checkpoint routing`). A later Python change needs them regenerated.

- **Opt-in features are not ported yet:** histogram-binning calibration and per-language
  temperatures (`lang_temperatures`). `predict_long`, `predict_batch` and the `min_confidence`
  gate are ported, without hooks or per-language temperatures. They are off by default in
  Python, so default predictions are unaffected. They are planned as follow-ups.
