# Laya for .NET

**Laya's System 1 decision engine for .NET.** This is a .NET port of the
[Laya Python SDK](https://github.com/NandhaKishorM/laya/blob/main/README.md) (the `laya` package on PyPI, from this repository). It reproduces
the Python inference path (the same tokenization, sequence building, calibration and answers),
verified against golden outputs recorded from the Python code (`tests/Laya.Tests/golden`), but
runs on [ONNX Runtime](https://onnxruntime.ai/) instead of PyTorch, so **no Python process runs
at runtime**. What is not ported yet is listed under [Not in v1](#not-in-v1).

You ask typed questions (`choice`, `score`, `noul`) about any state (text, an email, a ticket or a
JSON-shaped object), and they are answered in **one non-autoregressive forward pass** with
calibrated probabilities. It generates no text, so there is nothing to parse and nothing to
hallucinate.

| | |
|---|---|
| Package | **[`Laya.Onnx`](https://www.nuget.org/packages/Laya.Onnx) 0.1.0** (namespace `Laya`), versioned separately from the Python package |
| Target | **.NET 10** (`net10.0`) |
| Checkpoints | **multilingual** (`mmBERT-base`, 322M, 1024-token, 100+ languages), **english** (`ModernBERT-large`, 421M, 512-token, English-optimized), **typed-decisions** (`ModernBERT-large`, 421M, 1024-token, fine-tuned on four typed-decisions workflows). Default is `multilingual`. |
| Model format | **ONNX** (opset 18, weights in an external-data sidecar), exported from the official PyTorch checkpoints with [`tools/export_onnx.py`](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/tools/export_onnx.py). See [MODELS.md](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/MODELS.md). |
| Runtime | `Microsoft.ML.OnnxRuntime` 1.30 (CPU by default; CUDA and DirectML execution providers with the matching ORT package) |
| Tokenizer | [`Tokenizers.DotNet`](https://www.nuget.org/packages/Tokenizers.DotNet), a binding to the same Rust HuggingFace `tokenizers` library Python uses. Its token streams match Python's because it runs the same code, not a reimplementation. |
| Native RIDs | `win-x64`, `linux-x64`, `linux-arm64`, `osx-x64`, `osx-arm64` |

---

## Contents

- [Installation](#installation)
- [Getting the model](#getting-the-model)
- [Quickstart](#quickstart)
- [Writing questions](#writing-questions)
- [Supplying state](#supplying-state)
- [Long prediction](#long-prediction)
- [Batch prediction](#batch-prediction)
- [Reading results](#reading-results)
- [Confidence gating](#confidence-gating)
- [Built-in presets](#built-in-presets)
- [Hosting: lifetime, threading, DI](#hosting-lifetime-threading-and-dependency-injection)
- [Routing, language detection, email and shortlist](#routing-language-detection-email-and-shortlist)
- [API reference](#api-reference)
- [Errors](#errors)
- [Performance](#performance)
- [Tests](#tests)
- [Not in v1](#not-in-v1)
- [Known gaps](#known-gaps)

---

## Installation

The package is published on NuGet as **`Laya.Onnx`**:

```bash
dotnet add package Laya.Onnx --version 0.1.0
```

Or reference the project from a clone of this repository:

```bash
dotnet add reference path/to/laya-dotnet/src/Laya/Laya.csproj
```

The package and assembly are named `Laya.Onnx`, but everything lives in the `Laya` namespace. The tokenizer abstraction is in `Laya.Tokenization`.

```csharp
using Laya;
```

---

## Getting the model

This SDK runs **ONNX** models (opset 18), in either of two artifact layouts. The
**fused** layout — this repo's own `export_onnx.py` — is a `model.onnx` graph with its weights in
a `model.onnx.data` sidecar beside it, plus `rl_agent_config.json` and `tokenizer/`. The
**split** layout — produced by `laya-ts`'s exporter — is an `encoder.onnx` and `head.onnx`
pair, each with its own `.data` sidecar when that graph's weights exceed the protobuf limit,
plus the same `rl_agent_config.json` and a root-level `tokenizer.json`. No Python is needed at
runtime for either; Python (or, for the split layout, `laya-ts`) is used once, to export the
model from the published PyTorch checkpoint.

The engine auto-detects which layout a directory holds — nothing in the API asks you to choose
one. Point it at a checkpoint directory with `LayaEngine.FromDirectory(...)`,
`LayaOptions.ModelDirectory` or `LAYA_ONNX_ROOT`, as in the [Quickstart](#quickstart).

See [MODELS.md](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/MODELS.md) for exporting the ONNX model step by step (both layouts), where the
engine looks for it, and downloading.

---

## Quickstart

```csharp
using Laya;

using var engine = LayaEngine.FromDirectory(@"C:\models\laya-multilingual");

// 1. State: a string, or any dictionary/list shape. Insertion order is preserved.
var state = new Dictionary<string, object?>
{
    ["from"] = "user@acme.com",
    ["subject"] = "Duplicate charge on invoice #4411",
    ["body"] = "Hi, we were billed twice for March. Please refund the duplicate today "
             + "or we will cancel our plan.",
};

// 2. Typed questions, keyed by an id you choose.
var questions = new QuestionSet
{
    ["department"] = Question.Choice(
        "Which department should handle this request?",
        ("billing", "invoices, payments, refunds"),
        ("technical", "bugs, outages, system errors"),
        ("sales", "pricing, new contracts"),
        ("other", "everything else")),
    ["urgency"] = Question.Score(
        "How urgent is this request?",
        "not urgent", "soon", "critical deadline or blocking issue"),
    ["churn_risk"] = Question.Noul("Does the user threaten to cancel or leave?"),
    ["refund_requested"] = Question.Noul("Does the user explicitly request a refund?"),
};

// 3. One batch, one forward pass, every question answered.
LayaResult result = engine.Predict(state, questions);

Console.WriteLine(result["department"].AsChoice().Choice);       // billing
Console.WriteLine(result["urgency"].AsScore().Score);            // 1.8974
Console.WriteLine(result["churn_risk"].AsNoul().Value);          // False
Console.WriteLine(result["refund_requested"].AsNoul().Value);    // True

// 4. The same questions work in any language, with no router.
var hindi = engine.Predict(
    new Dictionary<string, object?> { ["body"] = "मार्च का बिल दो बार लिया गया, कृपया रिफंड करें।" },
    questions);
Console.WriteLine(hindi["department"].AsChoice());               // billing (0.995 confidence)
```

The runnable version of this code is in [`samples/Laya.Sample`](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/samples/Laya.Sample/README.md):

```bash
dotnet run --project samples/Laya.Sample -- --model-dir "<artifact dir>"
```

---

## Writing questions

There are three primitives. Each one is built with a static factory on `Question`.

| Primitive | Factory | Answer type | Output | Typical uses |
|---|---|---|---|---|
| **`choice`** | `Question.Choice(...)` | `ChoiceAnswer` | top label, probability per option, confidence | department routing, intent, topic |
| **`score`** | `Question.Score(...)` | `ScoreAnswer` | expected level on an ordinal scale, distribution, confidence | urgency, frustration, harm severity |
| **`noul`** | `Question.Noul(...)` | `NoulAnswer` | calibrated P(true) from 0.0 to 1.0 | phishing, spam, jailbreak, churn risk |

### Choice

```csharp
// Labels with descriptions: (label, description) tuples.
Question.Choice("Which team should handle this?",
    ("billing", "invoices, payments, refunds"),
    ("technical", "bugs, outages"),
    ("other", null));                 // null or "" means no description

// Bare labels, with no descriptions.
Question.Choice("What is `prompt` about?", "coding", "writing", "other");

// Any ordered sequence of label/description pairs, for example built at runtime.
IEnumerable<KeyValuePair<string, object?>> options = LoadLabelsFromDb();
Question.Choice("Which intent is this?", options);
```

- **Labels are positional.** The model scores one marker token per option, in order, so the order
  you declare options in is the order they are scored in. Every collection in the SDK preserves
  insertion order for this reason. Don't pass options through a plain `Dictionary<,>`, because its
  enumeration order is not guaranteed.
- A duplicate label throws `ArgumentException`. Python would silently merge duplicates into one
  option. An empty option list also throws.
- Descriptions are `object?`. Strings are used as written. Any other value (a number, a list, a
  dictionary) is rendered as JSON. `0` and `false` are real descriptions and are rendered.
- All options share a fixed head budget (`Config.HeadMaxLen`: 256 tokens on `multilingual` and
  `typed-decisions`, 192 on `english`). Each option is capped at 48 tokens, and with many options
  each one is squeezed further. If the options can't fit at all, `Predict` throws instead of
  answering a truncated question. Past about 20 options, labels start to blur together. Split the
  question into a coarse one and a fine one.

### Score

```csharp
// Levels from lowest to highest. The answer is an expected level index: 0 .. levels-1.
Question.Score("How frustrated does the customer sound?",
    "calm and neutral",
    "concerned but civil",
    "clearly annoyed",
    "very angry or using strong language");

// Or from any sequence.
Question.Score("How severe is this?", severityLevels);
```

Levels are `object?`. They are rendered as `level 0: …`, `level 1: …` and so on. At least one level
is required.

### Noul (yes/no)

```csharp
Question.Noul("Does the user explicitly request a refund?");

// Optionally replace the default "no, the statement does not hold" /
// "yes, the statement holds" wording for each side.
Question.Noul(
    "Is this email a phishing or scam attempt?",
    ifFalse: "a legitimate email",
    ifTrue:  "phishing, scam, or fraud");
```

### Instructions

`instructions` is `object`, not `string`. A string is used exactly as written, which is what you
normally want. Any other value is serialized as JSON with non-ASCII characters kept as written,
matching the Python SDK's behavior with non-string instructions. Refer to state fields in backticks
(`` `body` ``), as the presets do.

### QuestionSet

`QuestionSet` is an insertion-ordered map from question id to `Question`. Answers come back in the
same order.

```csharp
var questions = new QuestionSet
{
    ["spam"] = Question.Noul("Is this spam?"),          // indexer: add, or replace in place
};
questions.Add("urgency", Question.Score("How urgent?", "low", "high"));  // throws on a duplicate id
questions["spam"] = Question.Noul("Is this bulk marketing?");            // keeps its position

questions.Count;               // 2
questions.Ids;                 // ["spam", "urgency"]
questions.ContainsId("spam");  // true

// Build one from any ordered sequence, for example to combine presets.
var combined = new QuestionSet(LayaPresets.Guard().Concat(LayaPresets.Moderation()));
```

Ids must be non-empty. Re-assigning an existing id with the indexer replaces the question and keeps
its position. `Add` throws instead.

---

## Supplying state

`Predict(object? state, QuestionSet questions)` accepts:

| State | How it is rendered |
|---|---|
| `string` | used as written |
| `IDictionary`, `IEnumerable<KeyValuePair<string, object?>>` | a JSON object, in the dictionary's enumeration order |
| `IList` (arrays, `List<T>`, conversation turns) and tuples (`ValueTuple`, `Tuple`) | a JSON array; only an `IList` is left-truncated when over budget, as Python truncates a `list` (a tuple keeps its start) |
| a set, a `Queue<T>`, a LINQ iterator or any other non-list enumerable | **not accepted**: serializing it throws `JsonException`, as Python's `json.dumps` rejects a set or a generator. Pass `ToList()` / `ToArray()` |
| `null`, `bool`, integer types, `BigInteger`, `double`, `float`, `decimal` | the JSON scalar |

Serialization matches Python's `json.dumps` byte for byte (`", "` / `": "` separators, float
formatting like Python's `repr`, non-ASCII kept as written). This makes the prompt identical to the
one the Python SDK builds. Nesting is allowed.

```csharp
engine.Predict("My payment failed twice", questions);                      // plain text

engine.Predict(new Dictionary<string, object?>                             // an object
{
    ["subject"] = "Refund",
    ["amount"] = 42.5,
    ["tags"] = new[] { "billing", "vip" },
}, questions);

engine.Predict(new[]                                                       // a conversation
{
    new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Cancel my plan" },
    new Dictionary<string, object?> { ["role"] = "agent", ["content"] = "May I ask why?" },
}, questions);
```

- **POCOs and anonymous types are not supported.** A value with no JSON equivalent throws
  `System.Text.Json.JsonException`, as Python raises `TypeError`. Convert records to a dictionary
  first. Circular references also throw.
- Use an insertion-ordered dictionary (`Dictionary<,>` built by adding keys, or a `List` of
  `KeyValuePair`) when key order matters. The serialized order is part of the prompt.
- A long state is **truncated from the end**. The state gets whatever the question head leaves of
  `Config.MaxLen` (1024 tokens on `multilingual` and `typed-decisions`, 512 on `english`). Put the
  most important fields first.
- A literal `<mask>` anywhere in state, instructions or options is replaced with a space, so user
  input can't forge a marker token.

---

## Long prediction

`Predict` keeps only the state tokens that fit one sequence. To scan an entire long document,
use the port of Python **`ONNXAgent.predict_long`** (there is no long-prediction API in `laya-ts`):

```csharp
LayaResult result = engine.PredictLong(document, questions, batchSize: 2);
Console.WriteLine(result.Usage.Windows);
if (result["churn_risk"].Window is { } deciding)
    Console.WriteLine($"window {deciding.Index}: tokens {deciding.TokenStart}..{deciding.TokenEnd}");

// The router routes the original document once, then scans it on that checkpoint.
using var router = new LayaRouter();
LayaResult routed = router.PredictLong(document, questions, model: "english", batchSize: 2);
```

- `window` is a state-token count, defaulting to `max(64, max_len - head_max_len - 8)`,
  capped at the smallest state budget left by the questions. `stride` defaults to half the
  effective window (at least 1). Nonpositive values select the defaults, as in Python.
- Tokens are decoded back to text and scored as ordinary states, with the same overlapping
  windows and re-tokenization as Python. `batchSize` caps windows per forward pass; without
  a positive cap they share a run, except for Python's automatic cap when a small window
  multiplies the batch by more than 2x.
- `noul` takes the greatest rounded P(true). `choice` and `score` take the greatest rounded
  `AnswerConfidence`. Ties use the first window. The complete deciding answer (including
  probabilities, score legend and action) is preserved; it is **not** a calibrated probability
  for the whole document.
- `Answer.Window` reports zero-based index, start/end offsets into the original tokenized state
  (end exclusive), and total window count. A state that fits the room the questions leave
  (or, with an explicit `window`, that window) is scored unchanged by one `Predict` pass,
  with `Usage.Windows == 1` and no answer attribution.
- Scan usage sums `InputTokens`, `StateTokens`, `StateTokensDropped` and `Truncated`
  (a window count, including 0/1 for a single window). `TruncatedQuestions` is the last
  window's list; `Options` merges collapsed-option records by question id. These additional
  diagnostics are populated by `Predict`, `PredictBatch` and `PredictLong`.
- `aggregate` accepts only `"auto"`. Null state, unsupported aggregation, no state room, or a
  gap-causing stride throws `ArgumentException` (`ArgumentNullException` for null state).
  A stride valid for a requested window that gets clamped is reduced to half its new size.
  Window/stride clamps emit `System.Diagnostics.Trace` warnings; install a listener to see them.
- The router's usual `model`, `task` and `lang` hints select a checkpoint. Attached predictors
  must implement `ILayaLongPredictor`; otherwise the router throws `NotSupportedException`.
  Language-temperature and hook arguments remain unported.

The .NET adaptations are typed records instead of Python dictionaries, trace warnings instead of
`RuntimeWarning`, and .NET argument/capability exceptions. The existing `"laya-rl-agent"` model
identifier is kept, rather than Python ONNXAgent's `"laya-rl-agent-onnx"` suffix.

---

## Batch prediction

`PredictBatch` answers the same questions about many states, sharing forward passes. It ports Python
**`ONNXAgent.predict_batch`** (`batch_size`, `sort_by_length`):

```csharp
IReadOnlyList<LayaResult> results = engine.PredictBatch(
    [ticket1, ticket2, ticket3], questions,
    new LayaBatchOptions { BatchSize = 2, SortByLength = true });   // one result per state, input order
```

- One session run holds every state's question rows. `BatchSize` caps states per run; `null`, `0` or a
  negative value sends them all in one run, exactly as Python (which raises nothing for such values).
- `SortByLength` groups similarly sized encoded states inside windows of eight batches to reduce padding.
  It has an effect only with `1 < BatchSize < states.Count`; otherwise it is ignored, as in Python. Results
  keep input order. A different batch shape can move a prediction by a rounding step near a threshold.
- An empty `states` list returns an empty list and never looks at the questions. An empty
  `QuestionSet` gives each state an empty answer set with zero usage. A null state throws
  `ArgumentNullException` (`state must not be None; pass a string, dict, or list`, the text Python's `TypeError`
  carries, as `Predict` and `PredictLong` now do too); a question whose options
  cannot fit the head budget throws `ArgumentException`, as `Predict` does. States are encoded and run a
  window at a time, so an unanswerable state fails after the earlier windows have run, as in Python.
- Each result carries the usage diagnostics Python's ONNX path reports (`StateTokens`,
  `StateTokensDropped`, `Truncated`, `TruncatedQuestions`, `Options`), the same as `Predict` reports for a single state.
- `LayaBatchOptions` is a class of init-only properties, so further per-call options extend it without
  changing a signature. `ILayaBatchPredictor` is a separate interface; `ILayaPredictor` is unchanged.

**Router.** `LayaRouter.PredictBatch(requests, options)` takes `LayaBatchRequest`s (`State`, `Questions`,
and the optional routing overrides `Model`, `Task`, `Lang`, `LangGuess`, and the token budgets `MaxLen`,
`HeadMaxLen`), routes all of them first, groups them by routed checkpoint (first-seen order, so each checkpoint
loads at most once), splits each group by question schema (order-sensitive), token budgets and, for a checkpoint
that has [per-language temperatures](#per-language-temperatures-and-per-call-options), language, so only requests
that agree on all of those share a pass, and returns results in request order, each with `Routing`. `LayaRouter.RouteBatch(requests)` returns just the `RouteDecision`s and loads nothing. A null
state or questions, or an unknown model, fails the whole call before anything loads. Attached predictors must
implement `ILayaBatchPredictor`, else `NotSupportedException`. Hooks and the `diagnostics` dictionary are not
ported yet.

### Per-language temperatures and per-call options

Python's opt-in `lang_temperatures`, and the per-call `lang`, `max_len` and `head_max_len`. All are off by default,
so default predictions are unchanged.

```csharp
using var engine = LayaEngine.Create(new LayaOptions
{
    LangTemperatures = new Dictionary<string, LayaLangTemperature?>
    {
        ["de"] = new() { Temperature = [1.4, 1.8, 2.0] },                       // choice, score, noul
        ["fr"] = null,                                                           // the checkpoint's own temperature
        ["es"] = new() { TemperatureByOptions = new Dictionary<string, double> { ["choice:3-5"] = 2.5 } },
    },
});

var result = engine.Predict(state, questions, new LayaPredictOptions { Lang = "de-AT", MaxLen = 256, HeadMaxLen = 96 });
engine.PredictBatch(states, questions, new LayaBatchOptions { Lang = "de" });
engine.PredictLong(document, questions, new LayaLongOptions { Lang = "de", BatchSize = 2 });
```

- **The table** (`LayaOptions.LangTemperatures`) maps a language code to `LayaLangTemperature` (`Temperature`: exactly three
  values in `choice`, `score`, `noul` order, or `null` to inherit the checkpoint's; `TemperatureByOptions`: bucket to
  value). It is resolved when the engine is created, as Python's `resolve_lang_temperatures` does: a key is matched on
  its part before the first `-`, lower-cased (`"de-AT"`, `"DE"` and `"de"` are one language, `"de_DE"` is not), the last
  of several keys for one language wins, every value is clamped to 0.5..5, and a value the clamp moved is reported once
  per language through `System.Diagnostics.Trace` (the text Python warns with). A `Temperature` that is not three values
  throws `ArgumentException` carrying Python's `ValueError` text. A router hands the table to every engine it builds
  (`LayaRouterOptions.EngineOptions`).
- **A call that names a language the table holds** (`Lang` on `LayaPredictOptions` / `LayaBatchOptions` /
  `LayaLongOptions`) replaces the checkpoint's per-type temperature **and** its per-bucket map with that language's; there
  is no fallback to the checkpoint's bucket map. A missing, empty or unknown language changes nothing.
- **On a router** the language is the `lang` argument of `Predict` / `PredictLong` (it also routes), else the language the
  router's own detection found (a `lang_guess` hint or an explicit `model` leaves none), and it is handed to the engine.
  In `PredictBatch` the request's `Lang` (or the detected one) joins the grouping key, as written (`"de"` and `"de-AT"`
  are separate passes), only for a checkpoint that has a table.
- **`MaxLen` / `HeadMaxLen`** replace the checkpoint's `max_len` / `head_max_len` for one call (`LayaPredictOptions`,
  `LayaBatchOptions`, and per request on `LayaBatchRequest`; not on `PredictLong`, which takes none in Python). Python
  checks nothing, and neither does this: `0`, a negative value or one too small for a question's option markers makes the
  call throw the same `ArgumentException` a checkpoint with such a budget would (`only N of its M option markers fit in
  max_len=...`), while a small or negative `HeadMaxLen` squeezes each option to 4 tokens and the instruction to 8.
  Booleans are not a .NET `int`: `True`/`False` are `1`/`0`.
### Calibration files and histogram binning

Python's opt-in `calibration=` (`load_calibration`), off by default. A calibration file carries fitted temperatures and,
optionally, a histogram-binning map; this SDK reads and applies one, it does not fit or write them.

```csharp
using var engine = LayaEngine.Create(new LayaOptions { Calibration = "calibration.json" });
```

```json
{
  "version": 1,
  "temperature": [1.8, 2.2, 1.3],
  "temperature_by_options": { "choice:3-5": 2.4, "noul:2": 0.8 },
  "binning_map": {
    "choice:3-5": { "bins": 4, "values": [0.47, 0.31, 0.92, 0.68] },
    "noul:2": { "bins": 2, "values": [0.55, 0.85] }
  }
}
```

- **When it is read.** Once, when the engine is created, after the checkpoint's own temperatures and the
  `LangTemperatures` table (as Python's `ONNXAgent.__init__` does); there is no per-call form. A null or empty path reads
  nothing. A router hands the path to every engine it builds. The file is read as strict UTF-8 and parsed like
  `json.load` (`NaN` and `Infinity` are accepted), as the checkpoint config is.
- **Temperatures.** `temperature` (exactly three) and `temperature_by_options` replace the checkpoint's, both of them: a
  bucket the file does not name falls back to the file's per-type temperature, not the checkpoint's. Values are clamped to
  0.5..5 and the ones the clamp moved are reported once through `System.Diagnostics.Trace`, with Python's text.
  `temperature_by_options` may be absent or `null`.
- **Binning.** `binning_map` maps an option-count bucket (`MinConfidence.BucketKeys`) to `bins` equal-width bins over
  [0, 1] and one recalibrated value per bin. It recalibrates `AnswerConfidence`, and only that: it is applied to the
  unrounded `max(p)` after the temperature softmax, the bin is `int(p * bins)` clipped to `0..bins-1`, and the value found
  is rounded to four places. `Confidence`, the probabilities, a noul answer's `Probability` and a score's `Score` are
  not binned. A bucket the map does not name keeps the temperature-scaled value. A call that names a language of
  `LangTemperatures` is not binned (that language's temperatures replace the file's). A file with no `binning_map`, or a
  `null` one, installs none.
- **The gate and `PredictLong` see the binned value.** `MinConfidence` compares `AnswerConfidence`, and a long scan keeps
  for a choice or score question the window with the greatest `AnswerConfidence` (a noul question the greatest
  `Probability`, which is not binned).
- **Refusals.** A file Python's `apply_calibration_payload` refuses makes `LayaEngine.Create` throw
  `InvalidDataException`, whose message is Python's exception type and text (`ValueError: calibration JSON must contain
  temperature: [3 floats]`): a payload that is not an object, a `version` that is not an integer (a missing or `null`
  one is 1; `2.0` and `"2"` are 2, `1.9` is refused), a `temperature` that is not a list of three, a
  `temperature_by_options` or `binning_map` that is not an object, and a `binning_map` entry that is not an object with an
  integer `bins >= 1` and `values` of exactly that length, each a finite number in [0, 1] (a bool, a string or a `null` is
  refused). A bad temperature is not refused but clamped; a bad binning value is, because there is nothing to fall back to.

### Abstention: `MinConfidence`

Python's opt-in `min_confidence`, off by default. Pass a `MinConfidence` per call and every answer says
whether it cleared the gate; leave it unset and nothing is written (`Abstention`, `AbstentionThreshold`
are `null`, `LowConfidence` is `false`).

```csharp
var result = engine.Predict(state, questions,
    new LayaPredictOptions { MinConfidence = MinConfidence.Of(0.85) });
var answer = result["department"];
answer.LowConfidence;        // true only when AnswerConfidence fell below the threshold
answer.Abstention;           // "passed" | "abstained" | "unevaluated" (GateStates.*)
answer.AbstentionThreshold;  // the threshold this answer was gated at

// One cut per option-count bucket (MinConfidence.BucketKeys, plus "default"):
var perBucket = MinConfidence.ForBuckets(new Dictionary<string, double>
{
    ["choice:2"] = 0.9, ["choice:3-5"] = 0.8, ["default"] = 0.6,
});
engine.PredictBatch(states, questions, new LayaBatchOptions { MinConfidence = perBucket });
router.Predict(state, questions, null, null, null, null, new LayaPredictOptions { MinConfidence = perBucket });
```

The gate compares `AnswerConfidence`, falling back to `Confidence`; an answer with neither usable (NaN, infinite)
is `unevaluated`, never `passed`. A map is read per answer bucket (`choice:3-5`, `score:11+`, `noul:2`, ...): a
bucket the map omits falls to its `"default"`, then to 0.0 (ungated). A threshold outside [0, 1], NaN or infinite,
an empty map or an unknown key throws `ArgumentException` with Python's `ValueError` text, when the call receives it
(the engine checks it before anything else, the router after the state and questions and before routing or
loading). Answers are immutable; the gate returns new answers, so a reused result never carries a stale flag.
`PredictLong` takes no threshold, as Python's `predict_long` takes none. The router and `LayaBatchOptions.MinConfidence`
apply one batch-wide threshold; it is not part of the request grouping.

`Answer.JevConfidence()` is Python's `jev_confidence`: the `confidence` Jev would report for the answer's
(rounded) probabilities, `null` for noul answers and malformed probabilities.
It sums the probabilities as CPython 3.12 and later do (`sum()` of floats is compensated from 3.12 on), so its
last digits match laya on Python 3.12+, not on 3.10 or 3.11; the goldens are recorded with 3.12.

---

## Reading results

`LayaResult` is indexed by question id. `Ids` returns the ids in the order they were asked.

```csharp
LayaResult result = engine.Predict(state, questions);

result.Model;                  // "laya-rl-agent"
result.Usage.InputTokens;      // non-padding tokens across the whole batch
result.Usage.OutputTokens;     // always 0: nothing is generated
result.Count;                  // number of answers

foreach (var id in result.Ids)
{
    Answer answer = result[id];            // throws KeyNotFoundException for an unknown id
    answer.Type;                           // QuestionType.Choice / Score / Noul
    answer.Confidence;                     // 0..1 (see below)
    answer.AnswerConfidence;               // 0..1, max(p): the same quantity on every question type
    answer.Action.ActProbability;          // the action head: P(act rather than escalate)

    switch (answer)
    {
        case ChoiceAnswer choice:
            choice.Choice;                 // the argmax label
            choice["billing"];             // the probability of one label (KeyNotFoundException if absent)
            choice.Probabilities;          // IReadOnlyList<KeyValuePair<string, double>>, in option order
            break;

        case ScoreAnswer score:
            score.Score;                   // expected level: sum of i * p[i], fractional, 0..Legend.Count-1
            score.MostLikelyLevel;         // argmax level index
            score.Probabilities;           // IReadOnlyList<double>, one per level
            score.Legend;                  // rendered level descriptions (structured criteria become JSON text)
            break;

        case NoulAnswer noul:
            noul.Probability;              // P(true)
            noul.Value;                    // Probability >= 0.5
            break;
    }
}

// Narrow without a switch. These throw InvalidOperationException if the type is wrong.
ChoiceAnswer dept = result["department"].AsChoice();
ScoreAnswer urgency = result["urgency"].AsScore();
NoulAnswer churn = result["churn_risk"].AsNoul();

// For ids that may be absent:
if (result.TryGetAnswer("refund_requested", out var refund)) { /* ... */ }

// Or enumerate (id, answer) pairs in asked order:
foreach (var (id, a) in result.Answers) Console.WriteLine($"{id}: {a}");
```

**Confidence.** For `choice` and `score` it is normalized Shannon entropy, `1 - H(p) / log k`. A
single-option question gets 1.0. For `noul` it is `max(p, 1 - p)`. Every probability, score and
confidence is rounded to 4 decimal places, as in Python.

**`AnswerConfidence`.** `max(p)` (clipped to [0, 1]), the same quantity on every question type —
unlike `Confidence`, whose formula differs between noul and the other two. It is the quantity
temperature scaling fits and calibration figures are computed on, so it is the field to gate a
single abstention threshold on across mixed question types. `Confidence` is unchanged alongside it.

**`ToString()`** gives a readable summary: `billing (0.995 confidence)`, `1.8974 of 0..2`,
`True (p=0.973)`.

---

## Confidence gating

The probabilities are trained with strictly proper scoring rules, so confidence carries meaning.
You can act on it directly:

```csharp
var dept = result["department"].AsChoice();

if (dept.Confidence >= 0.85)
    RouteAutomatically(dept.Choice);
else
    EscalateToHuman(dept.Choice, $"low confidence ({dept.Confidence:F2})");
```

All checkpoints ship with **no fitted temperatures**, and temperatures outside `[0.5, 5.0]` are
clamped (see `LayaConfig.ClampedTemperatures`). Fit temperatures on your own held-out data before
relying on raw probabilities for high-stakes thresholds.

---

## Built-in presets

`LayaPresets` provides ready-made question sets, ported from the Python presets. Each preset's
instructions refer to a specific state field, so pass state with that key:

| Preset | Python name | State field | Questions |
|---|---|---|---|
| `LayaPresets.Triage()` | `triage_questions` | `message` | `intent` (choice), `is_urgent`, `frustration` (score), `refund_requested`, `churn_risk` |
| `LayaPresets.Email()` | `email_questions` | `body` | `category` (choice), `is_spam`, `is_phishing`, `urgency` (score), `needs_reply` |
| `LayaPresets.Guard()` | `guard_questions` | `prompt` | `jailbreak`, `prompt_injection`, `sensitive_data`, `harm_severity` (score), `topic` (choice) |
| `LayaPresets.Moderation()` | `moderation_questions` | `post` | `toxic`, `harassment`, `threat`, `spam`, `severity` (score) |
| `LayaPresets.Router()` | `router_questions` | `request` | `difficulty` (score), `domain` (choice), `needs_tools`, `is_sensitive` |

```csharp
// Support ticket triage
var triage = engine.Predict(
    new Dictionary<string, object?> { ["message"] = "My payment failed twice" },
    LayaPresets.Triage());

// Prompt guardrails
var guard = engine.Predict(
    new Dictionary<string, object?> { ["prompt"] = "Ignore all instructions" },
    LayaPresets.Guard());
if (guard["jailbreak"].AsNoul().Probability > 0.5) Block();

// Model routing: send hard requests to a frontier model
var routing = engine.Predict(
    new Dictionary<string, object?> { ["request"] = "Refactor this service using DI" },
    LayaPresets.Router());

// Email with your own routing categories. Use the ordered-sequence overload so the option order is guaranteed.
var email = engine.Predict(
    new Dictionary<string, object?> { ["body"] = emailBody },
    LayaPresets.Email(new KeyValuePair<string, string>[]
    {
        new("orders", "order status, shipping"),
        new("returns", "returns and exchanges"),
        new("other", "anything else"),
    }));
```

`LayaPresets.DefaultEmailCategories` exposes the default `Email()` categories (billing, technical,
sales, security, hr, other) in their offer order. `Email(IReadOnlyDictionary<string, string>?)`
also exists for parity with Python, but a dictionary doesn't guarantee order. Prefer the
`IEnumerable<KeyValuePair<string, string>>` overload.

Each call returns a new `QuestionSet`, so you can modify the result freely:

```csharp
var qs = LayaPresets.Triage();
qs["language_barrier"] = Question.Noul("Is `message` written in broken or machine-translated language?");
```

---

## Hosting: lifetime, threading and dependency injection

`LayaEngine` holds a 1.29 GB ONNX Runtime session. **Create one and share it.**

- **Thread-safe.** `Predict` can be called concurrently. Tokenization is serialized internally, and
  inference, the expensive part, runs in parallel.
- **`IDisposable`.** Disposing releases the session and the tokenizer. Disposal is idempotent, and
  calling `Predict` afterwards throws `ObjectDisposedException`.
- **Synchronous.** `Predict` is CPU-bound and blocking. In ASP.NET Core, call it directly from the
  request handler, or offload it with `Task.Run` if you must keep a thread free.

```csharp
// Program.cs (ASP.NET Core / generic host)
builder.Services.AddSingleton(_ => LayaEngine.Create(new LayaOptions
{
    ModelDirectory = builder.Configuration["Laya:ModelDirectory"],
    IntraOpThreads = 4,              // pin in containers with a CPU limit
}));

app.MapPost("/triage", (LayaEngine engine, TicketDto ticket) =>
{
    var r = engine.Predict(new Dictionary<string, object?> { ["message"] = ticket.Text },
                           LayaPresets.Triage());
    var intent = r["intent"].AsChoice();
    return Results.Ok(new { intent = intent.Choice, intent.Confidence });
});
```

The container disposes the singleton when the host shuts down.

---

## Routing, language detection, email and shortlist

Runnable end-to-end example: [`samples/Laya.Routing`](https://github.com/NandhaKishorM/laya/tree/main/laya-dotnet/samples/Laya.Routing).

**Language detection** — no model, no network:

```csharp
var analysis = LanguageDetection.Analyse("Mein Konto wurde zweimal belastet");
// analysis.Script == "latin", analysis.Language == "de", analysis.IsEnglish == false
```

**`LayaRouter`** picks a checkpoint per request from script/language detection (or an explicit
override), loading checkpoints lazily under an LRU cache:

```csharp
using var router = new LayaRouter(new LayaRouterOptions { MaxLoaded = 2 });
var result = router.Predict(
    new Dictionary<string, object?> { ["message"] = "I was charged twice" }, questions);   // -> english
var deDe   = router.Predict(
    new Dictionary<string, object?> { ["message"] = "Mein Konto wurde zweimal belastet" }, questions); // -> multilingual
Console.WriteLine(result.Routing!.Reason);   // "English Latin text"
```

`router.PredictBatch(requests, options)` and `router.RouteBatch(requests)` do the same for a mixed request list; see
[Batch prediction](#batch-prediction). `router.Route(state, questions)` decides without loading or running anything — useful for logging
or testing the routing logic in isolation. `Preload()` builds every checkpoint up front for a
server or demo, so no request pays a cold load.

**Language hints.** Precedence is explicit `model` > `task` > detected typed-decisions workflow (opt-in) >
explicit `lang` > `LangGuess` > detection > default. An explicit `lang` decides only when its code names a
language (`en`, `EN`, `en-US`, `en_US`, `en_US.UTF-8`, `pt-BR`); blank, `C`, `C.UTF-8`, `POSIX`, `und`, `zxx`
and `mul` fall through to detection. A `LangGuess` is Python's `lang_guess`, a code or a callback of the state
(a callback returning `null` abstains, so a language-identification model can decline); pass one per call
to the `Route`/`Predict`/`PredictLong` overloads that end in `LangGuess? langGuess`, or install one for every
request with `new LayaRouterOptions { LangGuess = LangGuess.FromCallback(state => Identify(state)) }`.
Route reasons quote caller text with Python's `repr`, escaping non-printable characters.

**Email cleaning** strips quoted history, signatures and disclaimers before you ask any questions:

```csharp
var cleaned = LayaEmail.CleanBody(rawBody);
var state = LayaEmail.State(subject, rawBody, sender: "user@acme.com");
var answers = router.Predict(state, LayaPresets.Email());
```

`LayaEmail.State(subject, body, sender, clean, extra, maxChars)` takes Python's `max_chars` budget (default 3000).
It is a slice budget, applied as Python slices: `0` gives an empty body and a negative value cuts from the end
(`5 * |maxChars|` characters in all), so pass a number larger than the body to mean "no limit". It is ignored
when `clean` is `false` and is never added to the state as a field.

**Embedding shortlist** narrows a high-cardinality choice question before the model sees it, using
a caller-supplied embedding function (bring your own bi-encoder — this SDK does not include one):

```csharp
LayaEmbedFunction embed = texts => myBiEncoder.Embed(texts);
var result = LayaShortlist.Predict(router, state, questions, embed, k: 20);
var kept = result.Shortlist!["banking_intent"];   // kept.Labels, kept.Scores, kept.K, kept.N
```

---

## API reference

### `LayaEngine` : `IDisposable`

Answers typed questions about a piece of state in one forward pass. It is the .NET counterpart of
Python's `laya.Agent`.

| Member | Description |
|---|---|
| `static LayaEngine Create(LayaOptions? options = null)` | Finds the artifacts (see [resolution order](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/MODELS.md#5-where-the-engine-looks)), loads the config, and builds the ONNX session and tokenizer. May download if `AllowDownload` is set. |
| `static LayaEngine FromDirectory(string directory)` | Shorthand for `Create(new LayaOptions { ModelDirectory = directory })`. |
| `LayaResult Predict(object? state, QuestionSet questions)` | Answers every question about `state` in a single batched inference call. See [Supplying state](#supplying-state). |
| `LayaResult PredictLong(object state, QuestionSet questions, int? window = null, int? stride = null, string aggregate = "auto", int? batchSize = null)` | Scans overlapping token windows. See [Long prediction](#long-prediction). |
| `IReadOnlyList<LayaResult> PredictBatch(IReadOnlyList<object?> states, QuestionSet questions, LayaBatchOptions? options = null)` | Answers the same questions over many states in shared passes. See [Batch prediction](#batch-prediction). |
| `LayaConfig Config { get; }` | The loaded checkpoint's `rl_agent_config.json`. |
| `ModelArtifacts Artifacts { get; }` | The paths the artifacts were loaded from. |
| `void Dispose()` | Releases the session and tokenizer. Safe to call more than once. |

### `LayaCheckpoint` (enum)

Selects which checkpoint to load.

| Value | Encoder | Parameters | max\_len | head\_max\_len | Best for |
|---|---|---|---|---|---|
| `Multilingual` (default) | `jhu-clsp/mmBERT-base` | 322M | 1024 | 256 | Mixed-language or unknown-language traffic |
| `English` | `answerdotai/ModernBERT-large` | 421M | 512 | 192 | English-only workloads (accuracy collapses on non-Latin scripts) |
| `TypedDecisions` | `answerdotai/ModernBERT-large` | 421M | 1024 | 256 | Four specific typed-decisions workflows only |

### `LayaOptions`

Controls where artifacts come from and how ONNX Runtime is configured. All properties are
settable (`{ get; set; }`).

| Property | Type | Default | Notes |
|---|---|---|---|
| `ModelDirectory` | `string?` | `null` | Highest-priority artifact location. When set, all other resolution steps are skipped. |
| `Checkpoint` | `LayaCheckpoint` | `Multilingual` | Which checkpoint to load. Derived subdirectory, HF subfolder and cache path all come from this value unless overridden by `HuggingFaceSubfolder` or `ModelDirectory`. |
| `ExecutionProvider` | `LayaExecutionProvider` | `Cpu` | `Cuda` and `DirectMl` need the matching ORT native package (`Microsoft.ML.OnnxRuntime.Gpu` / `.DirectML`) in place of the CPU one. |
| `IntraOpThreads` | `int?` | ORT default | Threads within one operator. Worth pinning in a container with a CPU limit. |
| `InterOpThreads` | `int?` | ORT default | Threads across independent operators. |
| `AllowDownload` | `bool` | `false` | Network access is never implicit. Needs the ONNX export to be published (see above). |
| `HuggingFaceRepo` | `string` | `convaiinnovations/laya` | Repository to download from. |
| `HuggingFaceSubfolder` | `string` | derived from `Checkpoint` | Subfolder in the repository. Prefer setting `Checkpoint` instead. |
| `HuggingFaceToken` | `string?` | `null` → `HF_TOKEN` | Bearer token for gated or private repos. `""` disables auth. |
| `CacheDirectory` | `string?` | per-user cache | Root that downloads land under: `%LOCALAPPDATA%\laya\onnx` or `~/.cache/laya/onnx`. |
| `DownloadProgress` | `IProgress<ArtifactDownloadProgress>?` | `null` | Per-file progress sink. |

### `LayaExecutionProvider` (enum)

`Cpu` (default), `Cuda`, `DirectMl`.

### `ArtifactDownloadProgress` (record struct)

`(string FileName, long BytesReceived, long? TotalBytes)`. `TotalBytes` is `null` when the server
omits `Content-Length`. Files are downloaded to a `.part` file and renamed when complete, so an
interrupted download never leaves a file that looks complete.

### `ModelArtifacts`

Validated paths to every file the engine loads. You normally don't need this type directly.
`LayaEngine.Create` calls it for you. Use it to check or pre-fetch artifacts, for example in a
startup health check or a build step.

| Member | Description |
|---|---|
| `static ModelArtifacts Resolve(LayaOptions options)` | Resolves, validates and (if allowed) downloads synchronously. |
| `static Task<ModelArtifacts> ResolveAsync(LayaOptions options, CancellationToken ct = default)` | Async version. Supports cancelling the download. |
| `string Directory` | Absolute artifact directory. |
| `LayaArtifactLayout Layout` | `Fused` or `Split` — whichever layout was found in `Directory`. |
| `string? ModelPath` | `…/model.onnx`, or `null` when `Layout` is `Split`. |
| `string? EncoderPath` | `…/encoder.onnx`, or `null` when `Layout` is `Fused`. |
| `string? HeadPath` | `…/head.onnx`, or `null` when `Layout` is `Fused`. |
| `string ConfigPath` | `…/rl_agent_config.json` |
| `string TokenizerPath` | `…/tokenizer/tokenizer.json` (fused) or `…/tokenizer.json` (split). |

```csharp
// Pre-fetch at deploy time, then start the engine from the resolved directory.
var artifacts = await ModelArtifacts.ResolveAsync(new LayaOptions { AllowDownload = true }, ct);
using var engine = LayaEngine.FromDirectory(artifacts.Directory);
```

### `LayaConfig`

The contents of `rl_agent_config.json`. Missing keys fall back to the same defaults as Python's
`cfg.get(...)` (`max_len` 512, `head_max_len` 192, temperatures 1.0).

| Member | Description |
|---|---|
| `static LayaConfig Load(string path)` | Reads a config file. |
| `static LayaConfig Parse(string json)` | Parses config JSON. |
| `string? Encoder` | The encoder the checkpoint was trained on. For information only. |
| `int MaxLen` | Total sequence budget, in tokens (1024 on multilingual). |
| `int HeadMaxLen` | Budget for instructions plus all option fragments (256 on multilingual). |
| `IReadOnlyList<double> Temperature` | Per-question-type temperature `[choice, score, noul]`, clamped; the entries of the config as written (a list shorter than three fails at the prediction that needs the missing one, as Python does). |
| `IReadOnlyDictionary<string, double> TemperatureByOptions` | Per-bucket temperature, clamped. Keys look like `choice:3-5` (buckets `2`, `3-5`, `6-10`, `11+`). |
| `IReadOnlyList<double> TemperatureRaw` / `TemperatureByOptionsRaw` | What the checkpoint shipped, before clamping, as numbers (a value that is not a number reads `NaN`). |
| `IReadOnlyList<object?> TemperatureRawValues` / `IReadOnlyDictionary<string, object?> TemperatureByOptionsRawValues` | The same entries as the JSON held them (`long`, `double`, `string`, `bool`, `null`, list). |
| `IReadOnlyList<string> ClampedTemperatures` | Every temperature that was clamped into `[0.5, 5.0]`. A non-empty list means confidence from those buckets is uncalibrated. Entries read `name=<Python repr of the value> -> <applied>`, as Python's warning lists them. One `Trace.TraceWarning` listing every entry is raised at load time. |

```csharp
foreach (var warning in engine.Config.ClampedTemperatures)
    logger.LogWarning("Laya temperature clamped: {Bucket}", warning);
```

### `Question` (abstract) and subclasses

| Member | Description |
|---|---|
| `static ChoiceQuestion Choice(object instructions, params (string Label, object? Description)[] options)` | Choice with described options. |
| `static ChoiceQuestion Choice(object instructions, params string[] labels)` | Choice with bare labels. |
| `static ChoiceQuestion Choice(object instructions, IEnumerable<KeyValuePair<string, object?>> options)` | Choice from any ordered sequence. |
| `static ScoreQuestion Score(object instructions, params object?[] levels)` | Score over levels, lowest first. |
| `static ScoreQuestion Score(object instructions, IEnumerable<object?> levels)` | Same, from a sequence. |
| `static NoulQuestion Noul(object instructions, object? ifFalse = null, object? ifTrue = null)` | Yes/no, with optional wording for each side. |
| `QuestionType Type` | `Choice`, `Score` or `Noul`. |
| `object Instructions` | The prompt, as supplied. |

- **`ChoiceQuestion`**: `IReadOnlyList<KeyValuePair<string, object?>> Options` (index N is the
  label for logit N), `IReadOnlyList<string> Labels`.
- **`ScoreQuestion`**: `IReadOnlyList<object?> Levels`, lowest first.
- **`NoulQuestion`**: `object? IfFalse`, `object? IfTrue` (`null` means the default wording).

Questions are immutable. You can reuse the same instance across calls, sets and threads.

### `QuestionType` (enum)

`Choice = 0`, `Score = 1`, `Noul = 2`. The values match Python's `QTYPES` and the model's type
embedding.

### `QuestionSet` : `IEnumerable<KeyValuePair<string, Question>>`

| Member | Description |
|---|---|
| `QuestionSet()` | An empty set. Supports collection-initializer syntax (`["id"] = …`). |
| `QuestionSet(IEnumerable<KeyValuePair<string, Question>> questions)` | Seeded from an ordered sequence. Later duplicates replace earlier ones in place. |
| `Question this[string id] { get; set; }` | Gets a question, or adds/replaces one. Replacing keeps the original position. |
| `void Add(string id, Question question)` | Appends a question. Throws `ArgumentException` on a duplicate id. |
| `bool ContainsId(string id)` | Whether the id is present. |
| `int Count` | Number of questions. |
| `IReadOnlyList<string> Ids` | Ids in insertion order. |

### `LayaResult`

| Member | Description |
|---|---|
| `Answer this[string id]` | The answer for one question. Throws `KeyNotFoundException` for an unknown id. |
| `bool TryGetAnswer(string id, out Answer? answer)` | Non-throwing lookup. |
| `IEnumerable<KeyValuePair<string, Answer>> Answers` | Every answer, in the order asked. |
| `IReadOnlyList<string> Ids` | Question ids, in the order asked. |
| `int Count` | Number of answers. |
| `string Model` | `"laya-rl-agent"`, the same value Python reports. |
| `Usage Usage` | Token accounting for the call. |

### `Answer` (abstract) and subclasses

| Member | Description |
|---|---|
| `QuestionType Type` | Which question type produced this answer. |
| `double Confidence` | 0..1. Entropy-based for choice and score, `max(p, 1-p)` for noul. |
| `double AnswerConfidence` | 0..1. `max(p)`, the same quantity on every question type. |
| `ActionInfo Action` | The action head's output. |
| `WindowInfo? Window` | Deciding window for a multi-window scan; otherwise null. Token end is exclusive. |
| `ChoiceAnswer AsChoice()` / `ScoreAnswer AsScore()` / `NoulAnswer AsNoul()` | Narrows to the typed answer. Throws `InvalidOperationException` on a mismatch. |

- **`ChoiceAnswer`**: `string Choice`, `IReadOnlyList<KeyValuePair<string, double>> Probabilities`
  (in option order), `double this[string label]`.
- **`ScoreAnswer`**: `double Score` (expected level), `int MostLikelyLevel`,
  `IReadOnlyList<double> Probabilities`, `IReadOnlyList<object?> Legend`.
- **`NoulAnswer`**: `double Probability` (P(true)), `bool Value` (`Probability >= 0.5`).

### `ActionInfo` and `Usage` (record structs)

- `ActionInfo(double ActProbability)`: the model's own estimate of whether the answer can be
  acted on directly rather than escalated. It corresponds to Python's `action.act_probability`.
- `Usage(int InputTokens, int OutputTokens)`: non-padding input tokens across the batch.
  `OutputTokens` is always 0.
  Optional scan properties are `Windows`, `StateTokens`, `StateTokensDropped`, `Truncated`
  (all `int?`), `TruncatedQuestions` (`IReadOnlyList<string>?`) and
  `Options` (`IReadOnlyDictionary<string, OptionUsage>?`).
- `WindowInfo(int Index, int TokenStart, int TokenEnd, int Count)` and
  `OptionUsage(int Total, int Distinct, int? TokensPerOption)` are immutable record structs.
  See [Long prediction](#long-prediction) for their semantics.

### `LayaPresets` (static)

`Triage()`, `Email(IReadOnlyDictionary<string, string>? categories = null)`,
`Email(IEnumerable<KeyValuePair<string, string>> categories)`, `Guard()`, `Moderation()`, `Router()`,
and `DefaultEmailCategories`. Each method returns a new `QuestionSet`. See
[Built-in presets](#built-in-presets).

### Lower-level building blocks

These types are public so that you can inspect the prompt the model sees, test calibration, or
plug in a different tokenizer. `Predict` uses all of them internally. Most applications never need
them.

| Type | Members | Purpose |
|---|---|---|
| `SequenceBuilder` (static) | `TypeName(QuestionType)`, `RenderInstructions(Question)`, `RenderOptions(Question)`, `Build(ILayaTokenizer, object? state, Question, int maxLen, int headMaxLen, bool truncateLeft = false)` | Builds `[CLS] <type> question: … [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]` and returns the marker positions. Useful for seeing exactly how an option is rendered. |
| `Calibration` (static) | `TempMin` (0.5), `TempMax` (5.0), `ClampTemperature`, `TempBucket`, `ResolveTemperature`, `Probabilities`, `Softmax`, `ConfidenceFromProbs`, `AnswerConfidence`, `ExpectedScore`, `ArgMax`, `Round4` | Logits to calibrated probabilities, confidence and scores. |
| `Collator` (static) | `MinMarkers` (2), `Collate(IReadOnlyList<SequenceItem>, int padId)` | Pads a batch into the tensors the ONNX graph takes. |
| `CollatedBatch`, `SequenceItem` | tensor arrays and shapes | Collator input and output. |
| `PythonJson` (static) | `State`, `Criterion`, `Instructions`, `Dumps(value, escapeNonAscii, fallbackToStr)`, `Repr(double)` | Serialization that matches Python's `json.dumps` byte for byte, in the three dialects the prompt uses. |
| `Laya.Tokenization.ILayaTokenizer` | `Encode(string)`, `PadId`, `ClsId`, `SepId`, `MaskId`, `UnkId`, `MaskToken` | Tokenizer abstraction. `Encode` must add no special tokens. |
| `Laya.Tokenization.HfTokenizer` | `HfTokenizer(string tokenizerJsonPath)`, `IDisposable` | The shipped implementation. It removes the `post_processor` so no special tokens are added (caching the result next to the file, or in `%TEMP%` if read-only). Special-token ids are resolved from `tokenizer_config.json` + `tokenizer.json` at load time, so both mmBERT (`<pad>=0`, etc.) and ModernBERT (`[PAD]=50283`, etc.) tokenizers work. Throws if a token string is present in the config but absent from the id maps. |

```csharp
// See the exact option text the model scores for a question.
foreach (var line in SequenceBuilder.RenderOptions(Question.Noul("Is this spam?")))
    Console.WriteLine(line);
// false: no, the statement does not hold
// true: yes, the statement holds
```

---

## Errors

| Exception | Thrown by | When |
|---|---|---|
| `DirectoryNotFoundException` | `Create`, `ModelArtifacts.Resolve*` | No artifact in any location and `AllowDownload` is false. Also thrown when the given directory doesn't exist, or a download returns 404 / 401 / 403. The message lists what was tried and what to do. |
| `FileNotFoundException` | `Create`, `ModelArtifacts.Resolve*` | The directory exists but is missing a file. Most often `model.onnx.data` was left behind when copying. |
| `InvalidOperationException` | `Create` | `tokenizer.json` belongs to a different checkpoint (special-token id mismatch). |
| `OnnxRuntimeException` | `Create` | ORT couldn't load the graph, or the requested execution provider isn't installed. |
| `ArgumentException` | `Predict` | Empty `QuestionSet`, or a question whose options can't fit in `HeadMaxLen`. |
| `ArgumentException` | `Question.*`, `QuestionSet` | Duplicate choice label or question id, no options or levels, empty id. |
| `System.Text.Json.JsonException` | `Predict` | State contains a value with no JSON equivalent (a POCO, for example), or a circular reference. |
| `ObjectDisposedException` | `Predict` | The engine has been disposed. |
| `KeyNotFoundException` | `LayaResult[id]`, `ChoiceAnswer[label]` | Unknown id or label. Use `TryGetAnswer` for optional ids. |
| `InvalidOperationException` | `AsChoice` / `AsScore` / `AsNoul` | Answer type mismatch. |

---

## Performance

Expect **a few hundred milliseconds per call on CPU** for a 322M-parameter encoder. That's around
190–230 ms warm on a recent desktop x64 core. The first call takes roughly an extra second while
the session warms up, so run one throwaway `Predict` at startup if first-request latency matters.
The 33 ms figure in the Python README was measured on a T4 GPU and doesn't apply to CPU inference.

Latency is per *call*, not per question. The architecture is built around batching every question
into one forward pass, so asking four questions together costs about the same as asking one. Put
related questions in one `QuestionSet` rather than calling `Predict` once per question.

---

## Tests

The suite has three tiers, so it stays useful without the 1.3 GB artifact:

| Tier | Needs | Covers |
|---|---|---|
| 1 | nothing | `PythonJson` dialects, collation, calibration math, option-budget truncation, presets, artifact-layout detection (`ModelArtifactsTests`), tokenizer special-token fallback against synthetic fixtures (`HfTokenizerFallbackTests`) |
| tokenizer | `tokenizer/tokenizer.json` | token ids and marker positions against recorded Python output |
| 2 | the full artifact | `Predict` against recorded Python answers, and a per-checkpoint shape sweep (72 configurations for multilingual and typed-decisions; 66 for english whose `max_len=512` drops the 1024-token length) — run against **both** artifact layouts when both are present (`onnx/<checkpoint>` fused, `onnx-split/<checkpoint>` split); a checkpoint missing one layout only skips that layout's rows |

```bash
# Run with all three checkpoints, fused layout (recommended)
LAYA_ONNX_ROOT=/path/to/onnx dotnet test --solution Laya.slnx

# Backward compat: multilingual only
LAYA_ONNX_DIR=/path/to/onnx/multilingual dotnet test --solution Laya.slnx
```

The split-layout theories (`...Split` test names) look for `<repo>/onnx-split/<checkpoint>`
via a fixed walk-up from the test binary — there is no split-layout environment variable yet.
They skip per-checkpoint like the fused ones do when that directory is absent.

Checkpoints and layouts whose artifacts are missing are skipped, not failed. **How many tests
run is not a fixed number** — most of Tier 2 is parameterized per checkpoint, per shape, and now
per layout, so the count grows every time golden-answer coverage or shape-sweep coverage grows.
As a concrete snapshot from this repository: the 646 tests that need no ONNX artifact at all
(Tier 1 plus the tokenizer-lookup fixtures) always run with `0` failed and `0` skipped; a filtered
run excluding the five model-backed classes (`PredictParityTests`, `ShapeSweepTests`,
`TokenizerParityTests`, `RouterEndToEndTests`, `ShortlistEndToEndTests`) confirms this:

```bash
dotnet bin/Release/net10.0/Laya.Tests.dll \
    -class- "Laya.Tests.PredictParityTests" \
    -class- "Laya.Tests.ShapeSweepTests" \
    -class- "Laya.Tests.TokenizerParityTests" \
    -class- "Laya.Tests.RouterEndToEndTests" \
    -class- "Laya.Tests.ShortlistEndToEndTests"
```

With every checkpoint present in **both** layouts, the full suite (including the five
model-backed classes above) is 1358 tests — up from a pre-split-layout snapshot of 1088;
the "621" this replaced was measured before several rounds of per-checkpoint parameterization and
had already gone stale independent of the split layout. Re-run the count yourself with
`dotnet test --solution laya-dotnet/Laya.slnx -c Release` once artifacts are in place, since this
number is expected to keep moving as coverage grows.

The parity vectors in `tests/Laya.Tests/golden/<checkpoint>/` are recorded by
[`tools/dump_golden.py`](https://github.com/NandhaKishorM/laya/blob/main/laya-dotnet/tools/dump_golden.py) `--checkpoint all`, which runs the **real** `laya.Agent.system_one` over the
same ONNX graph. The vectors come from the shipping Python code, not from a second implementation.
The same golden answers back both the fused and the split parity theories: the split graph is the
fused graph cut into two ONNX files, not a different model, so one recording serves both.

### Regenerating golden fixtures / CI

The goldens are recorded from the Python package, so they have to be re-recorded whenever a change
to `laya/` moves something they capture (sequence building, option rendering, calibration,
language detection, ...). One command does the whole chain: it downloads each checkpoint at the
Hugging Face revision pinned in `tools/regen_golden.py` (`HF_REVISION`), exports the fused layout
(`tools/export_onnx.py`) and the split layout (`laya-ts/scripts/export_onnx.py`), then runs
`tools/dump_golden.py` and `tools/dump_routing_golden.py` into `tests/Laya.Tests/golden/`.

```bash
# once: CPU-only torch plus the toolchain the committed goldens were recorded with
pip install "torch==2.14.0" --index-url https://download.pytorch.org/whl/cpu
pip install -r laya-dotnet/tools/requirements-regen.txt

# from the repository root; all three checkpoints and the routing goldens
python laya-dotnet/tools/regen_golden.py

# or one piece, with the exported artifacts somewhere that has room (1-3 GB per checkpoint)
python laya-dotnet/tools/regen_golden.py --checkpoint english --artifacts-root /path/to/artifacts
python laya-dotnet/tools/regen_golden.py --checkpoint routing   # no model needed

cd laya-dotnet  # global.json selects the SDK and Microsoft.Testing.Platform runner
LAYA_ONNX_ROOT=/path/to/artifacts/onnx dotnet test --solution Laya.slnx -c Release
```

Exports are reused when they are already present and were made from the same inputs, so only the
first run is slow. The tests find the split layout only at `<some ancestor of the test binary>/onnx-split`,
so with a custom `--artifacts-root` use a directory junction/symlink from the repository's
`onnx-split` to the exported `onnx-split` directory, or export at the default repository root.
A diff in `git diff -- laya-dotnet/tests/Laya.Tests/golden` after
regenerating is expected whenever the Python behavior changed on purpose; review it, and commit it
together with the C# change that follows it.

`.github/workflows/dotnet.yml` runs the same chain on every push and pull request (there is no path
filter, so a change under `laya/` triggers it): the `build` job re-records the routing goldens,
builds the whole solution with warnings as errors and runs the tests that need no model; one
`parity` job per checkpoint and layout (fused, split), and one for the router end-to-end tests
(which need english and multilingual together), exports that checkpoint, re-records its goldens
from the Python code at that commit and runs its model-backed classes against them.
`LAYA_TEST_CHECKPOINTS` selects the checkpoint theory rows and `LAYA_TEST_LAYOUTS` (`fused`,
`split`, or both when unset) the layout rows of the theories that have one per layout; the
split-only tests carry `[Trait("Layout", "split-only")]` and the layout-parameterised ones
`Layout=any`, which the split cell selects with `--filter-trait` (ordinary local runs still cover
everything). Every job then runs
`tools/check_test_skips.py`, because `dotnet test` reports skipped tests as success: it fails the
run when any test that should have run was skipped, and tolerates only the skips for checkpoints
the job does not export. The job logs print `git diff --stat` of the regenerated goldens against
the committed ones; that is informational (the C# tests and their tolerances are the gate).
Exports are cached on the pinned revision, the exporters, `requirements-regen.txt` and
`laya/common.py`; pinned Hugging Face downloads are cached separately.
The test project is an xUnit **v3** application running on Microsoft.Testing.Platform. The .NET 10
SDK no longer runs these projects through VSTest. That's why `laya-dotnet` has a
`global.json` that selects the `Microsoft.Testing.Platform` runner, and why the solution is passed
with `--solution` rather than as a positional argument.

---

## Not in v1

- Quantized artifacts.
- Async `Predict`.
- `embed_fn_from_agent` (mean-pooling the checkpoint's own encoder into a shortlist embedder) —
  `LayaShortlist` takes only a caller-supplied embedding function; see [Routing, language
  detection, email and shortlist](#routing-language-detection-email-and-shortlist).

## Known gaps

Default and long predictions are verified against goldens regenerated from Python
**0.4.11**, and routing against goldens from Python **0.4.12**. Language detection uses current script and mixed-language rules; `MixedSegment`
identifies the foreign line or field that overrides a mostly-English state. Undetermined
Latin text uses the router's configured default. Email cleaning follows current request,
disclaimer, signature and reply-header rules, including Portuguese, Spanish and French markers.

- **Opt-in features added in Python 0.3.21:** abstention (`min_confidence`) and per-language temperatures
  (`lang_temperatures`, with per-call `lang`) are ported (see "Abstention" and "Per-language temperatures" above).
  Calibration files and histogram binning (`calibration=` / `load_calibration`) are ported too (see "Calibration files and
  histogram binning" above); fitting and saving one (`fit_binning`, `save_calibration`) is not.
- **Other Python-only APIs are not ported:** hooks, question option ordering and digest controls.
  Batching is ported (`PredictBatch`, `LayaRouter.PredictBatch` / `RouteBatch`) without hooks and the router
  `diagnostics` dictionary.
- **Usage diagnostics:** Python's `state_tokens`, `truncated`, `state_tokens_dropped`, `truncated_questions`
  and `options` are reported by `Predict`, `PredictBatch` and `PredictLong`, as Python's `system_one` is
  `predict_batch([state])`.

## Differences from Python

Where this SDK deliberately does not match the Python reference, so nobody has to find out from a failing load:

- **`max_len` and `head_max_len` are checked when the config is loaded.** An integer `>= 0` or a boolean (`true` is 1,
  `false` is 0, as in Python) is accepted. Any other value is refused by `LayaConfig.Parse` / `LayaEngine.Create` with
  an `InvalidDataException`: a float, string, `null`, list or object gets the `TypeError` text Python raises at the first
  prediction (head first, then `max_len`), and a negative integer gets a message naming the field and value, because
  Python then slices from the end, which this port does not support. Python loads such a checkpoint and fails later. The
  related corner: a small *fractional* `head_max_len` that Python happens to survive (roughly up to 8 plus 4 per option)
  is refused here too.
- **`LangTemperatures` is a typed table.** Python also refuses a key that is not a string, an entry or
  `temperature_by_options` that is not a mapping, and a `temperature` that is not a list; a typed table cannot hold
  those, so only the three-value check (and the inherited-list check) remains. A warning prints a float the way Python
  prints a float (`99.0`), where Python prints the `99` an integer was written as.
- **A router takes the language, and the token budgets, in the places Python does.** `LayaPredictOptions.Lang` on
  `LayaRouter.Predict` and `LayaBatchOptions.Lang` / `MaxLen` / `HeadMaxLen` on `LayaRouter.PredictBatch` throw
  `ArgumentException`: Python's `predict` has the one `lang` argument, and `predict_batch` has no such arguments (it is a
  `TypeError` there), only the per-request keys.
- **A version 2 calibration file's checkpoint-identity warning is not emitted.** Python warns when the `model_id_or_path`,
  `subfolder` or `config` a file records differs from the agent's, but its `model_id_or_path` is the string the caller
  passed (a PyTorch checkpoint path or Hub id), which this SDK has no equivalent of. Temperatures, binning and errors are
  unaffected, and the file loads as Python's does.
- **A calibration file's refusals are `InvalidDataException`.** Python raises `ValueError` (and an `OverflowError` for an
  integer too large for a float); the message here is the exception's name and text, as for a malformed checkpoint config.
  The clamp warning is a `RuntimeWarning` in Python; it reaches `Trace` here as text, with no category, shown once per call
  site.
- **JSON nesting.** Config and tokenizer JSON is read like `json.loads`, with the nesting cut-off set to 2,997 levels,
  what CPython 3.12 measured from a shallow call. Python's real limit depends on the call depth at the point of loading.
- **Config and calibration files are decoded as strict UTF-8.** Python opens them with the locale's encoding (cp1252 on a default Windows
  install), which only differs for non-ASCII bytes; a BOM is rejected, as `json.loads` does.

---

## License

Apache 2.0. Developed by Convai Innovations.
