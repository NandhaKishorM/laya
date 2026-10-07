# Laya.Sample

**The SDK quickstart as a runnable console app.** It asks four typed questions about a support
email in one forward pass, once in English and once in Hindi, using the same question objects. It is
the smallest program that covers the whole .NET path:

1. finding the artifacts
2. tokenization
3. collation
4. one ONNX Runtime forward pass
5. reading out calibrated answers

No Python runs in the process at any point.

It is also a **parity fixture**. Its exact inputs are recorded as golden cases, and the test suite
checks them against the real Python `laya.Agent.system_one`. See
[What it is testing](#what-it-is-testing).

For the full library API, see the [SDK README](../../README.md).

---

## Run it

The app needs a local ONNX export. The directory must contain all four files, because the ONNX
external-data format looks for `model.onnx.data` relative to the graph file:

```text
model.onnx              # graph, 2.5 MB
model.onnx.data         # weights, 1.29 GB
rl_agent_config.json
tokenizer/tokenizer.json
```

> **`--model-dir` takes one checkpoint's folder, not the export root.** This sample loads a single
> checkpoint, so point it at the folder that holds `model.onnx` (for example `onnx/multilingual`),
> not at `onnx/` itself. `--model-root` belongs to [`Laya.Routing`](../Laya.Routing/README.md)
> and is rejected here with `Unrecognised argument: --model-root`.

Everything after the bare `--` is passed to the app; without it, `dotnet run` tries to read the
flags itself. From the repository root:

```bash
dotnet run --project laya-dotnet/samples/Laya.Sample -- --model-dir "/path/to/onnx/multilingual"
```

Or from this sample's own directory (`laya-dotnet/samples/Laya.Sample`):

```powershell
dotnet run -- --model-dir C:/path/to/onnx/multilingual
dotnet run -- --model-dir C:/path/to/onnx/english --checkpoint english
dotnet run -- --model-dir C:/path/to/onnx/typed-decisions --checkpoint typed-decisions
```

Pass `--checkpoint` whenever the folder isn't the multilingual one, so the engine loads it as the
right checkpoint.

Or set an environment variable once and leave out `--model-dir`. `LAYA_ONNX_DIR` points at one
checkpoint's folder; `LAYA_ONNX_ROOT` points at the export root, and `--checkpoint` picks the
subfolder:

```powershell
$env:LAYA_ONNX_DIR = "C:\path\to\onnx\multilingual"
dotnet run --project laya-dotnet/samples/Laya.Sample

# or
$env:LAYA_ONNX_ROOT = "C:\path\to\onnx"
dotnet run --project laya-dotnet/samples/Laya.Sample -- --checkpoint english
```

### Command-line options

| Option | Effect |
|---|---|
| `--model-dir <dir>` | One checkpoint's artifact directory (the folder holding `model.onnx`). Takes priority over `LAYA_ONNX_DIR` and `LAYA_ONNX_ROOT`. |
| `--checkpoint <name>` | Checkpoint to load: `multilingual` (default), `english`, `typed-decisions`. |
| `--download` | Fetch the artifact from Hugging Face into the per-user cache (`LayaOptions.AllowDownload`), with per-file progress on stderr. |
| `-h`, `--help` | Print usage and exit 0. |

The artifact is resolved from `--model-dir`, then `LAYA_ONNX_DIR`, then
`$LAYA_ONNX_ROOT/<checkpoint>`, then the per-user cache (`%LOCALAPPDATA%\laya\onnx\<checkpoint>`
or `~/.cache/laya/onnx/<checkpoint>`).

| Exit code | Meaning |
|---|---|
| `0` | Both runs printed. |
| `1` | No usable artifact. The resolver's message, which lists every location tried, is printed to stderr. |
| `2` | Bad arguments: `--model-dir`/`--checkpoint` with no value, an unknown checkpoint name, or an unrecognized argument (including `--model-root`, which only `Laya.Routing` accepts). |

Two things cause trouble in practice:

- **Quote Windows paths.** A POSIX shell (Git Bash, WSL) drops unquoted backslashes, so
  `C:\models\laya\multilingual` arrives as `C:modelslayamultilingual`. Windows treats
  that as a *drive-relative* path and resolves it against the current directory on `D:`. Quote the
  path or use forward slashes. The resolver prints both the requested path and the resolved one, so
  you can see when this has happened.
- **`LAYA_ONNX_DIR` must be absolute.** A relative value is resolved against each process's own
  working directory, and for the test host that is not the repository root.

`--download` is wired up but **can't succeed today**. `convaiinnovations/laya` publishes PyTorch
weights (`model.safetensors`), not an ONNX export, so every file returns 404. The downloader
reports this in plain words instead of showing a bare stack trace, and it will work without changes
once the export is uploaded.

---

## Expected output

```text
── English ─────────────────────────────────
model      : laya-rl-agent
latency    : 973 ms  (348 input tokens, 0 output tokens)
department       : billing  (confidence 1.00, act 1.00)
urgency          : 1.90     (confidence 0.69, act 1.00)
churn_risk       : no       (confidence 0.87, act 1.00)
refund_requested : yes      (confidence 0.97, act 1.00)
  department distribution: billing 100.0 %, technical 0.0 %, sales 0.0 %, other 0.0 %

── Hindi ─────────────────────────────────
model      : laya-rl-agent
latency    : 230 ms  (236 input tokens, 0 output tokens)
department       : billing  (confidence 0.99, act 1.00)
urgency          : 1.78     (confidence 0.48, act 1.00)
churn_risk       : no       (confidence 0.91, act 1.00)
refund_requested : yes      (confidence 0.98, act 1.00)
  department distribution: billing 99.9 %, technical 0.0 %, sales 0.0 %, other 0.0 %
```

Everything except the latencies is deterministic. The percentage format follows your current
culture, so the spacing around `%` may differ. The first call spends about a second warming up the
ORT session. The second takes about 200 ms on a desktop x64 core.

Latency is per *call*, not per question. All four questions go through one batch and one forward
pass. That is the main idea of the architecture.

---

## Code walkthrough

[`Program.cs`](Program.cs) has five parts.

**1. Argument parsing.** The parsing is written by hand, and anything it doesn't recognize is
rejected with exit code 2 rather than ignored. This matters for `--download <path>`: it reads as
though the path were the cache location, and silently dropping the path would send the run
somewhere the user never asked for.

**2. Creating the engine.**

```csharp
engine = LayaEngine.Create(new LayaOptions
{
    ModelDirectory = modelDirectory,          // null falls through to LAYA_ONNX_DIR, then the cache
    AllowDownload = allowDownload,            // network access is never implicit
    DownloadProgress = allowDownload
        ? new Progress<ArtifactDownloadProgress>(p => Console.Error.WriteLine($"  {p}"))
        : null,
});
```

`DirectoryNotFoundException` and `FileNotFoundException` are caught together, and only their
message is printed. The resolver's message already names every location it tried and what to do
next, so a stack trace would add nothing. The hint "point --model-dir or LAYA_ONNX_DIR at …" is
shown only when no download was attempted.

**3. State and questions.** The state is a `Dictionary<string, object?>`: `from`, `subject` and
`body`, in that order. The key order is part of the serialized prompt. The `QuestionSet` holds one
question of each type:

| Id | Factory | Shape |
|---|---|---|
| `department` | `Question.Choice(…, ("billing", "…"), …)` | four labels with descriptions, scored by position |
| `urgency` | `Question.Score(…, "not urgent", "soon", "critical …")` | three ordinal levels, answered with an expected level of 0..2 |
| `churn_risk` | `Question.Noul(…)` | yes/no with calibrated `P(true)` |
| `refund_requested` | `Question.Noul(…)` | yes/no with calibrated `P(true)` |

**4. Two predictions.** The English state is answered first. Then a Hindi state containing only
`body` (a duplicate March bill and a refund request) is answered by **the same `QuestionSet`
instance**. There is no router or language detection. The multilingual checkpoint (mmBERT-base,
322M parameters, 100+ languages) handles Devanagari directly.

**5. `Report`.** This function times `engine.Predict`, prints `result.Model` and
`result.Usage`, and loops over `result.Ids` in the order the questions were asked. For each id it
narrows the base `Answer` with a `switch` expression:

```csharp
var value = answer switch
{
    ChoiceAnswer choice => choice.Choice,
    ScoreAnswer score   => score.Score.ToString("F2"),
    NoulAnswer noul     => noul.Value ? "yes" : "no",
    _ => "?",
};
```

It prints `answer.Confidence` and `answer.Action.ActProbability` next to each value. Finally it
uses `AsChoice()` to get the full `department.Probabilities` distribution, which is what you need
when the top label alone isn't enough to act on.

### What each part demonstrates

| Part of the sample | What it shows |
|---|---|
| `Dictionary<string, object?>` state | State can be any dictionary or list shape, not only a string. Insertion order is preserved because it is part of the prompt. |
| `Question.Choice` with `(label, description)` tuples | Labels are **positional**. The model scores one marker per option in the declared order, so `QuestionSet` and `Question.Choice` both preserve order. |
| `Question.Score` with three level descriptions | It returns an expected level (a probability-weighted sum over level indices), not a class. |
| `Question.Noul` | Yes/no with a calibrated `P(true)`. |
| A single `engine.Predict(state, questions)` | All three question types answered in one batched forward pass. |
| The Hindi state | The same question objects work in another script, with no router. |
| `result[id]` plus a `switch` on the answer type | The indexer returns the base `Answer`. Pattern matching or `As*()` narrows it. |
| `department.Probabilities` | The full distribution, in option order. |
| `result.Usage.InputTokens` | Tokens across the whole batch. This is the key parity signal (see below). |

---

## Adapting the sample

The sample is meant to be copied as a starting point. Common changes:

**Use a preset instead of writing your own questions.** Each preset expects a specific state key:

```csharp
Report(engine, new Dictionary<string, object?> { ["message"] = "My payment failed twice" },
       LayaPresets.Triage(), "Triage");
```

`Report` prints the `department` distribution with a fixed id, so remove those last lines (or look
the id up with `TryGetAnswer`) when you use a question set that has no `department` question.

**Gate on confidence** instead of printing:

```csharp
var dept = result["department"].AsChoice();
if (dept.Confidence >= 0.85) Route(dept.Choice); else Escalate(dept);
```

**Pin the thread count** in a container with a CPU limit, or change the execution provider:

```csharp
LayaEngine.Create(new LayaOptions { ModelDirectory = dir, IntraOpThreads = 4 });
```

`LayaExecutionProvider.Cuda` or `DirectMl` also requires replacing the CPU ORT package with
`Microsoft.ML.OnnxRuntime.Gpu` or `.DirectML`.

**Keep the engine alive.** The sample creates one engine and disposes it at exit. In a real
service, register it as a singleton. It is thread-safe and holds the 1.29 GB session, so creating
one per request would repeat the load and warm-up every time.

If you change the states or questions **in this project**, read the next section first. They are
recorded test fixtures.

---

## What it is testing

The sample itself **asserts nothing**; it only prints. The checks are in the test suite, and the
two are linked on purpose. The sample's two states and four questions are duplicated in
[`tools/sample_cases.py`](../../tools/sample_cases.py).
[`tools/dump_golden.py`](../../tools/dump_golden.py) runs them through the **real,
unmodified** `laya.Agent.system_one` to record
[`case_sample_app_english.json`](../../tests/Laya.Tests/golden/multilingual/case_sample_app_english.json) and
[`case_sample_app_hindi.json`](../../tests/Laya.Tests/golden/multilingual/case_sample_app_hindi.json).

`PredictParityTests` replays those two cases through the C# engine and compares the results. The
tolerance is **2e-4** on probabilities, confidences and act-probabilities, and **2e-3** on scores.
Both sides are recorded to four decimal places. So the values printed above are not just what the
port happens to produce: they are what the shipping Python code produces on the same bytes.

The recorded answers, which the suite enforces:

| Question | English | Hindi |
|---|---|---|
| `department` | `billing`, p = 1.0000, confidence 1.0000 | `billing`, p = 0.9992, confidence 0.9947 |
| `urgency` | score 1.8974, confidence 0.6932 | score 1.7759, confidence 0.4776 |
| `churn_risk` | P(true) 0.1345, confidence 0.8655 | P(true) 0.0940, confidence 0.9060 |
| `refund_requested` | P(true) 0.9730 | P(true) 0.9770 |
| input tokens | 348 | 236 |

**The token counts are the most important row.** Laya reads the hidden state at each `[MASK]`
marker and emits one logit per option. If a marker position is off by one, nothing crashes. The
model silently answers a different question, with confidence that looks plausible. Matching 348 and
236 exactly shows that all of these agree with Python:

- the tokenizer
- the three JSON serialization dialects
- the marker layout
- the collator

The probabilities alone would not prove that.

The two cases also cover the parts of the pipeline the sample touches along the way:

- dictionary state serialization
- dictionary-valued choice descriptions
- a score legend
- padded marker columns. The graph has `TopK(k=2)` built in, so narrow marker axes are widened, and
  the padding must not leak into the distribution.
- non-Latin script, through the tokenizer's byte fallback

### If you change the sample

Editing the states or questions in [`Program.cs`](Program.cs) invalidates both golden cases. Update
`laya-dotnet/tools/sample_cases.py` to match, then record them again:

```bash
# from the repository root
python laya-dotnet/tools/dump_golden.py --onnx <artifact dir>
dotnet test --solution laya-dotnet/Laya.slnx
```

Copy the literals rather than retyping them. A hand-transcribed Devanagari escape once put U+0915
U+094B where the sample had U+091A. The test then compared two different sentences, and the
mismatch looked convincingly like a parity bug in the port.

The full suite needs `LAYA_ONNX_DIR` set to an absolute artifact path. Without it, the tiers that
depend on the artifact are skipped rather than failed.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Hugging Face has no 'model.onnx' ... (404)` | `--download` can't work yet, because the repository ships safetensors only. Use `--model-dir`. |
| `401` / `403` from Hugging Face | The repository is gated or private. Set `HF_TOKEN`. |
| `Laya artifact directory does not exist`, showing a resolved path you never typed | Unquoted backslashes in a POSIX shell. Compare the `Requested` and `Resolved` lines it prints. |
| `Missing artifact file: ...\model.onnx.data` | The 1.29 GB sidecar was left behind when the export was copied. It must be next to `model.onnx`. |
| `Unrecognised argument: ...` (exit 2) | Only `--model-dir`, `--checkpoint`, `--download` and `-h`/`--help` are accepted. `--model-root` is a `Laya.Routing` flag; here, pass one checkpoint's folder with `--model-dir`. Unknown arguments are rejected rather than ignored, so a stray path can't silently redirect the run. |
| Devanagari prints as `?` or boxes | The console code page or font can't display it. The answers aren't affected. Run `chcp 65001` or use Windows Terminal. |
| First call takes about 1 s | ORT session warm-up on a 322M-parameter encoder. Create one `LayaEngine` and share it. |
