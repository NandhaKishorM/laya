# Laya.Benchmark

A small console app that measures the Laya .NET SDK on one checkpoint: how long the first call takes, how fast later calls are, and how much memory the process uses. It also records the answers, so they can be compared with the Python SDK.

It pairs with [`tools/bench_python.py`](../../tools/bench_python.py). Both use the same inputs, so their numbers and answers can be compared directly.

## What it measures

| Metric | How |
|---|---|
| Cold latency | `LayaEngine.Create` plus the first `Predict` call, in ms. This includes loading the model and ONNX Runtime's first-run setup. |
| Warm latency | Median and p95 over 30 timed `Predict` calls, after 5 warm-up calls. The calls alternate between the two cases. |
| Peak memory | `Process.PeakWorkingSet64`, in MB, read after the warm runs |
| Answers | Each question's answer for each case: choice, score or yes/no probability, plus the probabilities, confidence and act probability |

ONNX Runtime runs on the CPU with its default thread settings.

## Inputs

There are two cases, both asked the same four questions. The inputs match `laya-dotnet/tools/sample_cases.py` and the `Laya.Sample` app.

| Case | Input |
|---|---|
| `sample_app_english` | A billing email about a duplicate charge that asks for a refund and threatens to cancel |
| `sample_app_hindi` | A short Hindi message about being billed twice for March |

The four questions:
- `department`: choice of billing, technical, sales or other
- `urgency`: score from 0 to 2
- `churn_risk`: yes/no
- `refund_requested`: yes/no

## Prerequisites

- .NET 10 SDK
- An exported ONNX folder for the checkpoint you want to test. See [`MODELS.md`](../../MODELS.md) for how to export them with `laya-dotnet/tools/export_onnx.py`.

## Usage

Run commands from `laya-dotnet`, and always build with `-c Release`: a Debug build gives misleading timings.

```powershell
# Easiest: point LAYA_ONNX_ROOT at the folder that holds all three checkpoints
$env:LAYA_ONNX_ROOT = "C:/models/laya"

dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint multilingual
dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint english
dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint typed-decisions

# Or give one checkpoint's folder directly
dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint english --model-dir C:/models/laya/english
```

| Argument | Meaning |
|---|---|
| `--checkpoint` | `multilingual` (default), `english` or `typed-decisions` |
| `--model-dir` | Folder containing `model.onnx`, `model.onnx.data`, the tokenizer files and `rl_agent_config.json`. If you leave it out, the app uses `$LAYA_ONNX_ROOT/<checkpoint>` when that variable is set, and otherwise the SDK's normal resolution order. |

`--model-dir` takes one checkpoint's folder (`onnx/english`), not the export root (`onnx/`). To
point at the root, set `LAYA_ONNX_ROOT` as shown above. There is no `--model-root` flag here;
that one belongs to [`Laya.Routing`](../Laya.Routing/README.md) and is rejected with
`Unrecognised argument`.

From this sample's own directory (`laya-dotnet/samples/Laya.Benchmark`), leave out `--project`.
Keep the bare `--`: it separates `dotnet run`'s options from the app's.

```powershell
dotnet run -c Release -- --checkpoint english --model-dir C:/models/laya/english
```

Progress lines go to stderr. The results are printed to stdout as one JSON object, so you can redirect them to a file:

```powershell
dotnet run --project samples/Laya.Benchmark -c Release -- --checkpoint english > bench_dotnet_english.json
```

Exit codes:
- `0`: success
- `1`: the model files weren't found
- `2`: bad arguments

## Output

```jsonc
{
  "checkpoint": "english",            // TypedDecisions is written as "typeddecisions"
  "runtime": "DotNet + ONNX Runtime",
  "threads": "ORT default",
  "cold_ms": 0.0,
  "warm_times_ms": [ /* 30 sorted timings */ ],
  "warm_median_ms": 0.0,
  "warm_p95_ms": 0.0,
  "n_warmup": 5,
  "n_warm": 30,
  "peak_wset_mb": 0.0,
  "answers": {
    "sample_app_english": {
      "department":       { "type": "choice", "choice": "...", "probabilities": { }, "confidence": 0.0, "act_probability": 0.0 },
      "urgency":          { "type": "score",  "score": 0.0,    "probabilities": { }, "confidence": 0.0, "act_probability": 0.0 },
      "churn_risk":       { "type": "noul",   "noul": 0.0,     "confidence": 0.0, "act_probability": 0.0 },
      "refund_requested": { "type": "noul",   "noul": 0.0,     "confidence": 0.0, "act_probability": 0.0 }
    },
    "sample_app_hindi": { /* same shape */ }
  }
}
```

## Comparing with Python

Run the Python counterpart with the repo's virtual environment. Set `HF_HOME` if you want the PyTorch weights to download somewhere other than the default cache.

```powershell
$env:HF_HOME = "C:\hf_cache"
.venv/Scripts/python.exe laya-dotnet/tools/bench_python.py --checkpoint all --output bench_results.json
```

Python runs the PyTorch model directly on the CPU. .NET runs the exported ONNX graph through ONNX Runtime. Expect the answers to match, with tiny probability differences from floating-point rounding (the parity tests allow for this), and expect the timings to differ because the runtimes differ.

## Notes

- **Where the project sits:** it isn't part of `Laya.slnx`, so `dotnet build` / `dotnet test` on the solution skip it. It's a measurement tool, not a shipped sample.
- **Running more than one:** run one checkpoint at a time. The ModernBERT-large checkpoints (english, typed-decisions) are about 1.7 GB each on disk.
- **When to measure:** cold latency depends on disk and OS file cache. Run twice and report the second cold number if you want warm file-cache timings.
