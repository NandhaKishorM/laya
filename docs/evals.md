# Evaluation harness

`laya.evals` turns a labelled dataset into a repeatable score, and a baseline into a
pass/fail gate, so a quality change is a reviewable diff instead of a hand-check.

The metric math and the dataset parser are pure Python plus numpy and never import torch, so
they run with no weights. Running a dataset against a checkpoint needs the checkpoint and
takes its normal load time.

## Quickstart

```bash
# check the format without a model
laya-evals validate research/evals/fixture.jsonl

# score a labelled set on one checkpoint, with thresholds and a baseline
laya-evals run data.jsonl --model english --device cpu \
    --min-accuracy 0.8 --max-ece 0.05 --slice language \
    --json report.json --markdown report.md

# compare a saved report to a baseline
laya-evals compare report.json --baseline baseline.json --tolerance choice_accuracy=0.02
```

`laya eval ...` is the same thing through the main CLI, so `laya eval validate data.jsonl`
works too.

Exit codes: `0` on success, `1` when a threshold or a baseline tolerance fails, `2` on a
usage error. `run` prints the overall metrics and any requested slices to stdout, and writes
the full report and a Markdown summary when `--json` / `--markdown` are given.

## Evaluating an ONNX export

`run --onnx PATH` scores an exported ONNX model through `ONNXAgent` instead of the torch
Router, so an ONNX deployment (including an INT8 copy from `scripts/export_onnx.py --quantize`)
gets gated by the same thresholds and baselines as the torch path:

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
laya-evals run data.jsonl --onnx laya.int8.onnx --max-ece 0.05
```

`--model` names the checkpoint the export came from — a Hub id or local path, not a Router
short name like `english`, since there is no Router on this path (default
`convaiinnovations/laya`). Its config and tokenizer are loaded from there. The agent serves one checkpoint, so a dataset row
whose `model` field names a different one fails with a clear error rather than being silently
answered by the wrong model; `--device` does not apply. `--batch-size` uses the agent's batch
API when it has one and falls back to one call per state otherwise. The report's `config` block
records the `onnx` path.

Measured on `research/evals/fixture.jsonl` (12 labelled rows, English checkpoint, CPU):

| runner | choice_acc | noul_acc | score_mae | ece | mean_conf | p50 ms |
|---|---|---|---|---|---|---|
| torch Router | 0.75 | 1.00 | 1.3418 | 0.1596 | 0.7304 | 116.8 |
| `--onnx` fp32 | 0.75 | 1.00 | 1.3418 | 0.1596 | 0.7304 | 66.3 |
| `--onnx` int8 | 0.75 | 1.00 | 1.3512 | 0.1658 | 0.7304 | 46.3 |

The fp32 export reproduces the torch numbers exactly, and the quantized copy moves `score_mae`
by 0.009 and `ece` by 0.006 — the kind of drift `compare --tolerance` is meant to gate.

## Dataset format

One JSON object per line (JSONL). Blank lines and lines starting with `#` are ignored.

| field | required | meaning |
|---|---|---|
| `state` | yes | text, email, ticket or JSON document to decide on |
| `questions` | yes | a Laya question dict, exactly as `Router.predict` accepts |
| `expected` | yes | ground truth keyed by question id: a label for `choice`, a number for `score`, `true`/`false` for `noul` |
| `tags` | no | strings to slice by |
| `language` | no | a code to slice by |
| `model` | no | force a checkpoint for this row; `--model` overrides it |

`research/evals/dataset.template.jsonl` has a commented example.

## Metrics

Each metric is computed per answer where it applies and aggregated over the dataset:

| metric | applies to | meaning |
|---|---|---|
| `choice_accuracy` | `choice` | fraction whose chosen label matches |
| `noul_accuracy` | `noul` | fraction whose boolean (probability >= 0.5) matches |
| `score_mae` | `score` | mean absolute error |
| `score_within_<tol>` | `score` | fraction within an absolute tolerance |
| `ece` | any answer with a confidence | expected calibration error, 15 bins, computed on `answer["answer_confidence"]`, the calibrated probability Laya reports on every answer type |
| `mean_confidence` | any answer with a confidence | mean reported `answer["answer_confidence"]` |
| `latency_p50_ms`, `latency_p95_ms` | per request | wall time, informational |

Add `ScoreWithin(0.25)` to the evaluator list for a tolerance metric; the default set is
`choice_accuracy`, `noul_accuracy`, `score_mae`, `mean_confidence`, plus `ece`.

## Slices

`compare` and `run` report overall numbers and, for `--slice language|model|qid|tag`, the same
metrics per slice value, so a regression in one language or one question is visible without
reading the aggregate.

## Baseline and CI gate

- Keep the dataset, a baseline report (`--json` output you have reviewed), and the tolerances
  together, committed, so a change is a reviewable diff. `--tolerance METRIC=VALUE` is the
  maximum absolute drift allowed for that metric.
- `laya-evals run ... --baseline baseline.json --tolerance ...` exits non-zero on drift, so it
  drops into CI unchanged. `laya.evals.EvalReport.compare` and `assert_regression` expose the
  same logic for tests.

Two CI surfaces use this:

- a weight-free job in `.github/workflows/ci.yml` runs `tests/test_evals.py` and
  `tests/test_evals_api.py`, so metric math, dataset parsing and the CLI are covered on every
  PR without downloading a checkpoint;
- `.github/workflows/evals.yml` runs weekly, before a release and on demand: it evaluates the
  English checkpoint on the MASSIVE English suite and compares to
  `research/results/eval_english_51_languages.json` with the tolerances in
  `research/evals/thresholds.json`. It uploads the report as an artifact and does not block a
  PR.

The harness is deterministic for a fixed checkpoint revision, so a report is reproducible.
`run` records the dataset, model and device in the report's `config` block.

## Adding the real labelled set

Drop a JSONL in `research/evals/` and a reviewed baseline beside it, then point a workflow (or
`research/evals/check_regression.py`) at both. The format is the same as the fixture; nothing in
the harness knows about MASSIVE.
