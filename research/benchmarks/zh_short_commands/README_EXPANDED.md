# Frozen 96-case Chinese command diagnostic

This extends the existing 18 fixtures; it does not edit their prompts, labels or
archive. It uses the same seven configurations and records original and repeated
FP32 logits. It is a synthetic diagnostic, not an independently annotated,
held-out or population-level benchmark. No training, threshold tuning, temperature
fitting or prompt selection is performed.

Additional fixtures were drafted with assistant support. On 2026-10-13, the
contributor reviewed all 96 fixtures and expected labels and approved them
unchanged before publication. Review occurred after the initial evaluation;
no labels were changed in response to model predictions. The original manifest
records the pre-evaluation policy and remains unchanged. This single-contributor
review is not an independently annotated population benchmark.

## Frozen policy

Six balanced labels, 16 fixtures each: faster, slower, stop, left, right and none.
The explicit requested final action determines the movement label. Informational
requests and mentions of command words are none, meaning no movement instruction;
they may still be requests to answer a question. Prohibitions without a clear final
action, vague conditional commands and incompatible multiple actions are excluded.

The families are direct, corrective, polite, negation and mixed-language. Their
label compositions differ, so family accuracy differences are not causal estimates.
Similar paraphrases are correlated. Independent multi-annotator validation remains future work.
The SHA-256 manifest was created before inference and must remain unchanged.

## Environment

Python 3.10 is supported by the source repository. The current local environment
uses torch 2.11 CPU, transformers 5.17, four intra-op threads and one inter-op thread.
It was created on D: from an existing Conda environment using
`python -m venv --system-site-packages .venv`. Local installations of transformers
and its dependencies shadow the base packages without changing that environment.
Unrelated inherited PDF/document packages have dependency conflicts; they are not
used by this diagnostic. A fresh environment avoids those unrelated conflicts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r research/benchmarks/zh_short_commands/requirements-expanded.txt
```

The fresh setup is a reproducibility recipe, not a separately validated installation.
The runnable environment depends on the retained D: Conda installation, not Codex.

## Model

Download only the runtime files for `convaiinnovations/laya-multilingual`, revision
`1720e3e3357cfe1e281542e223f8273b0890ca34`, into `models/laya-multilingual`:
`model.safetensors`, `rl_agent_config.json`, `encoder/*` and `tokenizer/*`.
The expected weight SHA-256 is
`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.
All inputs are short; the checkpoint's default token budgets apply.

## Run

From the repository root on Windows PowerShell:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:USE_TF="0"
.\.venv\Scripts\python.exe -m unittest discover -s research/benchmarks/zh_short_commands/tests -v
.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --device cpu --out research/benchmarks/zh_short_commands/expanded_rerun.json
.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/expanded_eval.py --audit research/benchmarks/zh_short_commands/expanded_report.json
.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/expanded_summary.py
.\.venv\Scripts\python.exe research/benchmarks/zh_short_commands/minimal_repro.py
```

The runner refuses to overwrite an existing archive. Audit verifies frozen hashes,
rebuilds requests and gold labels, recomputes decisions and metrics from logits,
and checks that repeat runs have no changed decisions. Offline tests deliberately
corrupt requests, records, metrics and repeats to check that audit refuses them.
The stub pipeline in tests provides no evidence of model performance.

The JSON archive contains all original and repeat decisions, effective calibration,
checkpoint file hashes and environment versions. CSV and the Chinese report are
derived from that audited archive. The SDK parity probe is a small sample at FP32;
it is not an exhaustive backend or precision parity test.

## Recorded results

The committed `expanded_report.json` contains 2,166 primary decisions and the
same number of repeat decisions, with no decision or raw-logit changes. All
original 18-fixture decisions reproduce the existing archive. The 38 sampled SDK
answers agree with the research scorer (maximum probability delta 0.00005).
`minimal_output.json` records the small public-SDK reproduction.

| Additional 96-fixture configuration | Decisions | Accuracy |
|---|---:|---:|
| choice_criteria | 96 | 0.7917 |
| choice_scenario | 96 | 0.7812 |
| choice_json_state | 96 | 0.7917 |
| noul_plain | 384 | 0.6354 |
| noul_criteria | 384 | 0.3776 |
| noul_scenario | 384 | 0.3750 |
| noul_json_state | 384 | 0.3464 |

Choice and noul accuracies measure different task shapes. The noul constant
baselines are 0.3333 (all true), 0.6667 (all false), and 0.8333 (majority per
dimension). These results describe the frozen synthetic fixtures, not general
Chinese-language accuracy.
