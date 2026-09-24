# Chinese decision reliability on one GPU at a time

This opt-in benchmark extends Laya's existing local Agent, sequence builder, collator and
DecisionModel. It does not change the runtime API. Only `choice` and `noul` are supported.
The older two-T4 notebook uses distributed training and a different objective; this runner
loads the actual multilingual architecture and uses supervised, frozen-encoder head tuning.

The 80 short Chinese examples in `fixtures.jsonl` are **agent-authored, human_verified=false**.
They are individually inspectable regression fixtures, not an independently annotated standard
benchmark. They cannot establish general Chinese accuracy or calibration improvements. No
claim is made that a public checkpoint has never seen these kinds of examples. Labels are
fixed before running; mistakes and difficult examples remain in predictions and reports.

## Environment and first run

From the repository root, use an existing project environment with the dependencies in
`pyproject.toml`. `pytest` is optional; the new tests use `unittest`. No FlashAttention,
TileLang, distributed framework, service or external inference API is used.

The environment used during this contribution is `.venv/bin/python`. It was created in the
project using Python 3.12 and uv because system Python lacked pip/ensurepip. CUDA PyTorch was
installed only in this new environment; drivers and pre-existing environments were not changed.
Exact versions and execution probes are saved in each `environment.json`.

```bash
set -euo pipefail
PY=.venv/bin/python
$PY -m benchmarks.zh_reliability --help
$PY -m benchmarks.zh_reliability check-env --run-dir runs/zh-env
$PY -m benchmarks.zh_reliability smoke --run-dir runs/zh-offline
```

Every subcommand supports `--help`. Run directories must be new. Default smoke runs the
CPU logic tests (including tiny random BERT training/resume), marks GPU work `NOT_RUN`, and
sets the overall manifest to `PARTIAL`. A passing mock test is never a real Laya result.
The GPU opt-in test additionally accepts `LAYA_ZH_REAL_MODEL=/absolute/local/checkpoint`.

`check-env` records a whitelist of environment variables, Python, OS, dependency versions,
commit/dirty state, CUDA/cuDNN, driver, visible names/UUIDs, VRAM, compute capability, compiled
PyTorch architectures and an actual small CUDA matrix operation. A GPU name alone is not
proof that the binary works. Physical `nvidia-smi` order can differ from PyTorch order.
Selectors match the currently visible devices by name, UUID or explicit process `cuda:N`.
The runner never changes `CUDA_VISIBLE_DEVICES`; if changing it externally, set it **before
starting a new Python process**. It never makes a masked-out GPU available.

## Data and fixed partitions

Each JSONL row has `id`, `group_id`, `task_id`, string `state`, `question`, hard semantic
`label`, string `tags` and `provenance`. Choice criteria are an insertion-ordered mapping
from stable option IDs to descriptions. Noul criteria use exactly `false` and `true`;
custom display labels never change the semantic order `[false, true]` or `P(true)`.
A task ID identifies one instruction/criteria definition. Optional `relation` contains a
`pair_id` and `kind` (`preserve` or `flip`), with exactly two members in the same group.

```json
{"id":"refund_001","group_id":"refund_case_001","task_id":"explicit_refund","state":"不要退款，我只是想查询进度。","question":{"type":"noul","instructions":"用户是否明确提出当前退款请求？","criteria":{"false":"没有当前退款请求","true":"明确要求当前退款"}},"label":"false","tags":["negation"],"provenance":{"kind":"agent_authored_fixture","human_verified":false}}
```

The supplied 40 independently authored source situations each have two variants in one
leakage group. They cover ordinary speech, negation, conditions, double negation, colloquial
language/typos, code switching, distractors and option order. Choice variants reverse the
rendered option order. Shared task instructions are not a source-template family. If adding
instances of the same source/template, give them the same group; `provenance.template_family`
is also checked for cross-group leakage. The tool cannot infer undisclosed common origins:
source grouping needs a human data audit before formal use.

Seed `20260924` gives 48/8/12/12 states in train/dev/calibration/test (24/4/6/6 groups).
Assignment rounds group boundaries, never splits groups to hit percentages. Content hashes
include option rendering order. `--split-manifest` validates and reuses an existing manifest.
Train updates weights, dev selects a checkpoint, calibration fits temperatures, and test is
used only for reporting. Training commands do not evaluate test.

To import user-owned, licensed, verified data, supply its real path:

```bash
PY=.venv/bin/python
DATA=/absolute/path/to/licensed_chinese.jsonl
$PY -m benchmarks.zh_reliability import-data --data "$DATA" --formal \
  --run-dir runs/zh-real-data
```

Formal rows additionally require a fixed `split`, and provenance `source`, `license`,
`annotation_status`, non-fixture `kind`, and `human_verified=true` for locally reviewed data.
The source-reviewed MASSIVE route below instead keeps this false and validates explicit
upstream annotation evidence. For locally reviewed data, the flag is an assertion by
the data provider, not verification of legal provenance or annotator agreement by the tool.
`--formal` enables group-paired bootstrap for E1−E0 and E3−E2 in a combined smoke run, with
1,000 resamples and percentile 95% intervals when at least 20 independent test groups exist.
Smaller data returns `SKIPPED`. Fixtures never receive population confidence claims.

## GPU smoke and reproducible steps

Use an inspected local multilingual directory or an offline HF cache. Downloads are opt-in
(`--download`); only `convaiinnovations/laya-multilingual` runtime files are fetched. Online
loading resolves the requested revision to its SHA first. Local inputs are content-hashed;
a local directory without revision metadata records SHA as unknown, rather than inventing it.
Agent loads a private copy under the run because its tokenizer compatibility patch may write
configuration. Shared caches and base weights are never overwritten.

```bash
PY=.venv/bin/python
MODEL=runs/zh-download/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
$PY -m benchmarks.zh_reliability smoke --gpu --model "$MODEL" \
  --train-device 'TITAN RTX' --device 'RTX 2080 Ti' \
  --max-steps 2 --max-len 256 --head-max-len 192 --run-dir runs/zh-gpu
```

The path above is the local pinned download produced in this workspace. On another machine,
use a local checkpoint or replace `--model "$MODEL"` with `--download` to opt into downloading
only the target model. A new run is required when changing any experiment configuration.
GPU OOM, incompatible kernels and SDK CPU fallback fail the measurement, with nonzero exit
and the stage/reason recorded. Missing CUDA is explicitly `SKIPPED`; no large CPU training
is substituted. Other users' processes are never stopped. Devices run sequentially.

The combined GPU smoke runs E0/E1, at most the requested head training steps on TITAN RTX,
exports and reloads the dev-selected model on the 2080 Ti, runs E2/E3, and measures the base
checkpoint in FP16 and FP32 on both cards. A missing card leaves its phases explicitly NOT_RUN and runs available phases with a PARTIAL
manifest. A failed stage cannot produce a successful full-experiment manifest. Completed
predictions remain available on failure.

The same operations can be run separately with a shared manifest:

```bash
PY=.venv/bin/python
MODEL=runs/zh-download/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
SPLIT=runs/zh-offline/split_manifest.json
$PY -m benchmarks.zh_reliability eval --model "$MODEL" --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 --run-dir runs/zh-e0
$PY -m benchmarks.zh_reliability calibrate --model "$MODEL" --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 \
  --minimum 1 --smoke-only --run-dir runs/zh-cal
$PY -m benchmarks.zh_reliability eval --model "$MODEL" --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 \
  --calibration runs/zh-cal/calibration.json --allow-smoke-calibration --run-dir runs/zh-e1
$PY -m benchmarks.zh_reliability train-head --model "$MODEL" --split-manifest "$SPLIT" \
  --device 'TITAN RTX' --max-len 256 --head-max-len 192 --max-steps 2 --run-dir runs/zh-head
$PY -m benchmarks.zh_reliability resume --model "$MODEL" --split-manifest "$SPLIT" \
  --device 'TITAN RTX' --max-len 256 --head-max-len 192 --max-steps 3 \
  --checkpoint runs/zh-head/checkpoints/step-00002 --run-dir runs/zh-resumed
$PY -m benchmarks.zh_reliability eval --model runs/zh-head/exported --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 --run-dir runs/zh-e2
$PY -m benchmarks.zh_reliability calibrate --model runs/zh-head/exported --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 \
  --minimum 1 --smoke-only --run-dir runs/zh-head-cal
$PY -m benchmarks.zh_reliability eval --model runs/zh-head/exported --split-manifest "$SPLIT" \
  --device 'RTX 2080 Ti' --max-len 256 --head-max-len 192 \
  --calibration runs/zh-head-cal/calibration.json --allow-smoke-calibration --run-dir runs/zh-e3
$PY -m benchmarks.zh_reliability report --source runs/zh-gpu --run-dir runs/zh-report
```

For formal experiments, use `--data "$DATA" --formal` and the real data's split manifest
consistently on every command. Standard sequence length defaults to 512; smoke defaults to
256. Head budget defaults to an explicit 192. Never change these between parity comparisons.
Use `--full --max-steps N` to explicitly authorize a longer head run.
`--checkpoint-every N` evaluates dev and saves a resumable checkpoint every N successful
optimizer updates and at the final step; the default is 1. Choose this schedule before
training: dev selection considers only these candidates. All successful steps are logged;
non-evaluation steps have `dev_nll=null`, and scaler overflows do not advance the schedule.
The interval is part of the resume contract. The multilingual checkpoint consumes about
169 MiB per saved head/optimizer checkpoint, so a 198-step run with interval 11 saves
18 candidates (about 3 GiB) instead of 198 (about 33 GiB), excluding model copies/exports. This does not enable
full-parameter training or any distributed behavior. Do not select N/seeds/settings using test.

## Numerical and training definitions

* E0 retains the actual runtime-clamped checkpoint temperatures, including option and language
  overrides. E1 fits one new scalar for each supported type, without weight updates.
* E2 uses the dev-selected tuned checkpoint at T=1. E3 fits temperatures for that exact checkpoint.
  All fitting consumes **unscaled, unrounded decision logits**, never action logits or public
  rounded JSON probabilities. Existing temperatures are not applied twice.
* Bounded inverse-temperature optimization minimizes calibration NLL in the runtime range
  `[TEMP_MIN, TEMP_MAX]` (currently `[0.5,5.0]`). It reports boundary hits and skipped fits;
  skipped types preserve their original temperature/buckets. Default minimum is 30 samples
  per type. The explicit smoke minimum of 1 only tests the pipeline, not deployment fitness.
* Calibration JSON includes model identity/file hashes, split/data/calibration hashes, IDs,
  sample counts, original/effective values and `lang_temperatures`. Trained type buckets are
  removed to prevent shadowing. Score/other language settings are preserved. Validate with
  `validate_calibration` before passing the returned mapping to Agent's existing
  `lang_temperatures=` argument. The CLI refuses a mismatched model/data or unapproved smoke
  calibration; Agent itself does not inspect benchmark metadata.
* Model forward returns `(decision_logits, action_logits)`. The adapter uses only the first,
  removes padding from probabilities/losses, records actual option order and semantic gold,
  and compares with official Agent results within four-decimal rounding error.
* Confidence means `max(p)` for every type. NLL uses stable float64 log-softmax; Brier is the
  mean **sum** of squared error across valid classes, including the full noul vector. Macro-F1
  is computed within each task using its full option set, then averaged across tasks; an
  absent class gets F1=0. ECE uses 15 bins `[i/15,(i+1)/15)`, including 1 in the last bin.
  Risk/coverage breaks ties by lexical sample ID, independently of gold. Curves are diagnostic.
* Reports include task/type/option-count/phenomenon/truncation slices and counts. Empty groups
  contain null plus a reason. Preserve-pair consistency and semantic-flip behavior are separate;
  both-correct rates are also shown. A changed wrong answer is not necessarily the expected flip.
* Each row has one state and one question/sequence. Micro-batch 1 and accumulation 8 mean up to
  8 states/decision sequences per update. The final short epoch window averages its actual
  size. Frozen encoder stays in eval; only `head`, `type_emb` and `scorer` train. Encoder and
  unsupervised `act_head` have no gradients and are hash-checked unchanged.
* FP32 master parameters, CUDA FP16 autocast and version-compatible GradScaler are used.
  Gradients are unscaled before clipping; finite loss/gradients/parameters are checked.
  Scaler overflow is logged as a skipped update, never an actual optimizer step. Defaults:
  AdamW lr=1e-4, weight_decay=.01, norm cap=1.0. These are starting values, not optimal claims.
* Scheduled immutable checkpoints hold safetensors head weights and local optimizer/scaler,
  RNG, epoch/cursor state. Resume validates model/data/split, lengths, precision, optimizer,
  device and seed; extending the total step budget is allowed explicitly. `weights_only=True`
  is used to load the locally generated recovery file. Do not use untrusted external recovery
  files. Full exported weights and tokenizer/config are directly Agent-loadable. Shared decision
  parameters also affect score outputs, which are outside this experiment and remain unvalidated. The internal
  head delta alone is not an Agent checkpoint.
* CPU tests compare uninterrupted/resumed tiny-model updates exactly. CUDA RNG and data position
  are restored, but cross-version/device/kernel determinism is not promised. FP16/FP32 and
  cross-card probability differences and argmax agreement are reported without invented tolerances.

Timing defaults to 10 warmup and 30 measured **batches**, batch size 1. End-to-end includes
input encoding, transfer, forward and probability postprocessing, excluding downloads,
loading and disk output. Model-stage inputs are already on GPU (H2D/D2H excluded). CUDA
synchronization brackets each sample. Reports give batch p50/p95, state/s, decision/s,
independent sample count, repeat count and actual lengths. Peak allocated/reserved memory
is reset for each range; loading, training (+scheduled dev selection) and inference are distinguished.

## Verification and review boundaries

```bash
set -euo pipefail
.venv/bin/python -m unittest discover -s tests -p 'test_zh_*.py' -v
.venv/bin/python tests/test_router.py
.venv/bin/python tests/test_criteria.py
.venv/bin/python tests/test_hooks.py
.venv/bin/python tests/test_hooks_api.py
.venv/bin/ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
.venv/bin/ruff check benchmarks/zh_reliability tests/test_zh_*.py \
  --select=E9,F63,F7,F82,F401,F811 --line-length=120
.venv/bin/python -m compileall -q laya/ tests/ benchmarks/zh_reliability
```

A review split can land data/metrics/calibration/runtime, the corresponding DataMetrics and
parity tests, CLI eval/calibrate/check-env/report, and documentation first (A). Then add
`training.py`, TinyTraining tests, CLI train-head/resume and the combined GPU smoke (B).
The source currently shares one CLI and one test file, so split those hunks by class/command.
The runtime API and English notebook are unchanged. See `PR_DRAFT.md` for the English
submission draft and recorded verification limitations.

## No private data: prepare the public MASSIVE alarm subset

Use the publisher's **MASSIVE 1.1**, whose dataset and SLURP source text are CC BY 4.0
([pinned publisher notice](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md)).
This is human-localized virtual-assistant text, not naturally occurring Chinese customer-service
logs. The first derived task deliberately covers the original `alarm_set`, `alarm_query` and
`alarm_remove` intents. Choice asks which alarm operation is intended; noul asks whether the
intent is setting an alarm. No refund labels, paraphrases or counterfactuals are manufactured.

```bash
set -euo pipefail
.venv/bin/python -m benchmarks.zh_reliability.prepare_massive --help
# Optional data preparation dependency; not required by Laya or offline unit tests.
uv pip install --python .venv/bin/python pyarrow
.venv/bin/python -m benchmarks.zh_reliability.prepare_massive \
  --download-parquet --run-dir runs/massive-alarm-data
.venv/bin/python -m benchmarks.zh_reliability import-data \
  --data runs/massive-alarm-data/data.jsonl --formal \
  --split-manifest runs/massive-alarm-data/split_manifest.json \
  --run-dir runs/massive-import
```

The Parquet route downloads only six Chinese/English partition files (about 2 MB) from the
publisher's `AmazonScience/massive` Hugging Face repository, fixed at conversion revision
`ed58ac423a2f4121720918bf5301577edce4ffd3`. It executes no dataset loading script and makes
no hosted inference calls. Label indices are decoded with the embedded ClassLabel names;
column-oriented judgments and slot methods are transposed into lists without changing text
or labels. The manifest records every Parquet and decoded JSONL hash, the decoding rule and
reader version. Decoded source-row hashes are **not** original archive-byte hashes.

Alternatively, use `--download` to fetch the publisher's roughly 40 MB archive using only the
Python standard library. Only the Chinese and English JSONL files and dataset license are
extracted; arbitrary archive paths are never extracted. That route records the archive hash.
Both routes save the pinned publisher notice and documentation. The Parquet route saves a
pinned copy of the CC BY 4.0 legal text from Creative Commons, with its URL in the manifest;
it is not represented as the original archive's LICENSE file. Dataset authorization comes
from the publisher notice; the repository code's Apache license does not replace it.

For offline conversion, replace the download option with
`--source-dir /path/to/the/source-directory` containing the files and `source_manifest.json`
from a completed preparation. This verifies saved hashes; it does not independently authenticate
an arbitrary user-supplied manifest. Use a fresh output directory for each conversion.

The annotation policy is fixed before model evaluation: retain at least three distinct
upstream human reviewers whose intent scores are all exactly 1 (confirmed intent). This does
not require unanimous grammar, spelling, slot or language-quality scores. A score
of 2 only means a reasonable interpretation, so it is not strict confirmation. Missing or
contested annotations stay in `diagnostic.jsonl`, with original source rows and exclusion
reasons; they are not silently relabeled. This changes the evaluated distribution, so results
must be described as a **source-reviewed MASSIVE alarm subset**, never full official MASSIVE.

`human_verified=false` and `local_human_review=false` remain on every converted row. Formal
validation accepts this narrowly defined public-source route only with pinned documentation,
source-row hash, original intent and matching upstream judgment evidence. This is structural
validation of the evidence supplied by the pinned converter, not a new human review of the
Chinese task wording or derived mapping. No claim is made that the base model has never seen
MASSIVE or SLURP.

Choice and noul views of the same utterance share a group. Groups also connect normalized
identical Chinese and English source templates with annotated slot values removed. If such a
group spans original partitions, test has priority over dev, then train: the lower-partition
rows are quarantined as diagnostics, rather than leaking the official test template into
training. Conflicting labels for identical normalized Chinese quarantine the affected group.
Eligible official test/dev rows retain membership; calibration comes only from 20% of the
remaining train groups (seed 20260924). Counts need not match the fixture's 60/10/15/15 ratio.
This conservative heuristic grouping can merge distinct templates and cannot discover all
undisclosed semantic families. Overlap quarantine precedes review filtering: a disputed test
member still prevents its shared template from entering train. Counts report original utterances
and derived decisions separately; choice/noul views are correlated, not independent new data.

Generated task prompts are code definitions; source utterances and labels retain their data
license. The output `attribution/` contains the publisher notice, dataset license and source
documentation. `report.md` records all retained and diagnostic counts. This command performs
no model inference, training or calibration. Existing eval/calibrate/train-head commands can
consume `data.jsonl` with `--formal` and its fixed split manifest; the normal calibration
minimum of 30 per type still applies.

Observed preparation in this workspace: 16,521 decoded Chinese source rows; 550 alarm rows;
471 retained utterances and 79 diagnostic rows. No source/model predictions were used for
selection. The derived split contains:

| Split | Source utterances | Groups | Derived decisions |
|---|---:|---:|---:|
| Train | 261 | 247 | 522 |
| Dev | 53 | 53 | 106 |
| Calibration | 64 | 62 | 128 |
| Test | 93 | 89 | 186 |

The live pinned-Parquet download, conversion, `import-data --formal`, and a second offline
conversion completed successfully. Data, diagnostics, source manifest and split manifest were
byte-identical on replay. The generated report is `runs/massive-alarm-data/report.md`; these
counts are data preparation evidence, **not** model evaluation results. Each question type
has 64 calibration rows, exceeding the default 30-row fitting minimum; confidence intervals
still require actual paired predictions, and statistical precision is not implied by passing
the group-count gate.

## Completed public-data experiment

The table below summarizes a completed formal run of this exact alarm subset. The committed
[`massive_protocol.json`](massive_protocol.json), [`reproduce_massive.sh`](reproduce_massive.sh)
and [`aggregate_massive.py`](aggregate_massive.py) provide the fixed settings, seven-stage
execution and verified aggregation. The original machine-specific outputs remain local under
`runs/massive-experiment-v1/`; a fresh clone does not need those outputs. The run used 198 optimizer updates, interval
11 (18 dev candidates), max_len=512/head_max_len=192, seed 20260924, FP16 and the defaults
for AdamW/micro-batch/accumulation. TITAN RTX trained the head; RTX 2080 Ti performed
calibration and test evaluation. Two scaler overflows were skipped, and the selected final
checkpoint passed frozen-parameter and exact same-device reload checks.

| Condition | Accuracy | Task macro F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| E0: Original checkpoint | 0.532258 | 0.440840 | 1.383747 | 0.730071 | 0.366488 |
| E1: Original + calibration | 0.532258 | 0.440840 | 0.785033 | 0.506898 | 0.197529 |
| E2: Head tuning, T=1 | 0.758065 | 0.729674 | 0.595625 | 0.350510 | 0.099407 |
| E3: Head tuning + calibration | 0.758065 | 0.729674 | 0.576033 | 0.332620 | 0.060256 |

The fixed test has 93 source utterances/89 groups/186 decisions. E3−E0 accuracy changed by
22.58 percentage points, with group-paired bootstrap 95% interval
[15.16, 29.67] points (1,000 resamples, fixed seed). The additional
E3−E2 calibration NLL change is -0.019592, interval
[-0.059509, 0.030108], which crosses zero. These are single-seed,
subset-specific estimates conditional on the fixed models, not general Chinese improvement
or uncertainty over training/calibration. No English or other-domain regression evaluation
was performed on this tuned checkpoint.

Actual test sequences were 43–60 tokens without truncation: this run does not establish
512-token stress capacity. E0/E2 p50 latency on RTX 2080 Ti was 52.556/56.141 ms; there is
no measured speedup. See the report for NLL/Brier/ECE, type slices, all paired intervals,
source/model verification and scope-specific peak memory.

To reproduce from a fresh clone, prepare the data using the public-data command above and
use the project's Python environment. Download the **fixed** multilingual revision with only
runtime files; this does not run model inference. The new download directory must not exist:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.zh_reliability.data import atomic_json
from benchmarks.zh_reliability.runtime import MODEL_ID, resolve_model

revision = 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
work = Path('runs/zh-pinned-base')
work.mkdir(parents=True, exist_ok=False)
model, identity = resolve_model(MODEL_ID, revision, False, work)
assert identity['revision'] == revision
atomic_json(model / 'zh_lineage.json', {
    'model_id': MODEL_ID, 'base_revision': identity['revision'], 'trained': False})
print(model)
PY
bash benchmarks/zh_reliability/reproduce_massive.sh \
  runs/new-massive-experiment runs/zh-pinned-base/model_input
```

The script requires the recorded TITAN RTX and RTX 2080 Ti device selectors. It runs base
calibration, head tuning, tuned calibration, then E0/E1/E2/E3 test evaluation and CPU aggregation.
Weights, temperatures and settings are locked before test evaluation. Training uses 198
successful updates with checkpoint interval 11; calibration uses the regular minimum of 30.
The output directory must be new. Each replay writes its own protocol timestamp and hashes
for the actual data, split and model before its first prediction; it does not copy the old
machine's paths, timestamp or data hashes.

Optional positional arguments are `OUTPUT_DIR [MODEL_DIR [DATA_JSONL [SPLIT_JSON]]]`.
Relative paths are interpreted from the repository root. Defaults use the `runs/massive-alarm-data/`
outputs above and the previously downloaded `runs/zh-download/<fixed-revision>` model.
`LAYA_ZH_PYTHON`, `LAYA_ZH_MODEL`, `LAYA_ZH_DATA` and `LAYA_ZH_SPLIT` can override those defaults.
The data must match the fixed subset's split/group counts, and the model lineage must name
the fixed original revision. Artifact hashes may differ with reader/environment versions;
these are bound to the new run rather than substituted into the historical result.

To verify and rebuild a completed replay's report without loading a model:

```bash
.venv/bin/python -m benchmarks.zh_reliability.aggregate_massive runs/new-massive-experiment
```

Aggregation checks every child status, model/data/split/calibration binding and prediction
before recomputing metrics and paired intervals. Generated artifacts remain local and ignored
by Git; no dataset, model weights or original predictions are included in the source tree.
