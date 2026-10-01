# Chinese alarm-intent evaluation and temperature calibration

This optional research benchmark measures `choice` and `noul` decisions through Laya's
existing local Agent, sequence builder and multilingual checkpoint. E0 uses the checkpoint's
runtime temperatures; E1 fits one bounded scalar temperature per question type on a separate
calibration partition. It does not update model weights or change the public runtime API.

The contribution is deliberately narrower than a general Chinese benchmark: a three-intent
MASSIVE alarm subset, grouping checks, auditable calibration/test separation, and paired
before/after reliability metrics. It complements the broader Chinese evaluation in
[#557](https://github.com/NandhaKishorM/laya/pull/557), short-command prompt ablations in
[#364](https://github.com/NandhaKishorM/laya/pull/364), and the generic evaluation harness in
[#382](https://github.com/NandhaKishorM/laya/pull/382).

## Inspect and recompute the historical result on CPU

[The result table](results/massive_alarm_2026-09-24.md) is generated from
[compact decision evidence](results/massive_alarm_2026-09-24.json). The JSON contains 128
calibration and 186 test records with source IDs, group IDs, labels, option order, raw logits,
actual temperatures and source hashes. Test records retain both runs' raw logits so the verifier
checks that calibration changed only temperature. It includes original file hashes, model
identity, selected dependency versions and numerical summaries. Source text, weights, local
absolute paths, GPU UUIDs and credentials are omitted.

**These are historical measurements from 2026-09-24**, executed on base commit `23a1752`
with the contribution subsequently recorded in `37cc584`. The execution worktree was dirty;
its recorded source-file hashes identify the measured implementation. The results were not
measured against the later upstream rebase or this evaluation-only refactor.

From the repository root, with the project's Python dependencies installed:

```bash
python -m research.benchmarks.zh_reliability.aggregate_massive \
  --artifact research/benchmarks/zh_reliability/results/massive_alarm_2026-09-24.json \
  --report /tmp/massive-alarm-result.md
```

This performs no inference or downloads and needs no CUDA/model checkpoint. It verifies record
integrity, partition/group/source pairing and original temperatures; refits temperatures from
calibration logits; rebuilds probabilities and metrics; recomputes 1,000 paired group bootstrap
draws; and compares all numerical summaries with the saved values. Hashes bind the archived
artifacts; they are not independent authentication of data provenance. Auditing original text
and annotation declarations additionally requires the prepared public dataset.

| Condition | Accuracy | Macro F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| E0 original temperatures | 0.532258 | 0.440840 | 1.383747 | 0.730071 | 0.366488 |
| E1 calibrated temperatures | 0.532258 | 0.440840 | 0.785033 | 0.506898 | 0.197529 |

Accuracy and F1 are unchanged because positive scalar temperature preserves argmax. The NLL
difference is −0.598713, with a group bootstrap 95% interval [−0.865774, −0.369099]. This
supports a calibration result on this subset, not general Chinese capability improvement.
Choice temperature is 3.135504; noul reaches the permitted upper bound of 5.0. Risk/coverage
does not improve at every coverage.

## Prepare public data

Amazon's MASSIVE 1.1 is human-localized virtual-assistant text, licensed CC BY 4.0
([pinned publisher notice](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md)).
It is not naturally occurring Chinese customer-service logs. The derived tasks use the original
`alarm_set`, `alarm_query`, and `alarm_remove` labels: choice selects the operation; noul asks
whether the intent is setting an alarm. No paraphrases or new semantic labels are generated.

```bash
# Optional dependency for the small Parquet download route only.
uv pip install --python .venv/bin/python pyarrow
.venv/bin/python -m research.benchmarks.zh_reliability.prepare_massive \
  --download-parquet --run-dir runs/massive-alarm-data
.venv/bin/python -m research.benchmarks.zh_reliability import-data \
  --data runs/massive-alarm-data/data.jsonl --formal \
  --split-manifest runs/massive-alarm-data/split_manifest.json \
  --run-dir runs/massive-import
```

The Parquet route downloads six Chinese/English files, about 2 MB, from the publisher's
`AmazonScience/massive` repository at conversion revision
`ed58ac423a2f4121720918bf5301577edce4ffd3`. It executes no dataset loading script. Embedded
ClassLabel indices are decoded, and column-oriented annotation lists are transposed without
changing text or labels. Saved manifests identify transport, decoding rules, reader version,
file hashes and pinned attribution/legal text. Decoded source-row hashes are not archive-byte
hashes. Alternatively, `--download` uses the publisher's approximately 40 MB archive and
records its hash; only selected regular members are read. `--source-dir PATH` converts an
already downloaded, hash-checked source directory offline. Every output directory must be new.

The fixed admission policy requires at least three distinct upstream reviewers whose intent
scores all equal 1. It does not imply unanimous grammar/spelling/slot approval. Provenance
retains these upstream judgments and `human_verified=false`; no independent local review is
claimed. Other records remain in `diagnostic.jsonl`, with reasons. No model scores determine
admission, grouping or labels.

Connected groups link identical source IDs, normalized Chinese text and delexicalized English
source templates. Cross-partition groups keep the highest-priority original partition
(test > dev > train); lower-priority members are quarantined. Conflicting-label groups are
quarantined. Calibration uses 20% of the remaining training groups with seed `20260924`.
Grouping is heuristic and can miss semantic dependencies.

The retained 471 source utterances generate 942 correlated decisions. Fixed partition decision
counts are train 522, dev 106, calibration 128 and test 186, with 247/53/62/89 groups.
**This evaluation consumes only calibration and test**; the other source partitions remain
reserved. Test has 93 utterances, each represented by both question types.

## Run a new evaluation

Use a local copy of `convaiinnovations/laya-multilingual` revision
`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, including the recorded `zh_lineage.json` identifying
that original revision. The script verifies fixed data/group counts and records paths/hashes
before predictions. Set `LAYA_ZH_PYTHON` if the interpreter is not `.venv/bin/python`.

On a clean clone, explicitly download that pinned revision and record its original-checkpoint
lineage. This downloads only runtime files to the Hugging Face cache, copies them into a new
local model directory, and performs no inference. It does not fall back to `main`:

```bash
.venv/bin/python - <<'PY_DOWNLOAD'
from pathlib import Path
from research.benchmarks.zh_reliability.data import atomic_json
from research.benchmarks.zh_reliability.runtime import resolve_model

revision = 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'
work = Path('runs/massive-model-download')
work.mkdir(parents=True)  # Refuse an existing output directory.
model, identity = resolve_model('convaiinnovations/laya-multilingual', revision, False, work)
if identity['revision'] != revision:
    raise ValueError('download resolved to an unexpected revision')
atomic_json(model / 'zh_lineage.json', {
    'model_id': identity['model_id'], 'base_revision': revision,
    'trained': False, 'weights_hash': identity['files']['model.safetensors']})
print(model)
PY_DOWNLOAD
```

The resulting path is `runs/massive-model-download/model_input`:

```bash
bash research/benchmarks/zh_reliability/reproduce_massive.sh \
  runs/new-massive-calibration runs/massive-model-download/model_input \
  runs/massive-alarm-data/data.jsonl runs/massive-alarm-data/split_manifest.json
```

The fixed recipe runs only `calibrate`, baseline `eval`, calibrated `eval`, and CPU aggregation.
It defaults to process-visible `cuda:0`, FP16, max_len=512, and head_max_len=192. Set
`LAYA_ZH_DEVICE` to a visible device selector, for example `LAYA_ZH_DEVICE="RTX 2080 Ti"`,
to use another device. The selected value is recorded before predictions in the generated
protocol; the historical measurements used RTX 2080 Ti. Selectors refer to visible CUDA devices; no device mask or driver is changed. Downloading models is
not implicit. A failed stage returns nonzero and records its stage in a FAILED manifest.
Run directories hold local predictions, environment, split, calibration and model identities;
private tokenizer copies protect the input checkpoint from compatibility patches.

Standalone commands are available with `python -m research.benchmarks.zh_reliability --help`.
`eval` and `calibrate` require consistent data, `--formal`, `--split-manifest`, model, dtype and
length settings. Calibration defaults to at least 30 samples per type. Artifact validation
rejects a different model/data/split or unapproved smoke-only fit. A new completed recipe can
be checked and its report rebuilt without GPU inference:

```bash
python -m research.benchmarks.zh_reliability.aggregate_massive runs/new-massive-calibration
```

The short agent-authored examples in `fixtures.jsonl` remain unverified regression diagnostics;
passing fixture tests is not a measured public-data result.

## Definitions and limits

- Only valid decision logits are used, never action outputs or rounded public probabilities.
  Noul option order is `[false,true]`; confidence is max(p). Existing temperatures are not
  applied twice. The bounded fit minimizes calibration NLL in the runtime range [0.5,5.0].
- NLL uses float64 log-softmax. Brier is the mean sum of squared class errors. Macro F1 averages
  task-local macro F1 over the full option set. ECE has 15 equal bins, the last including 1.
  Risk/coverage breaks confidence ties by lexical sample ID. Type/task/truncation slices and
  official-Agent probability parity are retained by standalone evaluation.
- Bootstrap resamples the 89 test groups together across both conditions and question types.
  Its 1,000 draws use seed `20260924`. Intervals condition on this checkpoint and fitted
  temperatures, omit calibration uncertainty, and make no corrected multiple-testing claim.
- This is one task subset and one split/seed. Checkpoint exposure to MASSIVE/SLURP is unknown.
  It does not establish full MASSIVE performance, general Chinese improvement, or unchanged
  behavior in other domains. Action outputs are outside scope.
- Historical test sequences were at most 60 tokens with no truncation. max_len=512 is a limit,
  not a 512-token stress test. Optional timings synchronize CUDA, distinguish loading and
  inference memory, and report single-decision batch p50/p95; GPUs are not exclusively reserved.

Offline checks use `python -m unittest discover -s tests -p 'test_zh_*.py' -v`. The archive
recomputation command above additionally verifies the submitted numerical evidence. Repository
Ruff and compileall gates apply; no hosted inference service or new runtime dependency is added.
