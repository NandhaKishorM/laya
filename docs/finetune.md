# Fine-tuning Laya on your own decisions

On the typed-decisions benchmark the base checkpoints score near chance zero-shot — 0.36 and
0.35 against a 0.318 random baseline — while the fine-tuned checkpoint reaches **0.766** on the
same 2,000 decisions, above TypeSafe Jev's published 0.727 and above the 0.735 teacher
self-agreement ceiling. Fine-tuning is where most of the value is, and the public
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
runs the whole loop on Kaggle's free 2xT4 GPUs: build the dataset, train with RLCD, fit
calibration temperatures, evaluate, and push the result to the Hub. This page walks that
notebook and points at the parts that stay load-bearing when you swap the data for your own.

The same loop is in the package. `laya-train` runs it on one device from a CSV or a JSONL of
labelled decisions in one command, and `laya.train.finetune` does the same from Python; both are
described first below, before the notebook.

The other worked example — a browser-agent decision head on a single 16 GB GPU with no paid API
— is at [Fine-tuning Laya as a browser-agent decision head](finetune_browser_agent.md).

## Fine-tune with `laya-train`

`laya-train` (also `laya train`) builds the training items, fine-tunes, fits calibration
temperatures on a slice held out before training, and saves a checkpoint `laya.load` opens. It
runs on one device: CUDA, MPS or CPU.

### From a CSV

One row per decision, with the text in one column and the correct label in another:

```bash
laya-train --data tickets.csv --text-column body --label-column department \
           --base english --out ./tickets-checkpoint
```

The label column becomes a choice question over its distinct values, in the order they first
appear, with the id given by `--question-id` (default `label`) and the instructions given by
`--instructions`. The question is saved as `questions.json` beside the checkpoint; ask the
fine-tuned model the same question at inference:

```python
import json, laya

agent = laya.load("./tickets-checkpoint")
questions = json.load(open("./tickets-checkpoint/questions.json"))
print(agent.predict("We were billed twice for March.", questions)["answers"])
```

Files saved from Excel with a byte-order mark are read as usual. Rows with an empty text or label
are skipped and counted.

### From JSONL

Each line is one case, and a case may ask several questions of any type. Two label shapes are
read, and a file may mix them:

```json
{"state": "We were billed twice for March.",
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                              "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}},
               "urgent": {"type": "noul", "instructions": "Does this need an answer today?"}},
 "expected": {"department": "billing", "urgent": true}}
```

- `expected` holds one answer per question, the format `laya-evals` reads, so the same file can
  train a checkpoint and evaluate it: a choice label, `true`/`false` for a noul, and a level for a
  score (a fractional level such as `1.5` splits its weight between levels 1 and 2).
  `--label-smoothing 0.1` moves a tenth of each answer's weight to the other options.
- `gold` holds a teacher's probabilities per question, the notebook's schema:
  `{"department": {"probabilities": {"billing": 0.9, "technical": 0.1}}}`. Noul probabilities are
  keyed `"false"`/`"true"` and score probabilities by the level index as a string.

### Check the data before a long run

```bash
laya-train --data tickets.csv --text-column body --label-column department --dry-run
```

`--dry-run` builds the training items with the tokenizer and config of `--base`, without loading
its weights or training, and prints how many rows were read, how many items they produced, and how many questions
were skipped and why:

| reason | the question was skipped because |
|---|---|
| `empty_text`, `empty_label` | the row had no text, or the question no answer |
| `invalid_question`, `invalid_target` | the question or its answer could not be read |
| `options_collapsed` | the head budget left two options with the same tokens (#538) |
| `options_beyond_max_len` | the question's options do not fit in `max_len` |

### What a run writes

`--out` gets the layout `laya.load` reads (`model.safetensors`, `encoder/`, `tokenizer/`,
`rl_agent_config.json`), plus `questions.json` and a `checkpoint_latest/` rewritten after every
epoch, so a crash costs at most an epoch.

The config records the `max_len` and `head_max_len` the run used, the training settings under
`training.laya_train`, and a calibration report under `training.laya_train_calibration`: per
question type, how many calibration items the temperature rests on, their accuracy and mean
confidence, and what to doubt. The same issues are printed as warnings during the run:

```
laya.train: choice calibration: not fitted: 2 calibration items, fewer than 10, so the temperature stays 1.0
```

A warning like that means the checkpoint's confidences were not calibrated; don't gate on
`min_confidence` with it until a run with more data fits them.

### Options

| flag | default | what it changes |
|---|---|---|
| `--base` | `convaiinnovations/laya` | checkpoint to start from: a directory, a built-in name (`english`, `multilingual`, `typed-decisions`) or a Hub repo id |
| `--loss` | `rlcd` | `rlcd` is the notebook's objective; `soft-ce` trains on the soft cross-entropy alone (#741) |
| `--shuffle-options` | off | re-encodes choice questions with a random option order every epoch, with the answer moved to match |
| `--label-smoothing` | `0.0` | weight moved off single answers |
| `--freeze-encoder` | off | trains the decision head only; fits a small GPU or a CPU, but gains much less than a full fine-tune |
| `--epochs`, `--micro-batch`, `--grad-accum` | 4, 8, 8 | the notebook's 4 epochs and effective batch of 64 |
| `--encoder-lr`, `--head-lr` | 2.5e-5, 1e-4 | learning rates |
| `--max-len`, `--head-max-len` | the base checkpoint's | token budgets; the notebook uses 1024 and 256 |
| `--device`, `--seed` | `auto`, 0 | |

Mixed precision is on for CUDA. On a GPU where fp16 runs slower than fp32 (the GTX 16xx cards, for
example), set `amp=False` in `TrainConfig` from Python.

### What the options are worth

Full fine-tunes of `convaiinnovations/laya` on one T4 per run, with the notebook's settings, from
[#887](https://github.com/NandhaKishorM/laya/issues/887).

On typed-decisions (6,000 training decisions, all 2,000 test decisions, one seed):

| run | accuracy | ECE | choice answers that change with the options reversed |
|---|---|---|---|
| base checkpoint | 0.362 | 0.174 | 0.357 |
| `--loss rlcd` (the default) | 0.7715 | 0.150 | 0.065 |
| `--loss soft-ce` | 0.790 | 0.157 | 0.078 |
| `--loss soft-ce --shuffle-options` | 0.7875 | 0.159 | 0.062 |

On MASSIVE intent in English, with 20 options per question (11,514 training rows, all 2,974 test
cases, each also answered in 3 shuffled orders; `--loss soft-ce`, two seeds per row):

| run | accuracy | answers that change when the options are reordered |
|---|---|---|
| base checkpoint | 0.730 | 0.187 |
| without `--shuffle-options` | 0.936 | 0.026 |
| with `--shuffle-options` | 0.941 | 0.019 |

With 4 options, fine-tuning alone already makes answers stable under reordering, and shuffling
changes nothing measurable. With 20 options it raises accuracy by 0.55 points and cuts the answers
that change by about a quarter, both significant over the two seeds, for about 14% more training
time. Use it when questions have many options.

## Fine-tuning from Python

`laya.train.finetune` is what `laya-train` calls:

```python
import laya
from laya.train import TrainConfig, finetune

report = finetune("train.jsonl", "english", "./my-checkpoint",
                  TrainConfig(loss="soft-ce", shuffle_options=("choice",)))
print(report["train_items"], report["skipped"], report["calibration"])

agent = laya.load("./my-checkpoint")
```

`TrainConfig()` has the defaults in the table above, plus the settings the command does not
expose: `amp` and `gradient_checkpointing` (on for CUDA by default), the calibration slice
(`calib_max=400` items or `calib_frac=0.1`, whichever is smaller), and `option_layout`. The
returned report has `train_items`, `calibration_items`, `skipped`, `epoch_loss`, the fitted
`temperature` and `temperature_by_options`, and the `calibration` report. For a data pipeline of
your own, `train_model` and `calibration_records` are the two halves `finetune` is built from; the
[API reference](reference/train.md) lists them all.

## What the notebook does, in order

| # | step | what happens |
|---|---|---|
| 1 | Environment | asserts both T4 GPUs are visible and allocated |
| 2 | Install | `laya`, `transformers`, `datasets` and the training dependencies |
| 3 | Preprocess | the 1,200 training cases (6,000 typed decisions) become tokenized items with soft targets, written to disk for both DDP ranks |
| 4 | Train | `train_ddp.py` under `torchrun --nproc_per_node=2`, four epochs |
| 5 | Calibrate | one temperature per type, fitted on a slice held out before training (inside the training script, after the last epoch) |
| 6 | Evaluate | the official `test` split answered by the fine-tuned checkpoint — 400 cases, 2,000 decisions — with per-case latency |
| 7 | Metrics | accuracy, soft accuracy, Brier, ECE, score MAE, within-one-level, KL/TV and latency percentiles; a head-to-head table against Jev and the teacher ceiling |
| 8 | Publish | (optional) a model card built from the run's own numbers, folder uploaded to the Hub |
| 9 | Report | `benchmark_report.json` with the metrics table and per-workflow accuracy |

Kaggle settings: **Accelerator** `GPU T4 x2`, **Internet** `On`. Outputs land in
`/kaggle/working/laya_finetuned_typed_decisions`.

## The training recipe

RLCD trains on the benchmark's **gold distributions**, not on hard labels: every item carries
the probability the teacher assigned to each option, and both halves of the loss read that
target —

- a **policy-gradient term** over sampled noisy logit projections (GRPO-style: four samples per
  item, exploration noise annealed 0.4 → 0.1), rewarded by proper scoring rules (spherical
  0.75, ranked probability 1.0);
- a full-weight **soft cross-entropy** term against the same distribution.

The knobs the notebook sets for a 16 GB card:

| | |
|---|---|
| epochs | 4 |
| effective batch | 64 sequences (8 per micro-batch, 2 GPUs, 4 accumulation steps) |
| learning rates | encoder 2.5e-5, head 1e-4 — AdamW, cosine schedule |
| memory | fp16 autocast, gradient checkpointing on the encoder and the head, gradient-norm clip 1.0 |
| sequence budget | `max_len` 1024, `head_max_len` 256, `max_tokens_per_batch` 4096 |

Runtime on 2xT4 is minutes for the demo and hours for real data: about 4–6 minutes for the
demo's 6,000 decisions, and roughly 4–5 hours for four epochs over ~30k questions.

To point it at your data, replace the two `load_dataset` calls and keep the row schema: each
case carries `state`, `questions` and `gold` (the teacher probabilities per question), and the
preprocessor turns them into items. The question types are `choice`, `score` and `noul`;
anything you can express with them over a state is fair game.

## Calibration is part of the run

This is the step most likely to be dropped when copying the loop, and it is load-bearing the
moment anyone gates on confidence.

The notebook takes a **calibration slice out of the training data before sharding it across
ranks** (up to 400 items, or 10%, at a fixed seed, identical on every rank). Fitting
temperatures on items the run has already trained on measures the fit rather than the
calibration — the model is near-certain and near-correct on them, so the optimiser has nothing
to soften and returns a degenerate scale.

After the last epoch, rank 0 fits **one temperature per question type** (`choice`, `score`,
`noul`) by LBFGS on the log-temperature, clamped to `[0.1, 10]` (`1.0` for a slice under ten
items, `1.2` if the fit raises). The values go into `rl_agent_config.json` as `temperature`,
and the notebook **removes any inherited `temperature_by_options`** in the same write: those
old bucket values take precedence at inference and would silently mask the new fit.

Temperature scaling leaves the argmax — and accuracy — unchanged; what moves is the
confidence. The checkpoints as shipped are over-confident, so fit before relying on any
threshold, and evaluate the result on held-out data before claiming an improvement. The
config-persistence regression runs without downloads or training:

```bash
python tests/test_calibration_persistence.py
```

## Evaluating before you trust it

The evaluation is a full pass over the official test split: 400 cases, 2,000 decisions across
Agent Trace Observability, Customer Service, Invoice Processing and Security Incidents. It
computes accuracy, soft accuracy, Brier, ECE (via `laya.common.ece_score`), score MAE,
within-one-level and latency percentiles, then builds a head-to-head table whose reference rows
are fixed:

| model | kind | accuracy | ECE |
|---|---|---|---|
| TypeSafe Jev 1.13.0 | general | 0.727 | 0.144 |
| ModernBERT-base (149M) | specialist | 0.646 | 0.179 |
| Teacher Self-Agreement | ceiling | 0.735 | — |
| Laya (published checkpoint) | fine-tuned | 0.766 | — |

The Laya row of your own run is computed the same way — the notebook rebuilds the table from
the run's own numbers. Two habits worth copying: keep the slices you care about (a language, a
workflow) inside held-out data, and report calibration next to accuracy, because the training
signal is a distribution, not just a label. When you have numbers, a post in the repository's
[Discussions](https://github.com/NandhaKishorM/laya/discussions) is the place to share them;
benchmarks and known limits live in `BENCHMARKS.md` at the repository root.

## Pushing to the Hub

The publish cell is the loop's last mile, and it is deliberately boring:

1. Put a write `HF_TOKEN` in Kaggle (Add-ons → Secrets). The cell raises with the exact
   instructions if it is missing.
2. Set the destination repo — the shipped cell defaults to a name in the project's own
   namespace, so change it before running.
3. Run it. It writes a model card whose numbers come from this run's comparison table, then
   uploads `model.safetensors`, `encoder/`, `tokenizer/`, `rl_agent_config.json`, the card and
   the benchmark report.

The result loads like any other checkpoint — there is no fine-tuning-specific API:

```python
import laya

agent = laya.load("your-org/your-checkpoint")   # the repo you just pushed
result = agent.predict(state, questions)
```

A rolling `checkpoint_latest/` is overwritten after every epoch, so a Kaggle timeout or OOM
costs one epoch rather than the run.

## What to watch

- **The loop is only as good as the targets.** RLCD imitates a teacher's distribution on your
  questions; collect the teacher confidences before (or alongside) training, and treat their
  quality as the ceiling.
- **The calibration slice is small on purpose.** Up to 400 items or 10% — enough for three
  per-type scalars, not enough to validate against. Hold out your own evaluation data.
- **Your labels must fit the three primitives.** If your decision is not a choice, a scale or a
  yes/no probability, shape it into one first. Two sharp edges are already documented: high
  option counts degrade confidence selection ([#394](https://github.com/NandhaKishorM/laya/issues/394)),
  and forced-choice negation can follow the question over the state ([#377](https://github.com/NandhaKishorM/laya/issues/377)).
- **Ship the config, not just the weights.** The removed `temperature_by_options` is the part
  that silently un-fits a calibration if it survives in a copied config.
