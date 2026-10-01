# Fine-tuning Laya on your own decisions

On the typed-decisions benchmark the base checkpoints score near chance zero-shot — 0.36 and
0.35 against a 0.318 random baseline — while the fine-tuned checkpoint reaches **0.766** on the
same 2,000 decisions, above TypeSafe Jev's published 0.727 and above the 0.735 teacher
self-agreement ceiling. Fine-tuning is where most of the value is, and the public
[fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
runs the whole loop on Kaggle's free 2xT4 GPUs: build the dataset, train with RLCD, fit
calibration temperatures, evaluate, and push the result to the Hub. This page walks that
notebook and points at the parts that stay load-bearing when you swap the data for your own.

The other worked example — a browser-agent decision head on a single 16 GB GPU with no paid API
— is at [Fine-tuning Laya as a browser-agent decision head](finetune_browser_agent.md).

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
