# Step 4 benchmark: reproduce typed-decisions through `laya-train`

Harness for #887 Step 4. Two evidence surfaces, kept separate:

1. **Typed-decisions reproduction** — the fine-tune is reproduced through the shipped
   `laya-train` CLI and scored on the official test split: **accuracy, ECE, Brier**. This is the
   surface that verifies acceptance criterion 1.
2. **20-option order stability** — measured on **`massive_intent.en` at 20 options**, not on
   typed-decisions (whose `choice` questions are 4-option). The original issue names that surface
   explicitly; do not substitute a "20-option typed-decisions" run.

The default loss is still `rlcd`. Whether `soft-ce` should replace it is decided by the Gate D
matrix below, not by a single seed.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | `LocalLLaMA/typed-decisions` → `train.jsonl` / `test.jsonl` / `test_meta.json` |
| `run_benchmark.py` | Runs `laya-train` for a gate's loss × seed matrix; writes a per-run manifest |
| `evaluate.py` | Accuracy / ECE / Brier on the test split, under shipped and raw temperatures |
| `metrics.py` | The notebook's metric definitions (shared by `evaluate.py` and the smoke test) |
| `order_flip.py` | Inference-only 20-option flip rate on `massive_intent.en` |
| `analyze_gate_d.py` | Summarise `step4_gate_d_results.json` into the 3-seed panel + verdict |
| `smoke_test.py` | CPU, weight-free: metric plumbing + CLI wiring + tiny end-to-end |
| `kaggle/` | Generators for the self-contained single-T4 Kaggle kernels (Gate R, Gate D) |

## Exact benchmark contract

Pinned from the published notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`)
and #887's thread. Confirm against the base checkpoint config at run time (read once, record it).

| | value | source |
|---|---|---|
| dataset | `LocalLLaMA/typed-decisions`, config `all` | notebook |
| train split | `train` — 1,200 cases / 6,000 decisions | notebook |
| test split | `test` — 400 cases / 2,000 decisions | notebook |
| base checkpoint | `convaiinnovations/laya` | notebook |
| loss (reproduction) | `rlcd` (`laya-train` default) | #887 |
| epochs | 4 | notebook |
| micro-batch / grad-accum | 8 / 8 — effective batch 64 single-device (notebook: 8 × 2 GPUs × 4) | notebook, `TrainConfig` |
| learning rates | encoder 2.5e-5, head 1e-4, AdamW cosine | notebook |
| `max_len` / `head_max_len` | 1024 / 256 | notebook, `docs/finetune.md` |
| training seed | 0 | #887 |
| calibration slice | held out inside `laya-train`: `calib_frac` 0.1 capped at 400, seed 20260922 | `TrainConfig`, notebook |
| mixed precision | fp16 autocast on CUDA; gradient checkpointing follows it | notebook, `TrainConfig` |

### Metric definitions (notebook cell 14)

- **accuracy** — top option vs gold label, over every decision (choice + noul + score).
- **ECE** — `laya.common.ece_score` of the top probability against correctness, 15 equal-width
  bins, over every decision.
- **Brier** — mean over **choice and noul** decisions of `sum((p - gold)**2)`; the notebook does
  not append score decisions to this mean, so a score change never moves Brier. Kept as-is so the
  number compares with the reference.
- **soft accuracy** — mean over **choice and noul** of `sum(p * gold)`.

### Reference numbers to compare against

From #887 (same split and seed, notebooks linked in-thread):

| run | accuracy | ECE | Brier |
|---|---|---|---|
| base `laya` | 0.362 | 0.174 | 0.329 |
| `loss="rlcd"` (notebook objective) | 0.7715 | 0.150 | 0.050 |
| `loss="soft-ce"` | 0.790 | 0.157 | 0.047 |
| published `laya-typed-decisions` | 0.766 | 0.213 | 0.061 |

Accuracy is the cleanest training-reproduction signal; ECE and Brier must name the calibration
path that produced them (next section).

### Calibration is not byte-comparable across contracts

`laya.train` fits with `laya.calibrate.fit_temperature_map` — per type, plus per option-count
bucket above a floor — under the runtime clamp `[TEMP_MIN, TEMP_MAX] = [0.5, 5.0]`. The old
notebook fitted one LBFGS temperature per type under `[0.1, 10]`. So `evaluate.py` supports either
path, one per invocation: `--temperatures shipped` uses the checkpoint's own fit and
`--temperatures raw` uses 1.0 everywhere. **The Gate D table in `GATE_D_RESULTS.md` is the shipped
path only**; no raw-temperature cells were measured. An ECE difference explained by the
calibration-contract change is not a training non-reproduction; say which path produced each number.

## Gates

### Gate R — reproduction

One run: `rlcd`, seed 0. Passes when accuracy lands on the published checkpoint / the maintainer's
target within single-seed noise, and ECE/Brier are reported with their calibration path. Nothing
about the default loss is concluded from Gate R.

```bash
python research/scripts/step4_benchmark/prepare_data.py --out-dir data/typed_decisions
python research/scripts/step4_benchmark/run_benchmark.py --gate R \
    --data data/typed_decisions/train.jsonl --out-dir runs/step4 --device cuda
python research/scripts/step4_benchmark/evaluate.py \
    --checkpoint runs/step4/rlcd_seed0 \
    --test-meta data/typed_decisions/test_meta.json --device cuda
```

### Gate D — default-loss decision

`rlcd` vs `soft-ce` over more than one seed. The existing soft-ce win is single-seed, so it is not
enough to recommend changing a shipped default. Minimal matrix (seeds 0/1/2, 6 runs):

```bash
python research/scripts/step4_benchmark/run_benchmark.py --gate D \
    --data data/typed_decisions/train.jsonl --out-dir runs/step4 --device cuda
```

Summarise paired per-decision outcomes across seeds, not just the two means. The measured
three-seed panel and its verdict are in [`GATE_D_RESULTS.md`](GATE_D_RESULTS.md), with the raw cells
under [`results/`](results/).

### 20-option order stability

`massive_intent.en`, 20 options (gold + 19 distractors, seed 13, matching
`research/scripts/build_benchmark_nb.py`). GuilhermeFusari's seed-0/1 numbers from `laya.train`
already cover this surface; reference them unless a reviewer wants the number reproduced on a
specific checkpoint, in which case run inference-only:

```bash
python research/scripts/step4_benchmark/order_flip.py \
    --checkpoint <checkpoint> --per-lang 300 --out flip.json
```

This does not train. Do not start a second MASSIVE training campaign by inertia.

## Worked example / smoke

`smoke_test.py` runs on CPU with a tiny local checkpoint and a tiny typed-decisions-shaped JSONL.
It proves the metric definitions against hand-computed values, the CLI command, and the loop from
JSONL through `laya-train` to `evaluate.py`. It downloads nothing and trains nothing large.

```bash
python research/scripts/step4_benchmark/smoke_test.py
```

## Runtime (keep the three apart)

- **Reference (notebook, 2×T4):** ~4–6 minutes for the demo's 6,000 decisions.
- **Guilherme measured (one T4):** ~48–54 minutes per run — but that is the **MASSIVE 20-option**
  experiment, not typed-decisions. Do not carry it over.
- **Our measured (one T4):** 2691.3–2708.5 s (≈44.9–45.1 min) per cell, over the six Gate D cells
  (`rlcd` seed 0 is the reused Gate R run).

Single-device `laya-train` uses micro-batch 8 × grad-accum 8 = effective 64; the notebook used
8 × 2 GPUs × 4 = effective 64. Same effective batch, run on one GPU instead of two; the different
epoch-order RNG formula and environment are noted under Limitations above.

## Constraints

- Does not modify `laya/train.py`, `laya/train_cli.py` or `tests/test_train.py` (open PRs
  #967/#968/#982).
- Uses `laya-train` as a subprocess, so it does not depend on training internals.
- Reuses `laya.common.ece_score` and the notebook's own metric definitions rather than adding a
  second evaluation framework.
