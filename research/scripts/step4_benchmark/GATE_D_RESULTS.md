# Gate D — RLCD vs soft-CE default-loss panel (three seeds)

Evidence for [#887](https://github.com/NandhaKishorM/laya/issues/887) Step 4. The matrix exists to
answer one question: does the single-seed soft-CE win hold up well enough to change the shipped
default loss? It does not.

- Upstream pin: `a4a8921afebfd852bba0000475cfb6ab737a124c`
- Compute: one Tesla T4 per run (`CUDA_VISIBLE_DEVICES=0`, exactly one visible GPU asserted),
  `torch 2.11.0+cu128`, Kaggle free tier.
- Contract: the Gate R contract — `LocalLLaMA/typed-decisions` config `all`, 1,200 train cases /
  6,000 decisions, calibration 400 items (`calib_frac` 0.1, `calib_seed` 20260922, held out inside
  `laya-train`), 400 test cases / 2,000 decisions, 4 epochs, micro-batch 8 × grad-accum 8,
  encoder LR 2.5e-5 / head LR 1e-4, `max_len` 1024 / `head_max_len` 256, no shuffling, no early
  stopping, `loss` the only varying field.
- `rlcd` seed 0 is the Gate R run, reused here; the other five cells were trained for this panel.
- Raw per-cell artifacts: [`results/step4_gate_d_results.json`](results/step4_gate_d_results.json),
  [`results/step4_gate_r_results.json`](results/step4_gate_r_results.json).

## Result

All cells use the shipped-temperature path (`--temperatures shipped`); no raw-temperature cells
were measured.

| loss | seed | accuracy | ECE | Brier | wall (s) |
|---|---|---|---|---|---|
| `rlcd` | 0 | 0.7750 | 0.1504 | 0.0519 | 2691.3 |
| `rlcd` | 1 | 0.7780 | 0.1477 | 0.0519 | 2697.6 |
| `rlcd` | 2 | 0.7755 | 0.1530 | 0.0490 | 2708.5 |
| `soft-ce` | 0 | 0.7605 | 0.1353 | 0.0532 | 2691.4 |
| `soft-ce` | 1 | 0.7915 | 0.1574 | 0.0481 | 2705.1 |
| `soft-ce` | 2 | 0.7770 | 0.1471 | 0.0513 | 2701.8 |

Same split and seed, from the #887 thread, for context:

| run | accuracy | ECE | Brier |
|---|---|---|---|
| base `convaiinnovations/laya` | 0.3620 | 0.1741 | 0.3287 |
| published `laya-typed-decisions` | 0.766 | 0.213 | 0.061 |
| `rlcd` single seed (#887) | 0.7715 | 0.150 | 0.050 |
| `soft-ce` single seed (#887) | 0.790 | 0.157 | 0.047 |

## Paired `soft-ce − rlcd`

| seed | accuracy Δ | ECE Δ | Brier Δ |
|---|---|---|---|
| 0 | −0.0145 | −0.0151 | +0.0013 |
| 1 | +0.0135 | +0.0097 | −0.0038 |
| 2 | +0.0015 | −0.0059 | +0.0023 |
| mean | **+0.0002** | −0.0038 | −0.0001 |
| min / max | −0.0145 / +0.0135 | −0.0151 / +0.0097 | −0.0038 / +0.0023 |

Across the three seeds `rlcd` accuracy is 0.7762 (std 0.0016) and `soft-ce` is 0.7763
(std 0.0155). The paired delta changes sign between seeds, so there is no consistent direction.

ECE and Brier are reported under the shipped temperature fit (see limitations) and only over the
notebook's Brier scope (choice + noul), so small movements there are not portable.

## Per-type accuracy (mean over seeds)

| type | `rlcd` | `soft-ce` |
|---|---|---|
| choice | 0.7422 | 0.7539 |
| noul | 0.8478 | 0.8533 |
| score | 0.7479 | 0.7354 |

The small choice/noul gain and the score loss swap rank between seeds; neither is stable.

## Verdict

`INSUFFICIENT`. The paired accuracy delta changes sign across seeds (mean +0.0002, range −0.0145 to
+0.0135), so the seeds disagree about direction; the soft-CE seed spread (std 0.0155) is also an
order of magnitude larger than RLCD's (0.0016). The analyser applies no materiality threshold — what
advantage would justify changing a shipped default is a maintainer judgement, not something this
panel can encode — but a sign that flips between seeds is not a basis for moving one. `rlcd` remains
the default; this PR does not change it.

## Reproduction

The harness runs `laya-train` as a subprocess and scores the test split with the notebook's metric
definitions. Locally (no GPU, weight-free):

```bash
python research/scripts/step4_benchmark/smoke_test.py
python research/scripts/step4_benchmark/analyze_gate_d.py \
    --results research/scripts/step4_benchmark/results/step4_gate_d_results.json
```

The six cells were trained on Kaggle from the committed generators, one process per GPU:

```bash
python research/scripts/step4_benchmark/kaggle/build_kernel.py     # Gate R (reused rlcd seed 0)
python research/scripts/step4_benchmark/kaggle/build_gate_d.py     # Gate D (remaining 5 cells)
kaggle kernels push  -p research/scripts/step4_benchmark/kaggle
kaggle kernels output yuyi722333/laya-step-4-gate-d -p out --file-pattern 'step4_gate_d_results\.json'
```

Each cell asserts the fail-closed readback (device count, `max_len`/`head_max_len` 1024/256,
dataset counts, finite metrics) before its numbers are accepted.

## Limitations

- **Notebook RNG mismatch.** `laya.train` derives the epoch seed as `seed + epoch`; the published
  notebook used `42 + epoch + rank`. The gap to the notebook's own numbers is therefore not a pure
  environment delta.
- **Environment.** `torch 2.11.0+cu128` here; `docs/finetune.md` describes 2.14.
- **Calibration scope.** ECE uses the checkpoint's shipped per-type temperature fit under the
  runtime clamp `[0.5, 5.0]`; the notebook fitted per-type LBFGS temperatures under `[0.1, 10]`.
  The two are not byte-comparable, so ECE/Brier name their path. Brier follows the notebook and
  averages choice + noul only, so a score change never moves it.
- **Base config metadata.** The base checkpoint's `training` block records `max_len` 512 /
  `head_max_len` 192; each run passed the contract's explicit 1024 / 256 and asserted it back.
- **Inherited metadata.** `training.*` counters (`updates`, `epochs_completed`, `hours`) are copied
  from the base checkpoint into the fine-tuned checkpoint.
- **Peak VRAM** was not captured.
- **n = 3 training seeds.** No significance claim is made from three seeds.

## Reused evidence, not rerun

20-option order stability is a different surface (typed-decisions `choice` questions have four
options). This PR does not train MASSIVE. It references
[GuilhermeFusari's public notebook *Laya: option shuffling at 20 options (MASSIVE intent,
English)*](https://www.kaggle.com/code/guilhermediasfusari/laya-option-shuffling-at-20-options-massive-inte)
as the order-stability evidence: `massive_intent.en`, gold intent plus 19 distractors drawn from
the 60 intents in random order, seed 13, `max_len` 512, one process per T4, on Laya
`8a6e132`. Our numbers here are on current main (`a4a8921`), so the two are reported separately
rather than merged.

## Follow-up candidates (not in this PR)

- The `training.*` inheritance above is a checkpoint-metadata defect, not a training defect; it
  would need its own issue.
- The notebook-vs-`laya.train` epoch-seed formula is a reproducibility mismatch, also separate.
