# laya-eval — a reproducible per-language evaluation harness

An independent harness for measuring a Laya checkpoint: per-language accuracy and
calibration, with machine-readable per-case output.

It exists because the repository's own benchmark scripts are research code. They
download every checkpoint, run every part, and print tables. There was no small,
reproducible harness a third party could point at a checkpoint to answer "how does
this model do on my language, and can I trust its confidence?" — and no per-case
record behind the published numbers, so they could not be re-derived without a GPU
and the original environment.

This addresses the ask in
[#35](https://github.com/NandhaKishorM/laya/issues/35):

> A fixed prompt format plus a per-language ECE report is exactly what the repo
> lacks ... Per-case JSON would be very welcome too.

## Install

Nothing beyond a normal Laya install, plus `datasets`:

```bash
pip install laya datasets
```

The harness is deliberately not part of the `laya` package: it is evaluation code,
it pulls a dataset, and `import laya` should stay dependency-light.

## Use

```bash
# one language
python research/eval/laya_eval.py --model convaiinnovations/laya --langs en

# several, with a JSON report
python research/eval/laya_eval.py --model convaiinnovations/laya \
    --langs en,de,ro --out report.json

# every MASSIVE language
python research/eval/laya_eval.py --model convaiinnovations/laya --langs all --out all.json

# the multilingual checkpoint
python research/eval/laya_eval.py --model convaiinnovations/laya \
    --subfolder multilingual --langs all --out multilingual.json

# a local checkpoint
python research/eval/laya_eval.py --model ./my-finetune --langs en
```

Output, per language:

```
  en       n=100  acc=0.8200 macro_f1=0.7876 ece=0.1789 conf=0.9989  (36.7s)

  macro over 51 languages: acc=...  ece=...  f1=...
```

and a JSON document with four parts:

| key | contents |
|---|---|
| `config` | checkpoint, device, `max_len`, `head_max_len`, dataset, `per_lang`, `n_opts`, seed, the fixed instructions, the temperatures in force, laya version |
| `report` | per language: `n`, `accuracy`, `macro_f1`, `ece`, `mean_confidence`, `acc_at_50_coverage`, `temperature` |
| `summary` | macro accuracy / ECE / macro-F1 over the languages that ran |
| `cases` | every individual decision |

Each case carries `state`, `instructions`, `options`, `gold_index`, `gold_label`,
`pred_index`, `pred_label`, `probability`, `p_gold`, `confidence`, `correct` and the
`temperature` used. That is enough to re-derive every number in `report` from the
file alone, with no model and no network:

```python
import json
d = json.load(open("report.json"))
n = len(d["cases"])
acc = sum(c["correct"] for c in d["cases"]) / n
assert abs(acc - d["report"]["en"]["accuracy"]) < 5e-5
```

## Method

Chosen so results are comparable with the published tables, which is the point of a
second implementation:

| | |
|---|---|
| dataset | `mteb/amazon_massive_intent`, split `test` |
| sampling | first `--per-lang` rows (default 100); `random.Random(13)` created **fresh per language** |
| options | `--n-opts` (default 20): the gold label plus `rng.sample` of the others, then shuffled |
| prompt | `What is the user asking for in \`utterance\`?` |
| option text | label with `_` → space and `.` → `: ` |
| metrics | accuracy, macro-F1, ECE over 15 equal-width confidence bins, mean confidence, accuracy at 50% coverage |
| temperature | the bucket `Agent` would apply, selected by `(question type, option count)` |

`--unclamped` scores with the checkpoint's **raw** bucket temperatures instead of the
clamped ones `Agent` applies. That is what reproduces the committed sweep, and it is
also how the two can be compared.

## Verification

Checked against the committed sweep, not only against itself. Both checkpoints over
**all 51 languages** (`--langs all --per-lang 100 --n-opts 20`), per-language accuracy
compared against `research/results/cpu_51_language_sweep.json`:

| checkpoint | per-language accuracy identical | `macro_accuracy` committed → mine | `macro_ece` committed → mine |
|---|---|---|---|
| **english** | **51 / 51** | 0.2269 → **0.2269** | 0.7331 → 0.5709 |
| multilingual | 6 / 51 | 0.3661 → 0.4008 | 0.3869 → 0.3911 |

The english checkpoint reproduces every per-language accuracy, not just the macro.
Those are deterministic outputs on a fixed sample, so they can only agree if the
sampling, prompt text, option construction and inference path are all identical to
the committed run.

The `macro_ece` gap on english is the temperature clamp — `choice:11+` is `0.1006`
raw and `0.5` as served ([#208](https://github.com/NandhaKishorM/laya/issues/208)).
`--unclamped` exists so both regimes can be produced from one tool. The single-language
view is the same result in miniature (`--langs en --unclamped`):

| metric | committed | `--unclamped` | default |
|---|---|---|---|
| `accuracy` | 0.82 | 0.82 | 0.82 |
| `macro_f1` | 0.7876 | 0.7876 | 0.7876 |
| `ece` | 0.1789 | **0.1789** | 0.1382 |
| `mean_confidence` | 0.9989 | **0.9989** | 0.9582 |
| `acc_at_50_coverage` | 0.94 | **0.94** | 0.98 |

### The multilingual checkpoint no longer matches its committed row

45 of 51 multilingual accuracies differ, so this is not a plumbing accident here —
the same code reproduces english 51/51. Most of the movement is upward
(`bn` 0.29→0.45, `kn` 0.15→0.30, `fa` 0.39→0.51), a few downward (`sv` 0.57→0.49).
`macro_ece` barely moves (0.3869→0.3911), consistent with the multilingual checkpoint
having an empty `temperature_by_options`, so the clamp cannot explain it.

Ruled out: the option sets (identical digest to the english run), the weights
(bundled and standalone multilingual are byte-identical, all 170 tensors
`torch.equal`), the dataset (revision `940fd47a`, last modified 2026-02-24), and
`build_sequence` (unchanged since `v0.2.0`). Also ruled out, on re-measurement:

* **the shipped `head_max_len`**, which matters here because this checkpoint ships
  `256` and english ships `192`. The harness reads it from the checkpoint's own
  config and the run's `config` block records `head_max_len: 256, max_len: 1024`, so
  the multilingual numbers above were not taken at english's budget. Re-running with
  the value read from config gives the same `0.4008`, and `6/51` again.
* **which of the two multilingual copies was measured.** The bundled `multilingual/`
  subfolder and the standalone `convaiinnovations/laya-multilingual` repo were each
  run end to end over all 51 languages and both give `macro_accuracy 0.4008`,
  `macro_ece 0.3911`, `6/51`.
* **a checkpoint change since the committed sweep.** `multilingual/model.safetensors`
  is `643835514` bytes at `sha256 b99c8bea…` and `multilingual/rl_agent_config.json`
  is `472` bytes at `sha256 00e35f88…` at every revision from the sweep's timestamp to
  today; the Hub commits in that window are model-card `docs:`/`assets:` only.

It is in the multilingual inference path between `laya 0.2.0` and `0.3.6` and is
**not** reconciled. Flagged rather than hidden.

Related: **`head_max_len` is load-bearing for accuracy**, not just for option
truncation. The english checkpoint at its shipped `head_max_len=192` scores 0.82;
forcing 256 or 512 drops it to 0.79.

## Tests

`research/eval/test_laya_eval.py` covers the pure functions and runs offline — no
checkpoint, no network:

```bash
python research/eval/test_laya_eval.py     # 64 passed, 0 failed
```

It pins the upstream constants (seed 13, 20 options, the exact instruction string),
the determinism of the sampler, that a fresh RNG per language is used, and the
metric arithmetic, including the `confidence == 0.0` bin boundary that this harness
shares with `laya.common.ece_score`, `research/scripts/bench_local.py` and
`research/scripts/build_benchmark_nb.py`. That boundary is asserted against all four,
not just against this harness's own arithmetic.

## Limits

* MASSIVE intent only. The same shape applies to `scenario` and to XNLI, but neither
  is wired up here.
* `per_lang=100` is the published setting, not a statistical one. Per-language ECE on
  100 cases is noisy; raise `--per-lang` and say so when quoting a number.
* The English checkpoint collapses on non-Latin scripts (see `BENCHMARKS.md`), so a
  low score in one language is not by itself evidence of a misroute — check
  `laya.lang.analyse` for the script before concluding which checkpoint was used.
* The `confidence == 0.0` bin boundary is the one
  [#39](https://github.com/NandhaKishorM/laya/pull/39) settled: the first bin is closed
  at the bottom, so `0.0` is counted. This harness used `conf > lo` for every bin until
  the divergence was found, which made it the only one of the four implementations that
  binned differently. It now matches `laya.common.ece_score`,
  `research/scripts/bench_local.py` and `research/scripts/build_benchmark_nb.py`, and
  `test_laya_eval.py` asserts that agreement.

### The temperature clamp, measured both ways

`research/results/cpu_51_language_sweep_clamped.json` carries the same re-run twice, once per
regime, against the committed columns. Macro accuracy reproduces the committed file exactly
and macro ECE is the only macro figure that moves:

| | committed | `--unclamped` | default |
|---|---|---|---|
| `macro_accuracy` | 0.2269 | **0.2269** | 0.2269 |
| `macro_ece` | 0.7331 | **0.7331** | 0.5709 |
| `macro_f1` | 0.2053 | **0.2053** | 0.2053 |

Per language, the unclamped run agrees with the committed file on `accuracy` and `macro_f1`
in **51/51**, on `ece` in **48/51** and on `mean_confidence` in **49/51**. The handful that
differ do so by `0.0001`, the last stored digit: the committed run used torch 2.8.0 and this
one 2.14.0. The clamped run differs from the committed file on `ece` and `mean_confidence` in
**51/51**, every one of them lower, because it is the only column the clamp can move.

`accuracy`, `macro_f1` and `n` are identical in all three columns by construction: scaling
logits by any positive temperature does not change the argmax. That is why a re-run can settle
the calibration question without reopening the accuracy numbers.

---

## Presentation checks (`presentation_checks.py`)

A label-free regression check for the `score` position prior in #131.
`laya-multilingual` rarely picks the first-listed `score` level, and the fix is a
position-balanced retrain. This script says whether a retrained checkpoint removed
the prior. Every input is fixed in the file (10 short English states written for it),
so it needs no dataset and no labels.

```bash
python research/eval/presentation_checks.py --model convaiinnovations/laya --subfolder multilingual
python research/eval/presentation_checks.py --model ./retrained-checkpoint --out report.json
```

Exit status: `0` every check passed, `1` a check failed, `2` the harness disagrees with
`Agent.system_one` by more than `1e-3` (nothing else is trusted then). CPU is the
default device: fp32 and deterministic, which is what the thresholds were set on.

### The two checks

| check | input | metric | gate |
|---|---|---|---|
| `score_slot0_identical` | one `score` question whose K levels all carry the same text; texts `moderate` and `a request`, K = 3, 4, 5 | raw slot-0 marker logit minus the mean over the K slots, averaged over 10 states × 6 configurations | `>= -0.20` |
| `score_first_slot_permuted` | `Not urgent` / `Soon` / `Work is blocked` in all 6 orders, per state | share of the 60 decisions whose argmax is the first slot | `>= 0.15` |

`score_slot0_identical` is the identical-option control from @AlKor13 in #131. With
identical texts the rendered options differ only by position and by the `level N:`
prefix that `render_options` always emits, so a checkpoint without a slot prior has
no reason to prefer or avoid any slot.

`score_first_slot_permuted` presents every order of the three levels, so each level
sits in each slot exactly twice per state. A checkpoint whose answer does not depend
on the order picks the first slot in exactly 1/3 of the decisions, whatever the states
say; the rate moves only through order dependence.

Both read raw marker logits (before temperature) through `laya_eval.score_cases`,
and the script first compares that path with `Agent.system_one` on every state.

### Measured on the shipped checkpoints

CPU, fp32, `convaiinnovations/laya@1c5edc1`, laya 0.3.7. Full output, per state and
per configuration: `research/results/presentation_checks_shipped.json`.

| checkpoint | `score_slot0_identical` (leave-one-out) | `score_first_slot_permuted` (leave-one-out) | verdict |
|---|---|---|---|
| `laya` (english) | **+0.664** (+0.520 .. +0.741) | **0.217** (0.204 .. 0.241) | PASS |
| `laya-multilingual` | **−0.492** (−0.563 .. −0.425) | **0.017** (0.000 .. 0.019) | FAIL |

Parity with `Agent.system_one`: max |Δp| 4.98e-5 (multilingual) and 4.92e-5 (english),
which is the 4-decimal rounding of `system_one`'s probabilities.

### Thresholds

The gates were set from the leave-one-out ranges above, not tuned to them. Two
conditions were fixed before the 10-state run:

1. the current multilingual checkpoint fails and the english checkpoint passes in
   **every** leave-one-out subset, and
2. the worst leave-one-out value of each checkpoint clears the threshold by at least
   0.10 logit (slot 0) and 0.05 (first-slot rate, 3 of 60 decisions).

| check | threshold | multilingual worst → margin | english worst → margin |
|---|---|---|---|
| `score_slot0_identical` | −0.20 | −0.425 → 0.225 | +0.520 → 0.720 |
| `score_first_slot_permuted` | 0.15 | 0.019 → 0.131 | 0.204 → 0.054 |

The tightest margin is the english first-slot rate, at 0.054 against the 0.05 rule.
An order-invariant checkpoint sits at exactly 0.333 on that check.

### Tests

`research/eval/test_presentation_checks.py` runs offline, with scripted logits in
place of a checkpoint:

```bash
python research/eval/test_presentation_checks.py     # 69 passed, 0 failed
```

It pins the fixed inputs and both gates. It checks that the identical-option
questions render as `level i: <same text>`, and that every level sits in every slot
exactly twice. It also checks the metric arithmetic by hand, the leave-one-out
bounds, the one-sided gates, and the exit codes. A scripted slot-0 hole fails both
checks, and an order-invariant model scores exactly 1/3.

### Limits

* **The gate is one-sided because the english checkpoint is not flat either.** With
  identical options it prefers the early slots, more strongly as K grows: slot 0 sits
  +0.10 / +0.41 / +0.85 above the mean at K = 3 / 4 / 5 with `moderate`, and
  +0.21 / +0.74 / +1.68 with `a request`. At K = 3 with `moderate` it is close to flat,
  which matches the #131 control. A two-sided "no position effect" gate would fail
  the english checkpoint, so the check asks the narrower question #131 is about:
  whether slot 0 is suppressed.
  (multilingual: −0.75 / −0.52 / −0.25 and −0.58 / −0.47 / −0.37.)
* Passing is not accuracy. A checkpoint can clear both gates and still rank urgency
  badly; this checks one known failure, not `score` quality.
* English only, `score` only, 10 states. The states are short support messages, so a
  checkpoint's behaviour on long inputs or other languages is not covered here.
* Thresholds were set on CPU fp32. On CUDA, `Agent` runs the forward pass under
  reduced-precision autocast and `score_cases` does not. The parity check reports that
  difference instead of hiding it.
* New checks are one function each, registered in `CHECKS`.


## Metamorphic option-order robustness (experimental)

`metamorphic.py` adds the initial scope of
[#244](https://github.com/NandhaKishorM/laya/issues/244): **choice option order robustness**, without changing model/runtime behavior. Label renaming, neutral labels, paraphrases, structured-state permutations, `score` and `noul` perturbations are intentionally deferred. Run from the repository root after installing Laya and `datasets`:

```bash
python -m research.eval.metamorphic --model convaiinnovations/laya \
    --langs en --per-lang 100 --n-opts 20 --batch-size 16 --out robustness.json
python -m research.eval.metamorphic --model convaiinnovations/laya \
    --subfolder multilingual --langs en --out multilingual-robustness.json
python -m unittest research.eval.test_metamorphic -v
```

Each MASSIVE case uses the existing harness's sampler and produces two inputs:

1. The unchanged baseline.
2. One seeded shuffle of option order; if the shuffle is the identity, a one-slot
   rotation is used. This is a bounded diagnostic, not exhaustive permutation testing
   or a uniform draw over all nonidentity permutations.

Instructions and state are otherwise unchanged. Option key/value pairs are moved together during permutation. Every result is mapped back to the original semantic option order **before** predictions and metrics are computed. Exact ties choose the first canonical option. The RNG starts fresh per language; `--seed` controls both sampling and transformations. `--batch-size` bounds the number of forward-pass inputs and does not alter the generated variants. Model inference may still have small floating-point differences across devices and batch sizes.

The JSON contains `config`, per-language `report`, and full `cases`. Each case saves its original input, canonical keys and optional gold index; each variant saves its presented keys, explicit `canonical_to_transformed` and `transformed_to_canonical` label mappings, slot-to-canonical indices, complete **canonical-order** probability vector, prediction, confidence, correctness (or `null`), and comparison to baseline. Probabilities are not rounded. The config records model/subfolder, temperature mode and values, truncation settings, dataset, seed and batch size. For reproducible checkpoint comparisons, use a pinned local snapshot and retain the environment versions alongside the report. `--unclamped` has the same meaning as in `laya_eval`. If any language fails, its error is saved and the command exits nonzero while retaining successful languages.

Metrics are grouped under `option_order` and `overall`:

| Metric | Definition |
|---|---|
| `semantic_agreement_rate` | Fraction of baseline/variant pairs with the same canonical argmax |
| `mean_probability_drift` | Mean absolute probability change across options, then pairs |
| `max_probability_drift` | Largest absolute change of any option across all pairs |
| `mean_js_divergence` | Mean Jensen-Shannon divergence using natural logs, in `[0, ln(2)]` |
| `mean_confidence_drift` | Mean signed change of maximum probability, variant minus baseline |
| `mean_absolute_confidence_drift` | Mean magnitude of that confidence change |
| `worst_confidence_increase_on_disagreement` | Largest positive confidence change among changed decisions, or zero if none |

`overall` is pair-weighted, not a fraction of cases where *all* variants agree. `quality` separately reports accuracy and the existing harness's 15-bin ECE for baseline and each transformation on labelled cases only. Empty groups contain `n: 0`; unlabelled quality groups contain `n_labelled: 0` without inventing an accuracy or ECE. Robustness agreement is not a correctness measure: consistently wrong predictions can be perfectly invariant.

For another corpus, the Python API accepts `(state, questions)` cases in the same shape as the harness, and a callback returning probability vectors in presented option order:

```python
from research.eval.metamorphic import evaluate, model_scorer
agent.model.eval()
result = evaluate(cases, model_scorer(agent), gold_indices=None, seed=13)
```

The first version intentionally accepts only **one choice question per case**, with at least two options. Label renaming and other metamorphic transforms are intentionally deferred as proposed in the issue.

For an explicit single-case experiment, the same implementation exposes:

```python
from research.eval.metamorphic import (
    MetamorphicCase, permute_options,
    evaluate_variants, compare_predictions,
)

case = MetamorphicCase(state, questions, gold_index=None)
variants = [permute_options(case, seed=42)]
agent.model.eval()
results = evaluate_variants(agent, case, variants)
report = compare_predictions(baseline=results.baseline, variants=results.variants)
```

For offline tests, pass `agent=None, score=fake_scorer` to `evaluate_variants`. The scorer takes a batch of harness `(state, questions)` inputs and returns one probability vector per input. The public transformations return independent copies and explicit mappings in both directions, including identity label mappings for order-only transformations.

**Semantic agreement and distribution stability are different properties.**
A shift from `[0.91, 0.06, 0.03]` to `[0.88, 0.08, 0.04]` preserves the decision while showing nonzero drift. Switching the winner is reported as disagreement, regardless of whether confidence rises or falls. No metric here automatically classifies either observation as a bug; acceptable variation depends on the use case, and the report deliberately defines no universal pass/fail threshold.
