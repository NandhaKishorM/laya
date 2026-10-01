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

### Other languages

`--lang` runs both checks on fixed states in Japanese, Korean, Hindi or Turkish (#602).
Each set translates the ten English states one for one, with the level texts in the
same language. Every state routes to `multilingual` under `Router`: Japanese, Korean
and Hindi by script, Turkish by its non-English letters. The run prints the checkpoint
`Router` picks for each language. Korean, Hindi and Turkish are among the languages
whose per-language MASSIVE gains the multilingual model card lists, and they cover
three routing paths (Hangul, Devanagari, Latin with diacritics). Japanese is where
#131 was found. The gates are the English ones. Without `--lang` the run and its report are
unchanged.

```bash
python research/eval/presentation_checks.py --model convaiinnovations/laya \
    --subfolder multilingual --lang ja,ko,hi,tr --out langs.json
```

CPU, fp32, `convaiinnovations/laya@55cf4c4` (subfolder `multilingual`), laya 0.3.21.
Full output: `research/results/presentation_checks_langs.json`. The same run without
`--lang` reproduces the English row above exactly.

| language | `score_slot0_identical` (leave-one-out) | `score_first_slot_permuted` (leave-one-out) | parity max \|Δp\| |
|---|---|---|---|
| `ja` | −0.018 (−0.120 .. +0.031) PASS | **0.000** (0.000 .. 0.000) FAIL | 4.68e-5 |
| `ko` | **−0.570** (−0.622 .. −0.506) FAIL | **0.050** (0.019 .. 0.056) FAIL | 4.97e-5 |
| `hi` | **−0.225** (−0.300 .. −0.182) FAIL | **0.067** (0.037 .. 0.074) FAIL | 4.91e-5 |
| `tr` | +0.139 (+0.068 .. +0.174) PASS | **0.033** (0.019 .. 0.037) FAIL | 4.95e-5 |

The first-slot check fails in all four languages, as it does in English. The
identical-option control passes in Japanese and Turkish, so on those states it would
not catch the prior on its own. That is the case for running both checks in every
language.

### The choice control (`choice_slot0_identical`)

The identical-option control, applied to `choice` (#602, part a). Choice keys must be
unique, so the options are numbered keys with one shared description (`1: a request`,
`2: a request`, ...), the counterpart of score's `level N:` prefix. The metric is the
score control's metric on the choice marker logits: slot 0 minus the mean over the
options, averaged over the states and both texts. It is gated at 4 options (the gate
is the same, −0.20), and 3 options are reported beside it. The control runs in every
`--lang` run, and `--checks choice_slot0_identical` names it anywhere. Without either,
the default English report is unchanged. With the control, the parity check also
compares the choice probabilities with `Agent.system_one`.

CPU, fp32, `convaiinnovations/laya@55cf4c4`, laya 0.3.21. Full output:
`research/results/presentation_checks_choice.json`. `Router` picks `english` for the
English states and `multilingual` for the rest; every row runs on the checkpoint named.

| checkpoint | language | `choice_slot0_identical`, K = 4 (leave-one-out) | K = 3 | choice parity |
|---|---|---|---|---|
| `laya` (english) | `en` | +0.902 (+0.833 .. +0.971) PASS | +0.322 | 4.84e-5 |
| `laya-multilingual` | `en` | +0.399 (+0.334 .. +0.455) PASS | +0.249 | 4.93e-5 |
| `laya-multilingual` | `ja` | +0.394 (+0.283 .. +0.493) PASS | +0.196 | 4.89e-5 |
| `laya-multilingual` | `ko` | +0.096 (−0.025 .. +0.204) PASS | −0.147 | 4.87e-5 |
| `laya-multilingual` | `hi` | +0.346 (+0.285 .. +0.389) PASS | +0.267 | 4.97e-5 |
| `laya-multilingual` | `tr` | +0.538 (+0.431 .. +0.588) PASS | +0.360 | 4.95e-5 |

On the same states `laya-multilingual` fails the score first-slot check in every
language, so its slot-0 deficit is specific to `score`: with choice options it
favours slot 0 instead. Korean is closest to the gate, and below 0 at 3 options.

### Tests

`research/eval/test_presentation_checks.py` runs offline, with scripted logits in
place of a checkpoint:

```bash
python research/eval/test_presentation_checks.py     # 219 passed, 0 failed
```

It pins the fixed inputs and both gates. It checks that the identical-option
questions render as `level i: <same text>`, and that every level sits in every slot
exactly twice. It also checks the metric arithmetic by hand, the leave-one-out
bounds, the one-sided gates, and the exit codes. A scripted slot-0 hole fails both
checks, and an order-invariant model scores exactly 1/3. For each language it
pins ten distinct states, three levels in every slot twice, and routing to
`multilingual`. It also checks the `--lang` parsing. The default run gives the same
report as `lang="en"`. For the choice control it checks the `N: <same text>` rendering,
the centring and the gate at 4 options by hand, the 3-option value beside it, and
that the control stays out of the default run.

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
* Score checks and the choice control only, 10 states per language. The states are short support messages, so a
  checkpoint's behaviour on long inputs is not covered here. The Korean, Hindi and
  Turkish states and levels were written by a non-native speaker; corrections from
  native speakers are welcome.
* Thresholds were set on CPU fp32. On CUDA, `Agent` runs the forward pass under
  reduced-precision autocast and `score_cases` does not. The parity check reports that
  difference instead of hiding it.
* New checks are one function each, registered in `CHECKS` (score) or `CHOICE_CHECKS`.


## Metamorphic option-order robustness (experimental)

`metamorphic.py` adds the initial scope of
[#244](https://github.com/NandhaKishorM/laya/issues/244): **choice option order robustness**, and the label-renaming transformation from [#512](https://github.com/NandhaKishorM/laya/issues/512), without changing model/runtime behavior. Paraphrases, structured-state permutations, `score` and `noul` perturbations are intentionally deferred. Run from the repository root after installing Laya and `datasets`:

```bash
python -m research.eval.metamorphic --model convaiinnovations/laya \
    --langs en --per-lang 100 --n-opts 20 --batch-size 16 --out robustness.json
python -m research.eval.metamorphic --model convaiinnovations/laya \
    --subfolder multilingual --langs en --out multilingual-robustness.json
python -m unittest research.eval.test_metamorphic -v
```

Each MASSIVE case uses the existing harness's sampler and produces three inputs:

1. The unchanged baseline.
2. One seeded shuffle of option order; if the shuffle is the identity, a one-slot
   rotation is used. This is a bounded diagnostic, not exhaustive permutation testing
   or a uniform draw over all nonidentity permutations.
3. A deterministic label rename: the option at each position keeps its slot and
   description, and its model-facing key becomes an opaque label (`A`, `B`, `C`, ...
   `Z`, then `key_26`, `key_27`, ...). Order and semantics are unchanged, so any
   drift in the `label_rename` group isolates lexical-label sensitivity (the failure
   mode of #156) from the position sensitivity measured by `option_order`.

Instructions and state are otherwise unchanged. Option key/value pairs are moved together during permutation. Every result is mapped back to the original semantic option order **before** predictions and metrics are computed, and each variant records the explicit bidirectional mapping so the comparison is auditable. Exact ties choose the first canonical option. The RNG starts fresh per language; `--seed` controls both sampling and transformations. `--batch-size` bounds the number of forward-pass inputs and does not alter the generated variants. Model inference may still have small floating-point differences across devices and batch sizes.

The JSON contains `config`, per-language `report`, and full `cases`. Each case saves its original input, canonical keys and optional gold index; each variant saves its presented keys, explicit `canonical_to_transformed` and `transformed_to_canonical` label mappings, slot-to-canonical indices, complete **canonical-order** probability vector, prediction, confidence, correctness (or `null`), and comparison to baseline. Probabilities are not rounded. The config records model/subfolder, temperature mode and values, truncation settings, dataset, seed and batch size. For reproducible checkpoint comparisons, use a pinned local snapshot and retain the environment versions alongside the report. `--unclamped` has the same meaning as in `laya_eval`. If any language fails, its error is saved and the command exits nonzero while retaining successful languages.

Metrics are grouped under `option_order`, `label_rename` and `overall`:

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

### Option budget and rendering information

At high option counts Laya's head token budget can cut the options themselves, and two options that share a prefix can come out of the cut as the same token span, which removes the question's ability to tell them apart. This matters for `label_rename`: a shorter label leaves more room for its description, so the baseline and the renamed input can survive the budget with **different content**. For `iot_hue_lightoff: iot hue lightoff` a roomy budget keeps the semantic description on both sides, but a tight budget can keep `iot_` on the baseline and `A: t` on the renamed variant -- no longer information-equivalent.

This section is an evaluation-side follow-up motivated by [#543](https://github.com/NandhaKishorM/laya/issues/543), [#517](https://github.com/NandhaKishorM/laya/issues/517) and [#569](https://github.com/NandhaKishorM/laya/issues/569). #543 raised option collapse at high option counts and was closed as addressed at runtime in v0.3.21 via #569/#542: the model side now reports, for an individual inference, which options lost a token span of their own. That report does not say whether the baseline and transformed variants of a *metamorphic comparison* still carry equivalent option information, which is the question this diagnostic answers -- per comparison, so the resulting drift is not read as pure lexical-label sensitivity.

With a real model, the CLI and `evaluate_variants()` derive a budget probe from the agent (`agent.cfg`) and attach a budget diagnostic to every variant, measured by rendering the options through the same tokenizer/`build_sequence` budget path the model input uses. `evaluate()` has no agent, so it probes only when it is passed an explicit `budget`:

* `budget` per variant: `options`, `distinct_spans` (how many options still have a token span of their own; fewer than `options` means some options can no longer be told apart), `tokens_per_option` (the uniform re-cap applied when the head budget runs out, or `null` when none was), `instruction_tokens`, `span_classes` (which presented options share a span), `retained_text` (the characters the surviving span covers, `null` when unattributable), `retained_descriptions` (the description characters inside that prefix, ignoring the label), `description_present` (whether each option carried description text at all -- `None`, empty and whitespace-only descriptions are absent; numbers, booleans and structured values render as text and count as present), `truncated` and `tail_truncated`.
* `budget_comparison` per pair: both sides' `options`, `distinct_spans`, `tokens_per_option` and `instruction_tokens`, their `retained_descriptions` aligned back to canonical order, both sides' `description_present` on `label_rename` pairs, `budget_confounded` (`true`, `false`, or `null` when the rendering could not be measured or attributed), and `reasons` naming what decided it: `clean`, `option_count`, `distinct_spans`, `instruction_tokens`, `tail_truncation`, `collision_partition` (the two sides group the options into indistinguishable spans differently, even when the number of collapsed options matches), `retained_description` or `unverified_retained_text` (with the affected canonical `slots`), `missing_semantic_description` (a `label_rename` pair whose options never had description text, with the affected canonical `slots`), or `measurement_error`.
* `report["budget"]`: `clean`/`confounded`/`unknown` counts and `confounded_rate` for `option_order`, `label_rename` and `overall`. The rate is over verifiable pairs only and is `null` when there are none, so an all-`unknown` run cannot read as clean.

Two conditions make a comparison's validity unknown and are reported as such, never as clean:

* **Renaming options that have no descriptions.** Choice criteria may legitimately carry `None`, empty or whitespace-only descriptions. Then the label is the only semantic content the option had, and rewriting it removes exactly what the transform promises to preserve, so `budget_confounded` is `null` with `missing_semantic_description` even when both sides report empty `retained_descriptions` and the budget never truncates anything. `option_order` pairs are exempt: a permutation moves every key/value pair together, so the content is preserved whatever the descriptions are, and the diagnostic never invents an `unknown` for a permutation.
* **Unattributable spans.** `retained_text`/`retained_descriptions` are attributed from character prefixes, and the only sound attribution is an exact witness: a prefix whose encoding equals the surviving span token for token. A growing prefix's token count is not guaranteed to be non-decreasing -- a BPE merge can encode a longer prefix to fewer tokens, and lossy normalization can drop characters -- so witnesses need not be adjacent and the search is global, over every character prefix. Exactly one witness is required: none, several, or an option longer than the probe's documented length bound (`metamorphic._ATTRIBUTION_LIMIT` characters) leaves the slot `null` and the pair unknown -- a conservative result instead of a possibly false clean one.

The diagnostic is deterministic and defines no threshold. It never erases or overrides the observed model metrics: a confounded pair still reports its `semantic_agreement` and drift, and `budget_confounded: true` with `semantic_agreement: false` (or with perfect agreement) are both representable. It does **not** prove semantic equivalence -- it reports whether the rendered option content still matches well enough for the drift to be read as lexical-label sensitivity. Options that the budget collapsed on both sides for the same reason stay clean, because option order and label renaming move or rewrite labels by design; only *content* differences count. `tokens_per_option` is reported but never confounds by itself: when the re-cap does not bite, the model sees the same sequence either way. Comparisons the probe cannot measure are reported as unknown rather than assumed clean. Without a model there is no tokenizer to probe, so `evaluate(..., budget=None)` leaves the report byte-identical to earlier versions.

For another corpus, the Python API accepts `(state, questions)` cases in the same shape as the harness, and a callback returning probability vectors in presented option order:

```python
from research.eval.metamorphic import evaluate, model_scorer
agent.model.eval()
result = evaluate(cases, model_scorer(agent), gold_indices=None, seed=13)
```

The first version intentionally accepts only **one choice question per case**, with at least two options. Paraphrases and other metamorphic transforms are intentionally deferred as proposed in the issues.

For an explicit single-case experiment, the same implementation exposes:

```python
from research.eval.metamorphic import (
    MetamorphicCase, permute_options, rename_labels,
    evaluate_variants, compare_predictions,
)

case = MetamorphicCase(state, questions, gold_index=None)
variants = [permute_options(case, seed=42), rename_labels(case)]
agent.model.eval()
results = evaluate_variants(agent, case, variants)
report = compare_predictions(baseline=results.baseline, variants=results.variants)
```

For offline tests, pass `agent=None, score=fake_scorer` to `evaluate_variants`. The scorer takes a batch of harness `(state, questions)` inputs and returns one probability vector per input. The public transformations return independent copies and explicit mappings in both directions, including identity label mappings for order-only transformations. Both entry points accept `budget=BudgetProbe(tok, max_len=..., head_max_len=...)` to attach the budget diagnostic to a stub tokenizer. `evaluate_variants()` derives one from `agent.cfg` automatically when a real agent is used, and skips it when a `score` is injected; `evaluate()` has no agent and probes only when a `budget` is passed explicitly.

**Semantic agreement and distribution stability are different properties.**
A shift from `[0.91, 0.06, 0.03]` to `[0.88, 0.08, 0.04]` preserves the decision while showing nonzero drift. Switching the winner is reported as disagreement, regardless of whether confidence rises or falls. No metric here automatically classifies either observation as a bug; acceptable variation depends on the use case, and the report deliberately defines no universal pass/fail threshold. **Experimental validity is a third property.** A `budget_confounded` comparison is still recorded with its metrics, but its drift no longer isolates lexical-label sensitivity.

### Selective prediction: does disagreement predict errors?

Robustness agreement is not a correctness measure, but on labelled cases it can be tested as a *signal* of correctness: are the baseline answers that change under a transform more often wrong? When gold indices are given, the report adds a `selective_prediction` section computed per labelled case from the baseline winner `w`:

| Signal | Definition |
|---|---|
| `confidence` | Baseline probability of `w` (the harness's maximum probability) |
| `agreement_<kind>` | Share of the `<kind>` variants whose canonical argmax is also `w` |
| `support_<kind>` | Mean probability the `<kind>` variants give `w` |
| `support_all` | Mean probability of `w` over the baseline and every variant |

`<kind>` is each transform present (`option_order`, `label_rename`). For every signal, `auroc` is the probability that a random correct case scores above a random wrong one (ties count half; `null` unless both correct and wrong cases exist). For the continuous signals, `accuracy_at_coverage` is the accuracy of the highest-scoring fraction of cases at 50, 70, 80 and 90% coverage; `agreement_*` is left out there because with one variant per kind it is binary, and the cut would depend on tie order. `n_labelled` and `n_wrong` give the sample size; unlabelled runs report `n_labelled: 0` only.

This measures whether a signal ranks errors below correct answers; it does not set a threshold. Consistently wrong predictions stay invisible to every `agreement_*` and `support_*` signal. With one variant per kind, `agreement_*` is coarse; the `support_*` signals use the full probability vectors.

#### Measured

MASSIVE `en`, `--per-lang 300 --n-opts 20`, seed 13, laya 0.3.21, CPU (65 wrong answers). Differences vs `confidence` are bootstrap 95% intervals over cases (2,000 resamples):

| Signal | AUROC | vs `confidence` |
|---|---:|---:|
| `confidence` | 0.827 | |
| `support_all` | 0.877 | +0.050 (+0.010 to +0.096) |
| `support_label_rename` | 0.866 | +0.039 (−0.018 to +0.094) |
| `support_option_order` | 0.805 | −0.023 (−0.066 to +0.018) |
| `agreement_label_rename` | 0.744 | −0.084 (−0.155 to −0.010) |
| `agreement_option_order` | 0.678 | |

With one variant per transform, only `support_all` clearly ranks errors below correct answers better than `confidence`; the binary `agreement_*` signals rank them worse. Keeping the top 70% of cases gives 93.3% accuracy by `support_all` and 91.0% by `confidence`.
