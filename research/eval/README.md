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
`build_sequence` (unchanged since `v0.2.0`). It is in the multilingual inference path
between `laya 0.2.0` and `0.3.6` and is **not** reconciled. Flagged rather than hidden.

Related: **`head_max_len` is load-bearing for accuracy**, not just for option
truncation. The english checkpoint at its shipped `head_max_len=192` scores 0.82;
forcing 256 or 512 drops it to 0.79.

## Tests

`research/eval/test_laya_eval.py` covers the pure functions and runs offline — no
checkpoint, no network:

```bash
python research/eval/test_laya_eval.py     # 47 passed, 0 failed
```

It pins the upstream constants (seed 13, 20 options, the exact instruction string),
the determinism of the sampler, that a fresh RNG per language is used, and the
metric arithmetic, including the `confidence == 0.0` bin boundary that this harness
shares with `bench_local.py` and `laya.common.ece_score`.

## Limits

* MASSIVE intent only. The same shape applies to `scenario` and to XNLI, but neither
  is wired up here.
* `per_lang=100` is the published setting, not a statistical one. Per-language ECE on
  100 cases is noisy; raise `--per-lang` and say so when quoting a number.
* The English checkpoint collapses on non-Latin scripts (see `BENCHMARKS.md`), so a
  low score in one language is not by itself evidence of a misroute — check
  `laya.lang.analyse` for the script before concluding which checkpoint was used.
* `confidence == 0.0` is not binned, matching `laya.common.ece_score` and
  `bench_local.py`. If [#39](https://github.com/NandhaKishorM/laya/pull/39) lands
  with a different boundary, this should follow it.
