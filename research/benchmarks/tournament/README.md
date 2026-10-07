# Many-option choice: one question, a wider budget, the shortlist, and a tournament

A choice question's options share one `head_max_len` budget, so past roughly 20 labels each label
is cut to a few tokens and some stop being distinguishable (README, "Honest limits"). This compares
the ways around that on three public intent sets, each asked as one choice over its whole label set:

| dataset | labels | test rows | development rows |
|---|---:|---:|---|
| BANKING77 (`mteb/banking77`) | 77 | 3,076 | 600 sampled from train |
| CLINC150 (`clinc/clinc_oos`, `plus`, in-scope only) | 150 | 4,500 | 600 sampled from validation |
| MASSIVE intent, en-US (`SetFit/amazon_massive_intent_en-US`) | 60 | 2,974 | 600 sampled from validation |

The question is the one `research/scripts/build_benchmark_nb.py` asks of BANKING77: the state is
`{"message": text}`, each label becomes an option with its underscores turned to spaces and no
description, and the instruction is ``Which banking intent does `message` express?`` (``Which
intent does `message` express?`` for the other two). Dataset files are pinned to the revisions in
`benchmark.py`.

Strategies, all on the English checkpoint at its defaults (`max_len=512`, `head_max_len=192`):

- `single`: the whole label set in one `predict`.
- `wide`: the same with `head_max_len=512, max_len=1024`, the first remedy the README lists.
- `shortlist`: `predict_shortlist` with `k=20` and `embed_fn_from_agent`, the embedder that needs no
  second model.
- `tournament`: `predict_tournament` at its default `group_size=16`; `tournament-<n>` sets the
  group size.

Every row is a separate call, as a caller would make it, timed after three warm-up calls.

```bash
python research/benchmarks/tournament/benchmark.py --dataset banking77 --split test --out banking77.json
python research/benchmarks/tournament/benchmark.py --dataset clinc150 --split test --out clinc150.json
python research/benchmarks/tournament/benchmark.py --dataset massive --split test --out massive.json
```

The group size was chosen on the development rows only; no tournament ran on a test split before
the default was fixed:

```bash
python research/benchmarks/tournament/benchmark.py --dataset banking77 --split train --limit 600 \
    --strategies single,tournament-8,tournament-12,tournament-16,tournament-24,tournament-32 --out dev.json
```

`pyarrow` reads the parquet files; it is not a dependency of the package.

## Recorded results

Apple M4 Pro, MPS, Python 3.12, torch 2.14, Laya 0.3.27, English checkpoint. `results.json` holds
every number below.

### Test splits: one question against the tournament

| dataset | labels | rows | one question | tournament | rows gained / lost | ECE, one question → tournament |
|---|---:|---:|---:|---:|---:|---|
| BANKING77 | 77 | 3,076 | 0.430 | **0.610** | 779 / 227 | 0.350 → 0.082 |
| CLINC150 | 150 | 4,500 | 0.625 | **0.876** | 1,302 / 172 | 0.297 → 0.057 |
| MASSIVE intent | 60 | 2,974 | 0.515 | **0.569** | 486 / 326 | 0.370 → 0.136 |

The final call reads only the group winners, so its probabilities are over a few labels and its
confidence is calibrated much closer to its accuracy than the one-question answer over every label.

### The other remedies, on 500 test rows per dataset

Accuracy, with ECE in brackets, every strategy on the same rows:

| dataset | one question | `wide` | `shortlist` | tournament |
|---|---:|---:|---:|---:|
| BANKING77 | 0.416 (0.355) | 0.554 (0.308) | 0.224 (0.455) | **0.584** (0.111) |
| CLINC150 | 0.604 (0.320) | 0.680 (0.266) | 0.188 (0.625) | **0.894** (0.054) |
| MASSIVE intent | 0.542 (0.349) | **0.622** (0.270) | 0.184 (0.636) | 0.590 (0.125) |

Raising the budget is the better choice when the labels fit uncut in the wider head: MASSIVE's 60
short labels do at `head_max_len=512`. It falls behind as the label set outgrows the wider head, and
on CLINC150's 150 labels it is still cut to 4 tokens per label. The shortlist lost to one question on
all three sets with `embed_fn_from_agent`; its docstring already says a dedicated bi-encoder will
usually shortlist better, and these numbers say the same about the built-in one.

### Latency

300 test rows per dataset (seed 1), one benchmark process on the GPU at a time:

| dataset | one question, p50 / p95 | tournament, p50 / p95 | calls |
|---|---:|---:|---:|
| BANKING77 | 69 / 82 ms | 145 / 174 ms | 2 |
| CLINC150 | 103 / 109 ms | 187 / 205 ms | 2 |
| MASSIVE intent | 56 / 61 ms | 100 / 110 ms | 2 |

The test and sample accuracy runs above had up to two benchmark processes sharing the GPU, so their
recorded times are left out of `results.json`.

### Group size, development rows only

| group size | BANKING77 | CLINC150 | MASSIVE intent |
|---|---:|---:|---:|
| one question | 0.370 | 0.632 | 0.543 |
| 8 | 0.570 | | |
| 12 | 0.593 | 0.832 | 0.588 |
| **16** | 0.583 | 0.855 | 0.580 |
| 24 | 0.582 | 0.863 | 0.563 |
| 32 | 0.610 | | 0.568 |

Sizes 12 to 24 are within about a point of one another on average, so the default is 16: the largest
size that settles up to 256 labels in a single round, and below the roughly 20 options at which the
README says option text starts to be cut, so longer labels and descriptions keep their tokens too.
Size 8 needs a second round for 77 labels and scored lower. Advancing the top two of each group of 16
instead of one, measured with a local variant of this script, scored 0.602 on BANKING77 and 0.587
on MASSIVE, within noise of one winner, so the library advances one. Size 8 ran on BANKING77 only, and
size 32 did not finish on CLINC150.
