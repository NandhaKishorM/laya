# Research

Benchmark harnesses and raw results for the Laya checkpoints. This branch is the evidence behind
the numbers quoted in the main README — nothing here is imported by the `laya` package.

## Scripts

| file | what it does |
|---|---|
| `scripts/laya_benchmark_colab.ipynb` | the full head-to-head on a Colab T4: typed-decisions, MASSIVE intent + scenario (14 languages), XNLI (15), English suites, latency, option-order robustness, calibration repair. Writes one JSON. |
| `scripts/build_benchmark_nb.py` | generator for that notebook (edit here, not the `.ipynb`) |
| `scripts/bench_local.py` | CPU sweep: MASSIVE intent across **all 51 languages**, plus typed-decisions on all three checkpoints |
| `scripts/bench_apps.py` | the six application workflows (support triage, email + phishing, guardrails, RAG relevance, moderation, model routing) plus the datasets where public Jev numbers exist |
| `scripts/bench_latency.py` | inference speed including what routing costs: detection overhead, hot path, cold-swap, mixed-language throughput at several `max_loaded` settings |
| `scripts/make_plots.py` | renders `assets/laya_benchmark.png` from the result JSONs |

Everything runs with `USE_TF=0` — `transformers` probes for TensorFlow at import, and when TF is
installed its abseil runtime can deadlock model construction on macOS/Python 3.9.

## Running the harnesses

The checkpoints come from `LAYA_MODELS` (default `~/laya_models`, a directory holding `laya/`,
`laya-multilingual/` and `laya-typed-decisions/`), so a plain checkout can point them at the
models `setup_laya.sh` already downloaded:

```bash
# from the repository root
LAYA_MODELS=$PWD/models python research/scripts/bench_local.py --langs 10 --per-lang 60 --skip-b
LAYA_MODELS=$PWD/models BENCH_N=80 python research/scripts/bench_apps.py
LAYA_MODELS=$PWD/models python research/scripts/bench_latency.py
```

| option | effect |
|---|---|
| `bench_local.py --langs N` | caps the language list (sorted, so `10` is the first ten codes; `0` = all 51) |
| `bench_local.py --per-lang N` | cases per language (default 120) |
| `bench_local.py --skip-a` / `--skip-b` | run only the typed-decisions half, or only the MASSIVE half |
| `bench_apps.py` with `BENCH_N` | cases per application suite (default 400) |

`bench_local.py` and `bench_apps.py` write `research/*_benchmark_results.json`, which is
gitignored — the curated runs live in `results/`. Set `HF_HOME` inside the checkout to keep the
dataset cache local.

Both scripts were adjusted so they run from a plain checkout: the model root is now
`LAYA_MODELS` instead of a hardcoded `~/laya_models`, and their `sys.path` entry points at the
repository root rather than a directory that does not exist. Results from a reduced CPU re-run,
and the two figures that match the published numbers exactly, are in
[`BENCHMARKS.md`](../BENCHMARKS.md#independent-reproduction-macos-intel-cpu-only).

## Results

| file | contents |
|---|---|
| `results/t4_colab_benchmark.json` | 17,416 questions on one T4, both checkpoints, identical questions per model |
| `results/cpu_51_language_sweep.json` | 51 languages x 2 checkpoints, MASSIVE intent, 20 options |

## Headline findings

**Routing takes Laya from 23 to 45 of 51 languages.** On MASSIVE intent (20 options, random =
0.050) the English checkpoint macro-averages 0.227 and clears 3x random on 23 of 51 languages;
the multilingual checkpoint reaches 0.366 and clears it on 45.

**The English checkpoint's confidence gives no warning when it cannot read the input.** Khmer:
0.000 accuracy at 0.952 mean confidence. Macro ECE 0.733 across 51 languages, with mean
confidence never dropping below 0.885 at any accuracy level. This is why routing has to happen
*before* the forward pass — confidence gating cannot catch it.

**Both checkpoints ship over-confident.** Refitting one temperature per (question type, option
count) on held-out data moves mean ECE 0.466 -> 0.081 (`laya`) and 0.314 -> 0.106
(`laya-multilingual`, which ships with no fitted temperatures at all).

**The base checkpoints are near chance on typed-decisions zero-shot** — 0.362 and 0.352 against
a 0.318 random baseline and a 0.461 majority-class baseline. The published 0.766 belongs to the
checkpoint fine-tuned on that benchmark's own training split.

**Speed.** 32.8 ms for one question and 7.2 ms/question at batch 10 on a T4; 103–332 questions/s
batched.

## On comparisons with Jev

There is no TypeSafe API credential in this project, so **Jev was never run here**. Every Jev
figure quoted is third-party published, with different sample sizes and prompts:

- [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) — AG News 0.910,
  Banking77 0.870, DAIR Emotion 0.480 (Brier 0.846, NLL 5.588, zero probability on the true label
  for 16% of examples)
- [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark) — ECE
  0.246 (worst in that study), banking77 0.763, option-order flip rate 13%, latency 264–276 ms p50

Treat those as indicative, not a controlled head-to-head.
