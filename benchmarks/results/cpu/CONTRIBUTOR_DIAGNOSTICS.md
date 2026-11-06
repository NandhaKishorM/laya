# Contributor diagnostic measurements

These results use a contributor-added timing harness and include added stress
cases. They are retained as diagnostics and are excluded from upstream benchmark
acceptance claims. See the [upstream benchmark report](README.md) for the primary
evidence. The checkpoints used here are still Laya's official public weights.

The combined path computes ModernBERT's existing local attention windows in
overlapping blocks, uses packed projections and SDPA in each decision-head layer,
and retains only consumed queries in the final head layer. The eight global
layers of the 22-layer checkpoint remain global; other checkpoints retain their
own configured global/local layer pattern. Weights, calibration and all input
tokens remain unchanged.

Encoder windowing is restricted to the supported Transformers 5 ModernBERT
structure, eager CPU fp32 evaluation, ordinary binary padding masks and sequence
lengths of at least 512. Transformers 4, other encoders, custom hooks, alternate
option layouts, training, gradients, autocast and compilation/export retain the
original encoder path. The head optimizations have their own corresponding
execution and instrumentation guards. There is no persistent prompt cache,
cross-request scratch state, new runtime dependency or remote inference service.

## Protocol

The standard corpus contains the original 60 public `parity_fast.py` fixtures
plus long inputs, one/32 options, padded batches and structured calls: 69 fixtures,
306 question rows. The separate `long-options` suite has four public synthetic
33-option inputs at 795, 910, 1200 and 1499 tokens, matching the shape range of
the original local experiment without using its private inputs or checkpoint.
Its results are reported separately from the standard corpus.

Each fixture has an excluded warmup for each variant and five paired repetitions
with alternating order. Timing covers tokenization, packing, inference and public
decoding, using fixed Torch/interop thread counts. The baseline disables both CPU
optimizations while sharing the same resident weights. `--optimization head` and
`--optimization windows` isolate the contributions for diagnostic comparisons.
Separate fresh processes measure each peak-RSS variant in identical fixture order.

Before timing, the benchmark checks raw option/action logits, calibrated
probabilities, argmax, categorical/boolean answers and public confidence gates.
The original tolerances remain unchanged: `atol=1e-4, rtol=1e-5` for logits,
`atol=1e-5, rtol=1e-5` for probabilities, exact argmax/label/gate agreement,
and at most one four-decimal rounding unit for public numeric fields.
Parameter identities/versions and checkpoint SHA-256 are checked after each run.
Raw timing samples, shapes, source hashes and environments are retained in JSON.
Archived JSON omits redundant per-case percentage fields; those are calculated
from the retained medians.

## Reproduce

```bash
python verify/checkpoints.py --fetch
python tests/test_cpu_window_attention.py
python tests/test_cpu_selective_head.py
python benchmarks/bench_cpu.py --checkpoint multilingual --json multilingual.json
python benchmarks/bench_cpu.py --checkpoint multilingual --suite long-options --json multilingual-long-options.json
python benchmarks/bench_cpu.py --checkpoint multilingual --memory baseline --json rss-baseline.json
python benchmarks/bench_cpu.py --checkpoint multilingual --memory selected --json rss-selected.json
```

Repeat for English and typed-decisions. Run performance commands sequentially,
without competing benchmarks. Choose new output filenames. For ablations, repeat
a case with `--optimization head` and `--optimization windows`; the default is
`--optimization combined`. Historical isolated final-head results remain at
[commit 603e71f8](https://github.com/bahram-po/laya/blob/603e71f8cd3687a363eabfe1a5d2dfc070969f0b/benchmarks/results/cpu-head/README.md)
and are not measurements of this combined path.

## Results on 2026-11-05

All measurements use implementation commit `6290eed206fc2ab3beefedf6abb191319d547d41`,
based on upstream `5a6dc4b809659dab3b9b3062f5e3902db983818a`. Source hashes identify the measured CPU code and harness
independently of later documentation changes and rebases. The [manifest](manifest.json) records source/harness hashes
and the raw/archived report checksums. Every latency run verifies checkpoint hashes
and unchanged parameter identities/versions.

Time columns sum each fixture's median public API latency. These are descriptive
measurements of the specified corpus, not a production traffic mix. Reductions are
`100 * (1 - combined / baseline)`. No original sample or slower fixture is discarded.

### Mac: full standard corpus

Apple M5 Max, 36 GiB RAM, macOS 26.5.2, Python 3.11.12, Torch 2.15.0,
Transformers 5.22.1; four execution threads and one interop thread. Each checkpoint
covers 69 fixtures / 306 question rows with five paired repetitions.

| Checkpoint | Baseline (s) | Combined (s) | Latency reduction | CPU-time reduction | Faster fixtures |
|---|---:|---:|---:|---:|---:|
| [english](mac-english-standard.json) | 70.687 | 62.648 | 11.37% | 20.08% | 61/69 |
| [multilingual](mac-multilingual-standard.json) | 28.158 | 23.503 | 16.53% | 26.44% | 66/69 |
| [typed-decisions](mac-typed-decisions-standard.json) | 65.470 | 58.085 | 11.28% | 18.28% | 65/69 |

### Linux: focused supplemental corpus

GitHub-hosted Ubuntu 24.04 x86-64 runners, four available vCPUs; two execution
threads and one interop thread. Python 3.11.18, Torch 2.15.0+cpu, Transformers
5.22.1. Runner CPU models are retained in [linux-environment.json](linux-environment.json).
The fixed ten-fixture set covers short/long inputs, one/32 options, padding, batches
and structured calls (23 question rows). This smaller corpus is reported separately
and cannot be compared directly with the Mac corpus totals.

| Checkpoint | Baseline (s) | Combined (s) | Reduction |
|---|---:|---:|---:|
| [english](linux-english-latency.json) | 84.780 | 64.885 | 23.47% |
| [multilingual](linux-multilingual-latency.json) | 36.701 | 23.867 | 34.97% |
| [typed-decisions](linux-typed-decisions-latency.json) | 57.337 | 39.109 | 31.79% |

### Separate 33-option workload

Each row aggregates the same four synthetic requests at 795, 910, 1200 and
1499 encoder tokens, with `head_max_len=768`. Captured shapes and option counts
confirm that every request retains all 33 options. These inputs approximate the
original experiment's shape range; they are not the private inputs or checkpoint.

| Platform | Checkpoint | Baseline (s) | Combined (s) | Reduction |
|---|---|---:|---:|---:|
| Mac | [english](mac-english-long-options.json) | 4.720 | 3.688 | 21.87% |
| Mac | [multilingual](mac-multilingual-long-options.json) | 2.142 | 1.651 | 22.91% |
| Mac | [typed-decisions](mac-typed-decisions-long-options.json) | 4.751 | 3.758 | 20.91% |
| Linux | [english](linux-english-long-options.json) | 28.375 | 23.430 | 17.43% |
| Linux | [multilingual](linux-multilingual-long-options.json) | 10.497 | 7.964 | 24.13% |
| Linux | [typed-decisions](linux-typed-decisions-long-options.json) | 18.370 | 13.874 | 24.48% |

### Long-input latency reductions

These are the `long/*` fixtures with two options; all absolute timings and
samples are retained in the corresponding standard/focused JSON reports.

| Platform | Checkpoint | 512 tokens | 1024 tokens | 2048 tokens | 4096 tokens |
|---|---|---:|---:|---:|---:|
| Mac | english | 10.58% | 20.68% | 31.35% | 45.22% |
| Mac | multilingual | 14.97% | 28.09% | 40.62% | 51.70% |
| Mac | typed-decisions | 11.21% | 21.94% | 35.07% | 45.01% |
| Linux | english | 5.79% | 14.06% | 22.85% | 36.48% |
| Linux | multilingual | 11.67% | 20.87% | 36.67% | 47.81% |
| Linux | typed-decisions | 11.25% | 21.87% | 30.42% | 46.57% |

### Peak resident memory

Each variant runs in a fresh process using the same fixture order (69 on Mac;
ten on Linux). RSS is the process high-water mark, including loading, allocator
retention and framework overhead. It is not a live-tensor workspace measurement.

| Platform | Checkpoint | Baseline (MiB) | Combined (MiB) | Change |
|---|---|---:|---:|---:|---:|
| Mac | english | 5583.45 | 2803.61 | -49.79% |
| Mac | multilingual | 3093.05 | 2532.31 | -18.13% |
| Mac | typed-decisions | 4699.97 | 2801.39 | -40.40% |
| Linux | english | 4322.82 | 2849.89 | -34.07% |
| Linux | multilingual | 3435.10 | 2358.59 | -31.34% |
| Linux | typed-decisions | 4344.41 | 2849.88 | -34.40% |

### Unrounded numerical parity

The following maxima cover each platform/checkpoint's main corpus and separate
33-option suite. Every tested argmax, public categorical/boolean answer and
confidence-gate decision agrees. Action logits use the stated combined absolute
and relative tolerance; their scale differs from calibrated probabilities.

| Platform | Checkpoint | Option logits | Action logits | Calibrated probabilities | Public numbers |
|---|---|---:|---:|---:|---:|
| Mac | english | 1.335e-05 | 1.465e-03 | 1.609e-06 | 0.000e+00 |
| Mac | multilingual | 9.537e-06 | 8.545e-04 | 2.086e-07 | 0.000e+00 |
| Mac | typed-decisions | 5.722e-06 | 1.465e-03 | 7.451e-07 | 0.000e+00 |
| Linux | english | 6.199e-06 | 1.465e-03 | 1.490e-06 | 0.000e+00 |
| Linux | multilingual | 1.049e-05 | 6.104e-04 | 4.172e-07 | 0.000e+00 |
| Linux | typed-decisions | 4.590e-06 | 9.766e-04 | 3.576e-07 | 0.000e+00 |

Mathematical equivalence does not imply bit identity. Near ties and exact
rounding/gating boundaries can remain sensitive to floating-point execution order.

### Ablations: Mac multilingual, separate 33-option suite

Each row uses its own paired baseline on the same four inputs. Contributions
are measured independently; their percentages should not be added.

| Optimizations | Baseline (s) | Optimized (s) | Reduction |
|---|---:|---:|---:|
| head | 2.462 | 2.103 | 14.60% |
| windows | 2.178 | 1.793 | 17.69% |
| combined | 2.142 | 1.651 | 22.91% |

### Diagnostic repeats

The [predeclared diagnostic rule](diagnostic-plan.json) selects up to three
fixtures per checkpoint whose initial median regresses by more than 5%, then
runs 21 paired repetitions. Only English `email/7` qualifies: its original
measurement regresses 7.54%, while the separate diagnostic measures 621.453 ms
baseline versus 604.625 ms combined (2.71% lower latency). This is consistent
with timing variability; it does not replace the original corpus result or
establish a general speedup for that fixture. All original samples are retained.
See [the diagnostic samples](mac-english-diagnostic.json).

## Validation

After rebasing onto upstream `3293116c89425c3d04b22aa4d9464fef82e2c858`, all
97 updated shared suites pass locally (828 pytest checks passed, 13 skipped).
The 37 real-checkpoint end-to-end checks, soak checks, lint/compilation and strict
docs build also pass again. The rebased source is `ad44df62`; its CPU implementation,
Agent inference path and benchmark harness are byte-for-byte identical to the
measured source. The original source remains on the
[evidence branch](https://github.com/bahram-po/laya/tree/evidence/cpu-combined-source).

The measured source passes all 96 shared suites locally (818 pytest checks
passed, 12 skipped). The [Linux compatibility matrix](https://github.com/bahram-po/laya/actions/runs/42090048166)
passes on Python 3.10/3.11/3.12/3.13, including Transformers 5.20.2 / Torch 2.14.1
on Python 3.10 and Transformers 5.22.1 / Torch 2.15.0 on the other versions.
The [fork CI workflow](https://github.com/bahram-po/laya/actions/runs/42089928651)
also passes, including Windows, weight-free ONNX export, packaging, lint,
evaluation harnesses and SDK checks. [Docs](https://github.com/bahram-po/laya/actions/runs/42089928791)
and [Docker](https://github.com/bahram-po/laya/actions/runs/42089928703) pass.

Additional local checks on the combined source:

- Real-checkpoint end-to-end: 37 passed. Real-checkpoint ONNX: 2 passed.
- All three checkpoints pass smoke, raw numerical and edge sweeps. SDPA versus
  eager maximum absolute differences remain at or below 1.54e-5, with matching labels.
- Soak passes 200 repeated calls, the additional traced-heap checks, four reload
  cycles, and 48 concurrent calls across four threads.
- The 100-example MASSIVE English regression gate matches the committed baseline:
  accuracy 0.8200, ECE 0.1382, mean confidence 0.9582. The
  [evaluation summary](massive-en-summary.json) retains the configuration and metrics.
- Required lint/compilation checks, 173 packaging checks, and the strict docs
  build pass, with no `griffe:` warnings.

[validation.json](validation.json) records the completed commands' exit statuses,
durations and log checksums. The [Linux benchmark workflow](https://github.com/bahram-po/laya/actions/runs/42090052470)
completed all three checkpoint jobs against the measured source commit. Its extra
workflows are fork-only and are not included in the upstream PR. CUDA-specific
suites were not run on the CPU hosts.

The separate fork [Security workflow](https://github.com/bahram-po/laya/actions/runs/42089928457)
flags pre-existing `subprocess.run` calls in `laya/video.py`. Both that file and
the scanner workflow are unchanged from upstream. Upstream Actions require
maintainer approval for this first contribution; independent fork validation is
not an upstream CI pass.

### Reproduce the supplemental checks

```bash
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LAYA_DEVICE=cpu LAYA_LAB_MODELS="$PWD/models"
python scripts/test_suites.py
python tests/test_local_e2e.py models
python verify/laya_smoke_test.py --models models --device cpu
python verify/numerics_check.py
python verify/edge_sweep.py --models models --device cpu
python verify/soak_check.py --models models --device cpu --calls 200 --cycles 4 --threads 4
python research/eval/laya_eval.py --model models/laya --device cpu --langs en --out fresh.json
python research/evals/check_regression.py fresh.json research/results/eval_english_51_languages.json research/evals/thresholds.json
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
zensical build --strict --clean
```

For the Linux focused benchmark, use `--threads 2` and this fixed case selection
for latency and each fresh-process memory variant:

```bash
python benchmarks/bench_cpu.py --checkpoint multilingual --threads 2 \
  --case triage/0 --case long/512 --case long/1024 --case long/2048 --case long/4096 \
  --case options/1 --case options/32 --case batch/padded \
  --case structured/mixed --case structured/batch --json linux-focused.json
```

Real-checkpoint ONNX uses the existing tests with only `MODEL_ID` pointed at the
verified local English checkpoint; test logic and tolerances are unchanged:

```python
from pathlib import Path
import pytest

class PinnedLocalCheckpoint:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if Path(item.module.__file__).name == "test_onnx.py":
                item.module.MODEL_ID = str(Path("models/laya").resolve())

raise SystemExit(pytest.main(["tests/test_onnx.py", "-q"], plugins=[PinnedLocalCheckpoint()]))
```
