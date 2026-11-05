# CPU final-head measurements

The eager CPU fp32 path computes only the last head layer's option-marker outputs
and token zero. All sequence positions remain keys and values. The earlier head
layers, encoder, weights, calibration and serialization format are unchanged.

## Apple M5 Max, 2026-11-05

macOS 26.5.2, Python 3.11.12, PyTorch 2.15.0, Transformers 5.22.1, 36 GiB
unified memory; four execution threads and one interop thread. The source for
these full-corpus runs is `eedbe4dadbf9caba5002f2ce8b02ab2b3082f3c4`, based on
upstream `5a6dc4b809659dab3b9b3062f5e3902db983818a`. The baseline runs that same
source with only the selection predicate disabled. Each checkpoint has 69
fixtures, 306 question rows, one excluded warmup per variant/fixture and five
paired timed repetitions.

The time columns sum the median public-call latency of each fixture: one pass
through this particular corpus, not a production traffic mix or throughput test.

| Checkpoint | Full head (s) | Selected head (s) | Reduction | Fixtures with lower median |
|---|---:|---:|---:|---:|
| [English](english-m5max.json) | 59.396 | 57.397 | 3.37% | 64/69 |
| [Multilingual](multilingual-m5max.json) | 24.499 | 23.200 | 5.30% | 65/69 |
| [Typed-decisions](typed-decisions-m5max.json) | 64.783 | 62.393 | 3.69% | 59/69 |

Individual fixtures include slower measurements; all samples are retained. The
diagnostic repeats inspect those cases without substituting their timings into
the full-corpus totals above.

| Checkpoint | Encoder sequence tokens | Full head (ms) | Selected head (ms) | Reduction |
|---|---:|---:|---:|---:|
| English | 2,048 | 3,048.48 | 2,720.44 | 10.76% |
| English | 4,096 | 8,261.15 | 7,389.59 | 10.55% |
| Multilingual | 2,048 | 1,436.55 | 1,297.72 | 9.66% |
| Multilingual | 4,096 | 4,447.22 | 4,203.95 | 5.47% |
| Typed-decisions | 2,048 | 3,110.26 | 3,011.80 | 3.17% |
| Typed-decisions | 4,096 | 9,900.46 | 8,669.34 | 12.43% |

All 918 question rows retain their argmax, public answers and confidence-gate
decisions. Every public numeric output is also identical in these runs. Maximum
unrounded differences (not rounded public confidence values):

| Checkpoint | Option logits | Action logits | Calibrated probabilities |
|---|---:|---:|---:|
| English | 3.815e-6 | 4.883e-4 | 1.192e-7 |
| Multilingual | 3.815e-6 | 3.662e-4 | 1.490e-7 |
| Typed-decisions | 9.537e-7 | 9.766e-4 | 1.192e-7 |

Action logits have a different scale from probabilities; their check includes
the declared relative tolerance below. No tolerance was changed after the runs.
Mathematical equivalence is not bit identity: a nearly tied decision or a value
at a rounding/gating boundary can change with floating-point execution order.

### Peak resident memory

Each cell comes from a fresh process running the same 69 fixtures in order.
This is the process high-water mark, including model loading, startup and allocator
retention, not just attention workspace. The English run uses slightly more peak
memory; these measurements do not establish a universal memory reduction.

| Checkpoint | Full head (MiB) | Selected head (MiB) | Change |
|---|---:|---:|---:|
| English | 4,527.23 | 4,566.69 | +0.87% |
| Multilingual | 4,253.97 | 2,619.33 | -38.43% |
| Typed-decisions | 5,542.47 | 5,460.92 | -1.47% |

The corresponding `*-m5max-memory-baseline.json` and
`*-m5max-memory-selected.json` files retain the high-water mark after loading and
after every fixture.

### Diagnostic repeats

The initial full-corpus measurements include individual wall-clock regressions
as large as 19%. Before repeating, [the diagnostic plan](diagnostic-plan.json)
fixed the three largest relative regressions per checkpoint and 21 paired
repetitions. The updated harness also measures process CPU time. All nine cases
have lower median wall and CPU time in these repeats, consistent with timing
variability in the initial samples. This is a targeted diagnostic, not an
independent estimate of overall speedup. Original samples and corpus totals
remain unchanged.

| Checkpoint | Fixture | Wall-time reduction | CPU-time reduction |
|---|---|---:|---:|
| [English](english-m5max-diagnostic.json) | moderation/10 | 3.66% | 4.33% |
| English | email/10 | 2.78% | 3.42% |
| English | email/11 | 3.63% | 4.23% |
| [Multilingual](multilingual-m5max-diagnostic.json) | moderation/10 | 5.60% | 6.56% |
| Multilingual | router/6 | 4.48% | 4.57% |
| Multilingual | email/11 | 6.12% | 6.76% |
| [Typed-decisions](typed-decisions-m5max-diagnostic.json) | triage/0 | 3.60% | 2.89% |
| Typed-decisions | triage/4 | 2.74% | 2.58% |
| Typed-decisions | triage/6 | 3.65% | 3.87% |

## Correctness and compatibility

Completed on the Mac environment above:

| Check | Result |
|---|---|
| Shared CI/release runner | All 95 suites passed; pytest portion: 818 passed, 12 skipped |
| Additional selective-head/eval/shortlist pytest run | 216 passed, 36 subtests passed |
| Local real-checkpoint end-to-end suite | 37 passed |
| Real-checkpoint ONNX parity | 2 passed |
| Weight-free batch-dynamic ONNX export | 99 passed |
| Checkpoint smoke, numerical and edge sweeps | All three pinned checkpoints passed |
| Soak check | Passed: 200 repeated calls, 4 load/unload cycles, 48 concurrent calls across 4 threads |
| MASSIVE English regression gate | Accuracy 0.8200, ECE 0.1382, mean confidence 0.9582; all equal to the committed baseline |
| Lint, byte compilation, packaging, strict docs | Passed; 173 packaging checks, no `griffe:` warnings |

The accuracy gate uses the repository's 100-example English MASSIVE sample,
20 options and seed 13. It is separate from the synthetic parity corpus and is
not a claim of a new full-dataset accuracy evaluation. Its
[summary](massive-en-summary.json) records the configuration and metrics.

```bash
python scripts/test_suites.py
python -m pytest tests/test_cpu_selective_head.py tests/test_evals.py tests/test_evals_shortlist.py -q
python tests/test_local_e2e.py models
python -m pytest tests/test_onnx.py -q
python tests/test_onnx_export_batch.py
python verify/laya_smoke_test.py --models models --device cpu
LAYA_LAB_MODELS="$PWD/models" LAYA_DEVICE=cpu python verify/numerics_check.py
python verify/edge_sweep.py --models models --device cpu
python verify/soak_check.py --models models --device cpu --calls 200 --cycles 4 --threads 4
python research/eval/laya_eval.py --model models/laya --device cpu --langs en --out massive-en.json
python research/evals/check_regression.py massive-en.json research/results/eval_english_51_languages.json research/evals/thresholds.json
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
python tests/test_packaging.py
zensical build --strict --clean
```

For the real-weight ONNX run, a pytest collection plugin set `test_onnx.py`'s
`MODEL_ID` to the already verified local `models/laya` checkpoint. The two tests
and their tolerances were unchanged; the command above otherwise downloads the
same English model. CUDA-specific suites were not run on this CPU host.

The shared runner also passed all 95 suites on macOS with Python 3.10.20,
PyTorch 2.14.1 and Transformers 5.20.2 (818 pytest checks passed, 12 skipped).
The Windows and Linux Python 3.12/3.13 test jobs passed in the
[fork's upstream CI workflow](https://github.com/bahram-po/laya/actions/runs/42065875065).
These are individual completed jobs; they are not a claim that every job in
that workflow or the upstream PR's CI has completed.

## Reproduce

From the repository root, with the project's dependencies installed:

```bash
python verify/checkpoints.py --fetch
python tests/test_cpu_selective_head.py
python benchmarks/bench_cpu_head.py --checkpoint english --json english.json
python benchmarks/bench_cpu_head.py --checkpoint multilingual --json multilingual.json
python benchmarks/bench_cpu_head.py --checkpoint typed-decisions --json typed-decisions.json
python benchmarks/bench_cpu_head.py --checkpoint multilingual --memory baseline --json rss-baseline.json
python benchmarks/bench_cpu_head.py --checkpoint multilingual --memory selected --json rss-selected.json
python benchmarks/bench_cpu_head.py --checkpoint english --repeats 21 --case moderation/10 --case email/10 --case email/11 --json english-diagnostic.json
python benchmarks/bench_cpu_head.py --checkpoint multilingual --repeats 21 --case moderation/10 --case router/6 --case email/11 --json multilingual-diagnostic.json
python benchmarks/bench_cpu_head.py --checkpoint typed-decisions --repeats 21 --case triage/0 --case triage/4 --case triage/6 --json typed-diagnostic.json
```

Run each command separately, without another benchmark competing for CPU. Repeat
the memory commands for each checkpoint; every memory variant needs a fresh process.
The output must not already exist.

The benchmark reuses the 60 public state/question fixtures in `parity_fast.py`,
then adds 512/1024/2048/4096-token requests, one and 32 options, a padded sorted
batch, and mixed-type structured single/batch calls. These synthetic fixtures
measure execution agreement, not task accuracy or calibration quality. Long
inputs explicitly override the checkpoint's default token budget.

One untimed call per variant per fixture checks the unrounded logits, action
logits, calibrated probabilities and public answers, and warms both paths. Five
timed repetitions alternate execution order across rounds and fixtures. Timings
include the public API, tokenization and decoding. The thread count (four by
default, `--threads 2` for the Linux runs) and one interop thread are fixed for
the process; no settings change between requests.

The baseline disables only the final-head selection predicate, leaving the full
forward in place. It shares the exact resident weights with the selected path.
JSON artifacts record every timing sample, forward shape, maximum numerical
delta, package versions, fixture/source hashes and pinned checkpoint SHA-256.
Parameter identities/versions and checkpoint hashes are checked after the run.
Comparisons use `atol=1e-4, rtol=1e-5` for logits (including action logits),
`atol=1e-5, rtol=1e-5` for calibrated probabilities, exact argmax/label/gate
agreement, and at most one 4-decimal rounding unit for public numeric fields.

Peak RSS is the process high-water mark, including loading and allocator retention;
it is not just tensor workspace. The memory runs follow the same fixture order.
No private inputs, weights or customer-derived predictions are included in these
artifacts. This path adds no service, shared request cache or dependency.

All full-corpus latency artifacts were generated by the harness at `eedbe4da`,
SHA-256 `3b12fa373a73f19f6904a47da62f8694acd1befc8db4c5a96eaf88d1cbef7081`.
The subsequent harness adds only named-case filtering and process CPU timers;
the diagnostic artifacts record its SHA-256
`66bd1b44017fc2d24b08b11e5fdf72aca2b397e6ad2b4c352077a72869254498`.
Every artifact uses the same production `laya/common.py`, SHA-256
`66126698100f3955b5b0ee1eed4ea034c9075930090d1ee9c8e536b67f3f7f46`.
