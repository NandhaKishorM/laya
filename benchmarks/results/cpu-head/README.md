# CPU final-head measurements

The eager CPU fp32 path computes only the last head layer's option-marker outputs
and token zero. All sequence positions remain keys and values. The earlier head
layers, encoder, weights, calibration and serialization format are unchanged.

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
include the public API, tokenization and decoding. Four Torch threads and one
interop thread are fixed for the process; no settings change between requests.

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
