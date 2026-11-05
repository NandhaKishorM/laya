# Combined CPU inference measurements

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
`--optimization combined`. Historical isolated final-head results remain in
[cpu-head](../cpu-head/README.md) and are not measurements of this combined path.

Completed measurements and validation results will be added after the runs finish.
