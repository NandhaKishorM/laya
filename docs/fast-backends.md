# Fast backends

## TileLang portability

All five kernels in `laya/tl_kernels.py` can execute on TileLang's CPU (`c`)
target through `compile_cpu`. This is an explicit, scalar fp32 specialization,
not a CPU acceleration backend for `Agent.accelerate`. It requires TileLang and
a local C++ compiler. It introduces no runtime service dependency.

```python
import torch
from laya import tl_kernels as K

A = torch.randn(17, 67)
W = torch.randn(70, 67)
b = torch.randn(70)
C = torch.empty(17, 70)
kernel = K.compile_cpu(K.gemm_kernel, 70, 67, bias=True, act="gelu")
kernel(A, W, b, C)
```

`compile_cpu(factory, *args, **kwargs)` takes the same shape and operation options
as the five GPU factories. It selects `cpu=True`, `dtype="float32"`, target `c`,
and disables vectorization. All floating inputs and outputs must be contiguous
CPU float32 tensors; attention lengths remain int32. Convert 16-bit activations
with `.cpu().float().contiguous()` before calling it. LayerNorm still updates its
residual tensor in place, and RoPE still updates the Q/K columns in place.
Retain the compiled kernel to reuse its dynamic dimensions.

CPU specialization replaces fragment and shared allocations with local buffers,
uses serial `T.grid` loops, and omits attention's GPU swizzle annotation. GEMMs
use TileLang's scalar CPU implementation and reductions use local buffers.
The original tiled algorithms, padding predicates, attention masks, and online
softmax remain shared with the GPU implementation. CPU intermediates stay fp32,
including attention probabilities; GPU intermediates retain their original dtype.
No CPU speedup or full-model CPU backend is claimed.

### Re-probe on TileLang 0.1.14

Observed on Linux, Python 3.12.13, torch 2.11.0+cu130, TileLang 0.1.14,
and an RTX 4070 Ti SUPER. The earlier portability concern applies to compiling
the **unchanged GPU specialization**, not to the possibility of a CPU lowering.
At base commit `fa9a2a7`, this documentation page was absent from the checkout.

The test compiles `factory.get_tir(...)` with `target="c"` and the GPU's `FAST`
pass configuration. These are the exact diagnostic texts (source locations and
stack traces omitted). Both bf16 and fp16 were probed for every kernel.

| Kernel and probe dimensions | First bf16 failure | First fp16 failure |
| --- | --- | --- |
| `gemm_kernel(128, 64)` | `Check failed: layout_map.count(buffer) != 0 (0 vs. 0) : The layout for fragment C_l can not be inferred correctly.` | Same |
| `gemm_geglu_kernel(64, 64)` | ``CPU fill only supports local and global buffers, but got dst scope `local.fragment`.`` | Same |
| `add_ln_kernel(128)` | ``CPU reduce only supports local src and local/local.var dst buffers, got src scope `local.fragment` and dst scope `local.fragment`.`` | Same |
| `rope_kernel(2, 64)` | `Cannot convert type bfloat16 to C type` | C++ compiler: `error: no matching function for call to ‘vec_type<float, 4>::vec_type(half4&)’` |
| `attn_kernel(1, 64, 2, 64)` | `Check failed: layout_map.count(buffer) != 0 (0 vs. 0) : The layout for fragment s_c can not be inferred correctly.` | Same |

The first four fragment/reduction failures are `tvm.error.InternalError`.
The bf16 codegen failure is also an `InternalError`. FP16 RoPE raises
`RuntimeError: Compilation Failed!` followed by the compiler invocation and
source; the diagnostic above is emitted on stderr. Its generated vector casts
also fail converting float vectors back to half vectors.

To isolate dtype support from fragment support, the tests compile **each** kernel
again with `cpu=True` (local buffers and serial loops), retaining bf16 and disabling
vectorization. All five then fail with exactly:

```text
Cannot convert type bfloat16 to C type
```

Thus fragments are the first blocker for GEMM, GEGLU and attention, fragment
reductions for LayerNorm, and bf16 is independently a blocker for all five.
RoPE has no fragment allocation or reduction; GEMM and GEGLU have no explicit
`T.reduce_*` operation. Attention's reductions are initially masked by its layout
failure. Separate fp32-only fragment probes reproduce the fill error and the
reduction error for both `reduce_sum` and `reduce_max`, without any GEMM or bf16.
Replacing scopes alone also produced this semantic error for the GEMM
buffer (the other affected buffers were `Ci`, `x`, and `s`):

```text
[Tilelang Semantic Check] Local buffer `C_l` is indexed by T.Parallel loop variable `i`. Local buffers are thread-private and do not participate in parallel layout inference. Use T.serial/T.vectorized/T.unroll for per-thread local indexing, or T.alloc_fragment when the indexed dimension should be distributed across threads.
```

The serial fp32 specialization removes these blockers. Diagnostic assertions are
version-pinned to 0.1.14 and skip on another version, where the errors should be
re-probed. Numerical CPU tests continue to run on other versions.

### Numerical checks

Run `python -m pytest tests/test_fast_cpu.py -q -s`. On the environment above:
**58 passed**. CPU-only checks do not require CUDA; only the GPU comparisons skip
without it. Coverage includes uneven GEMM M/N/K tiles, all GEMM epilogues, GEGLU,
all residual/bias combinations, large residual values, RoPE position wraparound
and untouched V columns, static/dynamic attention shapes, sliding windows,
partial tiles, unequal lengths, empty sequences, and finite padding outputs.

The seed is 1234. GPU comparisons use identical input values rounded to bf16 or
fp16, then promoted to fp32 for CPU execution. These are absolute tolerances for
the bounded fixtures, not a guarantee for arbitrary magnitudes or model depth.
Attention comparisons use valid query rows, as in `tests/test_fast.py`.

| Kernel | Max CPU vs fp32 reference | Max CPU vs GPU (both dtypes) | CPU/GPU tolerance |
| --- | ---: | ---: | ---: |
| GEMM | 2.38419e-7 | 0.00770831 | 0.05 |
| GEGLU | 2.98023e-8 | 0.000208303 | 0.05 |
| LayerNorm | 1.07288e-6 | 0.0156183 | 0.05 |
| RoPE | 0 | 0.0130053 | 0.05 |
| Attention | 5.96046e-7 | 0.00377572 | 0.02 |

CPU/reference tolerance is 2e-5 (2e-6 for RoPE). Residual stream updates are
exactly equal, including the no-residual case.

### GPU preservation evidence

The `cpu` keyword defaults to false. Existing bf16/fp16 defaults, GPU allocation
scopes, parallel loops, swizzles and `FAST` options are unchanged. Default fp32
GPU calls still raise `ValueError`.

Before/after runs used the original module extracted with
`git show fa9a2a7:laya/tl_kernels.py` and the changed module, respectively.
The original was loaded as `laya.tl_kernels` via `importlib` for baseline runs;
all other Laya code and the Python environment stayed the same.

- `python -m pytest tests/test_fast.py -q`, with the full-forward tests configured
  to load the cached English checkpoint identified below:
  **13 passed before; 13 passed after**. Initially without a checkpoint it was
  11 passed / 2 skipped. Both full runs emitted the checkpoint's existing
  temperature-clamping warning (`choice:11+=0.10058280825614929 -> 0.5`).
- Seeded capture of the existing kernel tests: **24 output tensors bit-identical**,
  including residual updates; maximum before/after difference **0**.
- Generated CUDA source for all five probe shapes above, in both dtypes:
  **10/10 byte-identical**. This also covers RoPE, absent from the original fast suite.
- `python benchmarks/parity_fast.py --model "$MODEL" --dtype bf16 --json ...`
  and the equivalent fp16 command: **288 questions over 60 states per dtype**.
  All before/after JSON records (fp32, stock, and fast probabilities) compare
  exactly equal; maximum before/after probability difference **0**.

`MODEL` was the cached `convaiinnovations/laya` English snapshot
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; runs used `HF_HUB_OFFLINE=1`.
No multilingual or typed-decisions checkpoint comparison is claimed here.

| dtype | type | n | Max fast-stock, before = after | Fast/stock argmax agreement, before = after |
| --- | --- | ---: | ---: | ---: |
| bf16 | choice | 48 | 0.0310 | 47/48 |
| bf16 | noul | 180 | 0.0756 | 180/180 |
| bf16 | score | 60 | 0.0152 | 60/60 |
| fp16 | choice | 48 | 0.0069 | 48/48 |
| fp16 | noul | 180 | 0.0092 | 180/180 |
| fp16 | score | 60 | 0.0040 | 60/60 |


### Repository checks

All commands used `/home/ckl/projects/S/laya/.venv/bin/python`; `ruff` and
`zensical` came from that virtualenv's `bin` directory.

| Command | Result |
| --- | --- |
| `ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120` | `All checks passed!` |
| `python -m compileall -q laya/ tests/` | Exit 0, no output |
| `python tests/test_router.py` | 703 passed, 0 failed |
| `python tests/test_criteria.py` | 198 passed, 0 failed |
| `python tests/test_hooks.py` | 240 passed, 0 failed |
| `python tests/test_hooks_api.py` | 415 passed, 0 failed |
| `python tests/test_packaging.py` | 131 passed, 0 failed |
| `uv pip install --python /home/ckl/projects/S/laya/.venv/bin/python -r requirements-docs.txt` | Checked 3 packages (already installed); virtualenv has no pip |
| `zensical build --strict --clean` | `No issues found`; no `griffe:` lines |

The new suite is registered as requiring the optional TileLang extra and a C++
compiler in the packaging test's existing exemptions. The API contract test pins
the additive keyword-only CPU selector, unchanged GPU dtype default, and
`compile_cpu` signature without importing TileLang into the base CI environment.


## AOTInductor

`DecisionModel` can be exported, compiled into a `.pt2` package, and loaded with
`torch._inductor.aoti_load_package`. The package returns both decision logits and action logits.
Tokenization, padding, temperature calibration and answer formatting remain the caller's job;
this does not add an `Agent` backend or change its default execution path.

The [closed PR #472](https://github.com/NandhaKishorM/laya/pull/472) documented the earlier dtype
blocker. The action head's pooled state and confidence features are computed in fp32, even
when a model's weights are explicitly bf16. Without autocast, the first action linear therefore
received an fp32 input and bf16 weights. The forward now casts the concatenated input to the
head's weight dtype **only outside autocast**. Softmax, entropy and the returned decision logits
retain their fp32 calculations. Existing fp32-parameter eager inference, including fp16/bf16
AMP, keeps its numerics; mixed weight/autocast dtypes also retain autocast's original conversion.

Export a separate evaluation copy converted to bf16, outside autocast. Declare the token
axis as `16 * Dim("tokens16", ...)` to satisfy the attention alignment guards. Rows and marker
counts can be dynamic independently. Do not assume an export captured under autocast can be
packaged outside that context: use explicit weight dtypes for this recipe.

### Reproduce offline

The assert-based check uses a tiny randomly initialized ModernBERT by default, without network
access. Pass a local checkpoint directory for a real-model measurement. Missing CUDA,
AOTInductor APIs or a C++ compiler produces an explicit `SKIP`; compilation and parity failures
on a supported installation fail the check.

```bash
python scripts/check_aoti.py --output-dir /tmp/laya-aoti-smoke
HF_HUB_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/laya-aoti/cache \
  python scripts/check_aoti.py --model /path/to/local/multilingual \
  --output-dir /tmp/laya-aoti
```

The output directory holds `decision.pt2`, `results.json`, `inputs.pt` and `eager.pt`. The JSON
includes full `nvidia-smi` snapshots, export/package/load times, package bytes, latency and
maximum absolute logit/probability deltas for both heads. Binary artifacts and compiler caches
are deliberately kept out of the repository.

### Measured on RTX 4070 Ti SUPER

2026-10-03, Linux x86-64, Python 3.12.13, torch 2.11.0+cu130, transformers 5.17.0,
NVIDIA driver 615.71.09, 16,376 MiB VRAM. Checkpoint: `convaiinnovations/laya`,
`multilingual` subdirectory at revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
The baseline is main `fa9a2a7`. Full machine snapshots and unrounded measurements are in
[`aoti_multilingual_rtx4070.json`](https://github.com/NandhaKishorM/laya/blob/main/benchmarks/results/aoti_multilingual_rtx4070.json).

| Measurement | Before | After |
|---|---:|---:|
| Export | 5.00 s | 3.89 s |
| Packaging | failed after 16.59 s | 60.02 s |
| Package size | no artifact | 645,729,621 bytes (615.82 MiB) |
| Load in the already initialized process | unavailable | 0.447 s |
| Load in a fresh process with an empty cache | unavailable | 4.867 s |

The reproduced failure is at packaging, after successful export:
`mat1 and mat2 must have the same dtype, but got Float and BFloat16`.
Each run used a separate initially empty Inductor cache; AMP compilation ran before packaging,
so the package timing is not a fresh-interpreter cold-start measurement.
The fresh-process check loaded only the package and saved inputs (no checkpoint), with
`torch.compile` and packaging entry points blocked. It reproduced both heads' answers.
PyTorch still compiled a small CPU AVX capability probe into the initially empty cache;
no model graph or CUDA kernel was compiled. A blanket "no compiler activity at load" claim
would therefore be inaccurate on this runtime.

The same saved inputs contain seven decision rows across three batches, with real tokenized
billing/refund text, choice/score/noul questions, padding and unequal valid marker counts.
Each latency is the median of five groups of 30 forwards after ten warmups, using synchronized
wall-clock timing. `torch.compile(dynamic=True)` uses the default mode. No explicit CUDA graphs,
tokenization, export or compilation are included in these latency numbers.

| Rows × tokens × marker slots | AMP eager before → after (ms) | AMP compiled before → after (ms) | Explicit bf16 eager (ms) | Explicit bf16 compiled (ms) | AOTI bf16 (ms) |
|---|---:|---:|---:|---:|---:|
| 2 × 128 × 3 | 15.1173 → 14.1200 | 6.7117 → 6.9076 | 13.9635 | 4.8835 | 2.2674 |
| 1 × 256 × 3 | 15.4687 → 14.4156 | 6.7274 → 5.4772 | 14.7994 | 4.7250 | 2.4154 |
| 4 × 160 × 5 | 14.8452 → 14.5384 | 7.3872 → 5.5934 | 14.7115 | 5.1393 | 3.6018 |

AMP here means fp32 parameters with bf16 autocast. Explicit bf16 means weights and the residual
stream are bf16, without autocast; use those columns for the closest execution comparison to
the package. Explicit bf16 eager/compiled and AOTI were unavailable before this fix.

| Comparison, maximum over all seven rows | Decision logits | Decision probabilities | Action logits | Action probabilities |
|---|---:|---:|---:|---:|
| Existing eager before vs after, fp32 and fp16/bf16 AMP | 0 | 0 | 0 | 0 |
| AOTI vs explicit bf16 eager | 0.5625 | 0.00659859 | 0 | 0 |
| AOTI vs existing bf16 AMP eager | 0.4375 | 0.00769910 | 8.0 | 0 |

Both decision and action argmax agree on **7/7 rows** for both AOTI comparisons. Probabilities
are raw softmax outputs, without calibration. The action distribution is saturated on these
inputs, so its zero probability delta does not imply identical underlying logits against AMP.
The explicit bf16 compiled forward also differs from explicit bf16 eager (maximum decision
logit delta 0.5, probability delta 0.00659859). Reduced-precision compilation is not bit-exact.

The GPU was shared with desktop applications and other Python jobs; CPU compilation was also
shared, and clocks were not locked. These are observed timings, not an isolated speedup claim.
`nvidia-smi` snapshots bracketing the runs:

| Run / snapshot | GPU utilization | Used VRAM | Power | Temperature / state |
|---|---:|---:|---:|---|
| Before / start | 28% | 2,817 MiB | 12 W | 33°C / P8 |
| Before / end | 10% | 8,560 MiB | 23 W | 36°C / P3 |
| After / start | 0% | 2,634 MiB | 12 W | 33°C / P8 |
| After / end | 73% | 6,476 MiB | 160 W | 42°C / P2 |

### Remaining limits

- The artifact is specific to this checkpoint, precision, PyTorch/runtime stack and GPU target;
  portability to other hardware or PyTorch versions was not tested.
- This check exports rows 1–8, tokens 32–512 in multiples of 16, and marker slots 2–8. It exercises
  three shapes, including shapes different from the export example, rather than every point
  in those ranges. One valid option can use padded marker slots; a true one-slot tensor takes
  the separate single-option branch and needs a separate export. Arbitrary token lengths and
  a production bucketing/dispatch layer are outside this change.
- Seven rows verify the packaging regression, not broad checkpoint accuracy or calibrated
  confidence parity. Casting an export copy to bf16 differs from retaining fp32 weights under
  autocast; the existing eager path itself remains unchanged.
- The script requires a CUDA-capable PyTorch build and local compiler toolchain to create the
  artifact. The `.pt2` embeds the model and CUDA kernels; no hosted service is involved.
