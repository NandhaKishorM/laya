# Inference backends

Laya runs one forward pass per decision, and `laya.load(..., backend=...)` picks how that
forward is executed. Everything else, from tokenization to temperature scaling, is shared, so
every backend answers the same question set the same way within rounding; `benchmarks/parity.py`
is the check. The README's [Inference backends](https://github.com/NandhaKishorM/laya#inference-backends)
section has the latency table and the user-facing contract; this page keeps the engineering
notes that would not fit there.

## The contract

`laya.backends.base.Backend` is the whole interface: `install()` swaps `model.forward` for the
backend's, `uninstall()` restores the class method, `forward(...)` takes the collated batch on the
agent's device and returns `(logits, act_logits)` exactly as the stock forward would, and
`warmup()` runs a few representative shapes at load. Two policies live next to it so every
backend sees the same inputs:

- **dtype**: `choose_dtype(device, cfg)` is applied once in `Agent.__init__` and the result is
  `agent.dtype`. A backend honours it or declines; it never picks another precision silently.
  The TileLang kernels in this tree are bf16, so an fp16 agent (compute capability < 8, or a
  checkpoint whose `amp_dtype` is fp16) gets the eager forward and one warning; `auto` routes
  it to `compile` instead. fp16 kernels are proposed in
  [#467](https://github.com/NandhaKishorM/laya/pull/467).
- **padding**: `bucket_shape(rows, tokens, max_len)` pads rows to a power of two and tokens to
  16-token buckets up to 256, 64-token buckets beyond, capped at the checkpoint's `max_len`.
  Both graph backends (`tilelang`, `compile`) pad to it, so each records a bounded set of CUDA
  graphs; `eager` pads nothing.

A backend that cannot run where it was asked raises `BackendUnavailable`; `Agent.set_backend`
turns that into one `RuntimeWarning` and the eager forward, or re-raises under `strict=True`.

## `compile`: cold start

`torch.compile(model.forward, dynamic=True, mode="reduce-overhead")`. Dynamic shapes mean one
compile serves every request length (static shapes recompile per length, ~18 s each); CUDA
graphs remove the per-kernel launch cost that dominates a short request. The cost is the first
compile, and two things cut it:

- **The inductor FX-graph cache** is persisted in a stable directory: `LAYA_INDUCTOR_CACHE_DIR`
  (default `~/.cache/laya/inductor`). torch's own default is under `/tmp`, which does not
  survive a reboot or a container restart; an explicit `TORCHINDUCTOR_CACHE_DIR` in the
  environment is still honoured. A second process reads the compiled graphs and Triton kernels
  from there instead of compiling them.
- **Warm-up at load** (`LAYA_COMPILE_WARMUP=0` to skip): (1 x 64 x 3), (4 x 128 x 5) and
  (8 x 256 x 6) rows x tokens x markers are run three times each at load, so the first request
  pays nothing.
  `agent.backend_object.warmup_s` is what it cost.

Two things found while measuring, both handled in `laya/backends/compile.py`:

- **Duck sizing.** Dynamo gives two dimensions that are equal at the first trace one symbol
  and then guards on them staying equal. With four questions of four options, rows == markers,
  and every later request where they differed recompiled (about 25 s each). The backend sets
  `torch.fx.experimental._config.use_duck_shape = False` before compiling and warms up on
  shapes whose rows, tokens and markers are pairwise different, so each dimension keeps its
  own symbol.
- **The attention mask is materialised.** Eager SDPA takes ModernBERT's `(rows, 1, L, L)` mask
  as a broadcast view. Under dynamic shapes inductor cannot prove the last dimension aligned,
  so it expands the mask to every head and pads it into a real buffer, `rows x 12 x L x L` in
  bf16: 0.8 GB at 32 rows x 1024 tokens. On a GPU with headroom that is bandwidth (tens of ms);
  on one that is nearly full, the caching allocator thrashes and the same call takes tens of
  seconds. `benchmarks/bench_compile.py` records that case as "GPU memory contention, not timed"
  rather than printing it as latency. The TileLang backend reads the packed QKV buffer with its
  own masking and has no such buffer, which is one reason `auto` prefers it.

Measured with `benchmarks/bench_compile.py`; the README table has the numbers.

**AOTInductor** (`bench_compile.py --aoti`, `torch._inductor.aoti_compile_and_package`) was
evaluated as the way to ship a precompiled artifact per checkpoint and GPU architecture and
skip the compile entirely. Where it stands on torch 2.11:

- `torch.export` of `DecisionModel` succeeds in about 5 s with dynamic rows, markers and
  tokens, once tokens are declared as a multiple of 16 (`16 * Dim(...)`): a plain range fails
  the exporter's own alignment guard (`L % 8`), the same mask-alignment padding described above.
- Packaging does not: exported under autocast, the program carries dtype asserts that AOTI
  trips outside autocast (`Tensor dtype mismatch! Expected: torch.bfloat16, Got: torch.float32`);
  exported from a copy of the model cast to bf16 without autocast, tracing fails inside the
  forward (`mat1 and mat2 must have the same dtype, but got Float and BFloat16`) because
  `DecisionModel.forward` upcasts the pooled state and the confidence features to fp32 before
  the action head, which autocast normally reconciles.

So the artifact route needs a dtype-explicit forward (cast the action head's input to the
head's dtype, or run the head in fp32), a small model-code change that is outside this branch.
Until then the cold-start story is the FX-graph cache plus warm-up, and `tilelang`, whose
kernels compile in about a second per shape bucket and are cached on disk. The trial's raw
outcomes are in `benchmarks/results/compile_multilingual_rtx4070.json` under `aoti`.

## TileLang portability: what was tried, what fails

tilelang 0.1.14 registers targets for CUDA, HIP, Metal, WebGPU and a C/CPU backend. Without AMD
or Apple hardware here, the question that could be answered is whether `laya/tl_kernels.py`
lowers for the CPU target at all, and which constructs are CUDA-only. Probed on this build
(`tilelang.compile(prim_func, target=...)`, Linux x86-64, no CUDA context touched):

| target | result |
|---|---|
| `"cpu"` | rejected up front: `Target cpu is not supported. Pass target options as a dict when the target needs attributes.` The string tilelang accepts for its CPU backend is `"c"`. |
| `"llvm"` | `Cannot find global function target.build.llvm`: the wheel ships no LLVM backend. |
| `"c"` | lowers to C source (`tl_templates/cpp/common.h`) and runs on CPU tensors, but only for a subset of the language, below. |

Every Laya kernel fails on `"c"`, each for one of three reasons:

| kernel | first failure on `target="c"` | construct |
|---|---|---|
| `gemm_kernel`, `gemm_geglu_kernel` | `CPU fill only supports local and global buffers, but got dst scope local.fragment` | `T.alloc_fragment` accumulator (`T.clear` on it) |
| `add_ln_kernel` | `CPU reduce only supports local src and local/local.var dst buffers, got src scope local.fragment` | `T.reduce_sum` / `T.reduce_max` over fragments |
| `rope_kernel` | `Cannot convert type bfloat16 to C type` | bf16 tensors (the C backend has no bf16 type) |
| `attn_kernel` | fails at `T.alloc_fragment((bm, bn), DT)` | fragments again; `make_swizzled_layout` and `GemmWarpPolicy.FullRow` were never reached |

What the C backend does accept, checked with minimal kernels: fp32 elementwise loops over
global buffers (`T.Parallel`, correct results on CPU tensors), `T.Pipelined` (lowered to a plain
loop), and `T.gemm` when the accumulator is `T.alloc_local` rather than a fragment (it lowers to
a scalar triple loop; a 64 x 64 fp32 product matched torch). `T.alloc_shared` plus fragments,
which every Laya kernel is written around because that is what feeds the tensor-core `T.gemm`
on CUDA, is what does not exist on the CPU target: `local.fragment` is a warp-level register
tile and the C backend has no notion of it.

So a CPU lowering of the existing kernels is not a matter of a target flag. It would be a second
set of kernels, fp32 or fp16, written with `alloc_local` accumulators and no fragment reductions,
and the `T.gemm` that results is a scalar loop with no blocking or vectorisation beyond what the
C compiler finds; it would not compete with PyTorch's MKL/oneDNN path that the eager backend
already uses on CPU. No CPU measurement against eager was therefore taken (item (b) of the
experiment does not apply). The same three constructs are the ones to check first on HIP and
Metal when that hardware is available: fragments and `T.gemm` with `GemmWarpPolicy` are the
CUDA/HIP tensor-core path, and bf16 is a per-target question.

The probes are small enough to re-run: `tilelang.compile(K.gemm_kernel(768, 768).prim_func, target="c")`
reproduces the first row of the table.
