# compile=True and the TileLang fast path: engineering notes

These notes cover how `compile=True` and `fast=True` behave beyond what the README says. They
come from measurements taken while working on #472, #576 and #718, on an RTX 4070 Ti SUPER with
torch 2.11 and tilelang 0.1.14. They are here so the next person does not have to measure them
again.

## Backend selection

`Agent(..., backend="auto")` and `laya.load(..., backend="auto")` opt into the backend
class layer. The default remains eager. Explicit `backend=` takes precedence over `compile`
and `fast`; omitting it preserves both flags' existing behaviour.

- `eager`: the stock PyTorch forward, on any supported device.
- `compile`: CUDA-only `torch.compile` with dynamic shapes and `reduce-overhead` mode,
  bucket padding, persistent inductor cache, and warmup at installation. It reuses the same
  independent-dimension scope as `compile=True`, which keeps its existing default mode and
  CPU support. Set `LAYA_COMPILE_WARMUP=0` to defer backend warmup and `LAYA_INDUCTOR_CACHE_DIR`
  to choose its cache directory (default `~/.cache/laya/inductor`).
- `tilelang`: an adapter around the current fast path, using the agent's bf16 or fp16 dtype.
- `auto`: TileLang on CUDA with a supported ModernBERT encoder and dtype when TileLang is
  installed, otherwise compile on CUDA; eager on other devices.
- `onnx`: `laya.load(..., backend="onnx", onnx_path="model.onnx")` returns the existing
  `ONNXAgent`. Without `onnx_path`, it uses `laya.onnx`.

An unavailable backend emits a `RuntimeWarning` naming the resolved backend and falls back to
eager. To require a backend, use `agent.set_backend("tilelang", strict=True)`. Switching waits
for active inference; `agent.backend` reports the active name and `agent.backend_object`
exposes the installed object. `agent.set_backend("compile", warmup=False)` defers compilation
until inference, so compilation errors then surface on the request. `agent.warmup()` remains
available. `agent.deaccelerate()` removes a backend installed through the class layer.

Routers forward an explicit selection through `Router(agent_kwargs={"backend": "auto"})`.
They pass no backend argument by default, preserving compatibility with existing Agent-like
constructors. Scoped CPU OOM retries detach the backend and restore it when the model returns
to its original device.

## `compile=True` materialises the attention mask

Eager SDPA takes ModernBERT's `(rows, 1, L, L)` attention mask as a broadcast view. Under the
dynamic shapes that `compile=True` uses, inductor cannot prove the last dimension is aligned. It
expands the mask to every head and pads it into a real buffer of `rows x heads x L x L`. In bf16
with 12 heads, that is **0.8 GB at 32 rows x 1024 tokens**.

- **GPU with headroom.** The buffer costs bandwidth, tens of ms per long batch.
- **GPU nearly full.** The caching allocator thrashes, and the same call can take tens of seconds.

If you compile with long batches on a busy GPU, cap the batch size (`predict_batch(...,
batch_size=)`) or use `fast=True`. The TileLang attention reads the packed QKV buffer and masks
by sequence length, so it has no such buffer.

## Cold start

- **First compile.** It takes tens of seconds per graph. `compile=True` needs two graphs: one for
  batches and one for a single row, which torch specialises. `compile=True` now calls `agent.warmup()`
  during load. `compile_warmup=False` restores lazy compilation, and `agent.warmup(shapes=...)`
  remains available manually. Eager and TileLang loads do not warm automatically. These shapes
  cover common requests, not every possible shape guard.
- **Warm-up failure.** Automatic warm-up is best effort: a failure emits a `RuntimeWarning`
  naming the error (including the underlying compiler error) and load returns with the
  `torch.compile` wrapper and compile settings intact. For example, Windows without MSVC can
  load with `compile=True` even though warm-up fails. Later requests still use the compiled
  model and surface compilation failures; Laya does not switch them to eager execution.
  Explicit `agent.warmup()` calls also propagate failures, including after a failed automatic
  warm-up. A successful load therefore does not guarantee that compiled inference is ready.
- **Across restarts.** Inductor's FX-graph cache keeps compiled graphs under
  `TORCHINDUCTOR_CACHE_DIR`. The default is under `/tmp`, which does not survive a reboot or a
  container restart. Set it to a persistent directory, or a volume in a container, and a second
  process loads the graphs instead of compiling them. In #472's measurement that took warm-up from
  about 120 s to about 50 s.
- **Laya cache opt-in.** `laya.load(..., compile=True, compile_cache=True)` sets the process-wide
  `TORCHINDUCTOR_CACHE_DIR` only when absent, to `$XDG_CACHE_HOME/laya/torchinductor` or
  `~/.cache/laya/torchinductor` when XDG is unset or not absolute. An existing setting, including
  one set by an earlier PyTorch compile, wins. The directory is created at load; filesystem
  errors propagate. `compile_cache=False` (default), eager, and TileLang loads leave the
  environment alone. This does not move or delete old caches. Containers still need a persistent
  home/volume. Cache compatibility and invalidation are managed by PyTorch; a GPU, torch,
  compiler, model, or input guard change can require compilation again.

## Opt-in CUDA graphs

```python
agent = laya.load("convaiinnovations/laya", compile=True,
                  compile_cache=True, compile_mode="reduce-overhead")
```

`compile_mode` defaults to `"default"`; only `"default"` and `"reduce-overhead"` are accepted
on the active compiled path. Eager and TileLang loads ignore the compile options. CPU compilation
still works, but CUDA graph recording only applies on CUDA. The CUDA mode requires PyTorch's
`torch.compiler.cudagraph_mark_step_begin` API; older builds without it raise an explicit error.

Dynamic Dynamo graphs do not imply shape-independent CUDA graphs: new concrete shapes may
require warm-up and recording again, without a new Dynamo graph. The two default synthetic
warm-up shapes do not pre-record every request shape. Repeated shapes can benefit, but varying
shapes can pay extra latency and retain graph pools. PyTorch may skip CUDA graphs for unsupported
operations or configurations; setting this mode is not a guarantee of capture.

Laya marks each compiled CUDA forward as a new step, serializes these forwards across its agents,
and clones both output tensors outside the compiled graph before releasing the lock. This keeps
retained outputs valid across replays, at the cost of two copies and serialized forward execution.
The lock does not coordinate unrelated application-owned compiled models; callers sharing CUDA
graph iterations or using custom streams must manage their own coordination. Disk caches reuse
compiled code, not live CUDA graph recordings or their device memory, across processes.

Reproduce cold/restart timings, memory, and cache counters with
`benchmarks/bench_compile_defaults.py`; see
[the recorded measurements](https://github.com/NandhaKishorM/laya/blob/main/benchmarks/results/compile-defaults/README.md).

## AOTInductor: not yet

Shipping a precompiled artifact per checkpoint and GPU architecture
(`torch._inductor.aoti_compile_and_package`) would remove the compile entirely. On torch 2.11 it
stops at packaging:

- **Export works.** `torch.export` of `DecisionModel` succeeds, in about 5 s, with dynamic rows,
  markers and tokens. Tokens must be declared as a multiple of 16 (`16 * Dim(...)`); a plain range
  fails the exporter's own `L % 8` alignment guard. This is the same mask alignment as above.
- **Packaging fails.** How it fails depends on how the program was exported:
  - **Under autocast**, the program carries dtype asserts that AOTI trips outside autocast:
    `Tensor dtype mismatch! Expected: torch.bfloat16, Got: torch.float32`.
  - **From a bf16 copy without autocast**, tracing fails inside the forward:
    `mat1 and mat2 must have the same dtype`. `DecisionModel.forward` upcasts the pooled state
    and the confidence features to fp32 before the action head, and autocast normally
    reconciles that.

The artifact route therefore needs a dtype-explicit action head: either cast its input to the
head's dtype, or run the head in fp32.

## TileLang portability: the kernels are CUDA-only

tilelang registers targets for CUDA, HIP, Metal, WebGPU and a C backend. Without AMD or Apple
hardware, the answerable question was whether `laya/tl_kernels.py` lowers for the CPU at all.
Probed with `tilelang.compile(kernel.prim_func, target=...)` on Linux x86-64:

| target | result |
|---|---|
| `"cpu"` | Rejected up front: `Target cpu is not supported`. tilelang's CPU backend is `"c"`. |
| `"llvm"` | `Cannot find global function target.build.llvm`. The wheel ships no LLVM backend. |
| `"c"` | Lowers to C and runs on CPU tensors, but only for a subset of the language. |

Every Laya kernel fails on `"c"`, for one of three reasons:

| kernel | failure on `target="c"` | construct |
|---|---|---|
| `gemm_kernel`, `gemm_geglu_kernel` | `CPU fill only supports local and global buffers, but got dst scope local.fragment` | `T.alloc_fragment` accumulator |
| `add_ln_kernel` | `CPU reduce only supports local src and local/local.var dst buffers` | `T.reduce_sum` / `T.reduce_max` over fragments |
| `rope_kernel` | `Cannot convert type bfloat16 to C type` | bf16 tensors |
| `attn_kernel` | fails at `T.alloc_fragment` | fragments |

The C backend does accept:

- fp32 elementwise loops (`T.Parallel`);
- `T.Pipelined`, which lowers to a plain loop;
- `T.gemm` with a `T.alloc_local` accumulator, which lowers to a scalar triple loop.

A CPU version would therefore be a second set of kernels, not a target flag. Its GEMM would be an
unblocked scalar loop, and it would not compete with the MKL/oneDNN path that the stock forward
already uses on CPU. The same three constructs are the ones to check first on HIP and Metal:
fragments, `T.gemm` with `GemmWarpPolicy`, and bf16/fp16 support.

To reproduce the first row of the second table:
`tilelang.compile(K.gemm_kernel(768, 768).prim_func, target="c")`.
