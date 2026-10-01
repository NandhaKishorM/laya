# compile=True and the TileLang fast path: engineering notes

These notes cover how `compile=True` and `fast=True` behave beyond what the README says. They
come from measurements taken while working on #472, #576 and #718, on an RTX 4070 Ti SUPER with
torch 2.11 and tilelang 0.1.14. They are here so the next person does not have to measure them
again.

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
  batches and one for a single row, which torch specialises. `agent.warmup()` (#718) builds both
  before traffic arrives.
- **Across restarts.** Inductor's FX-graph cache keeps compiled graphs under
  `TORCHINDUCTOR_CACHE_DIR`. The default is under `/tmp`, which does not survive a reboot or a
  container restart. Set it to a persistent directory, or a volume in a container, and a second
  process loads the graphs instead of compiling them. In #472's measurement that took warm-up from
  about 120 s to about 50 s.

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
