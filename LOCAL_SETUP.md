# Laya on this machine — local setup

Laya (the checkout in `laya/`) now runs here on real weights: all three checkpoints, real forward
passes, on CPU and on the AMD GPU through Metal (MPS). This file records what was installed, the
two code changes that were needed, how to re-run everything, and the numbers that were measured.

## Host

| | |
|---|---|
| machine | `MacPro7,1` (2019 Mac Pro), Intel Xeon, 56 logical cores, 256 GB RAM |
| OS | macOS (Darwin 25.6.0), `x86_64` |
| GPU | AMD Radeon Pro Vega II (32 GB) + Radeon PRO W6800X Duo, Metal 3 |
| accelerator APIs | **no CUDA** — CPU, or MPS via Metal |

## What was installed (all inside the repository; everything below is gitignored)

| path | contents |
|---|---|
| `.venv/` | Python 3.11.5 virtualenv — torch 2.2.2, transformers 4.57.6, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 0.36.2 |
| `models/` | the three checkpoints (~2.3 GB): `laya/` (English), plus `laya-multilingual` and `laya-typed-decisions` symlinked into the bundle |
| `.pip-cache/`, `.hf-cache/` | download caches, kept local so nothing is written outside the project |
| `setup_laya.sh` | idempotent setup: venv → pinned deps → editable install → checkpoints → verify |
| `laya_smoke_test.py` | end-to-end check on real weights: routing, all three checkpoints, all three primitives, presets, latency |
| `verify/numerics_check.py` | RoPE base actually used vs. trained, run-to-run determinism, SDPA vs. eager |
| `verify/bench_devices.py` | CPU vs. MPS latency, plus CPU thread scaling |

`laya` is installed editable, so `import laya` from anywhere uses this checkout and picks up edits.

## Code changes that were required

Both are real portability bugs, not local hacks; each is covered by `laya/tests/test_portability.py`
(13 checks) which is now wired into CI.

### 1. `laya/laya/common.py` — transformers 5 checkpoints on transformers 4

The checkpoints were saved by transformers 5, which records ModernBERT's RoPE bases as
`rope_parameters = {"full_attention": {...}, "sliding_attention": {...}}`. transformers 4.x does
not read that key, so it silently keeps its own defaults — global `160000`, local `10000`:

* `laya` and `laya-typed-decisions` want `160000 / 10000` → they happened to match.
* `laya-multilingual` (mmBERT) wants `160000 / 160000` → it ran with the **wrong RoPE base** on
  every sliding-attention layer, with no error. This is the "quietly wrong numbers" class of bug.

`build_model()` now maps `rope_parameters` onto the `global_rope_theta` / `local_rope_theta`
attributes that 4.x reads (and that the upstream `jhu-clsp/mmBERT-base` config still uses). On
transformers 5 the function is a no-op. Effect on the multilingual checkpoint, same input:

| | before | after |
|---|---|---|
| Hindi "charged twice, refund" → `billing` | 0.8647 | **0.9328** |
| Hindi → `refund_requested` | 0.9897 | 0.9909 |
| English / typed-decisions outputs | — | **bit-identical** (their configs already matched) |

### 2. `laya/laya/agent.py` — MPS crashed every call

`system_one` entered `torch.autocast(device_type=self.device.type, enabled=use_amp)` on every
call. Autocast is only ever *enabled* on CUDA, but torch validates the device type regardless, and
torch has no MPS autocast backend:

```
RuntimeError: User specified an unsupported autocast device_type 'mps'
```

Laya selects MPS automatically when it is available, so on this machine every `predict()` raised.
The forward pass now goes through `_amp_context(device, dtype)`, which returns
`torch.autocast(..., "cuda")` on CUDA and a `nullcontext()` otherwise.

## Verify it

```bash
cd /Users/threaded/projects/Laya/laya      # repository root
./setup_laya.sh                            # venv + deps + checkpoints + smoke test (idempotent)

# or individually
.venv/bin/python laya_smoke_test.py --models ./models            # device auto → MPS
.venv/bin/python laya_smoke_test.py --models ./models --device cpu
.venv/bin/python verify/numerics_check.py
.venv/bin/python verify/bench_devices.py
.venv/bin/python tests/test_local_e2e.py ./models                # the repo's own e2e suite
```

Results on this machine:

| check | result |
|---|---|
| `laya/tests/test_local_e2e.py` (real weights, CPU) | **23 passed, 0 failed** — multilingual billing 8/8, all 11 routing languages correct |
| `laya/tests/test_router.py` | 106 passed, 0 failed |
| `laya/tests/test_criteria.py` | 34 passed, 0 failed |
| `laya/tests/test_portability.py` | 13 passed, 0 failed |
| `laya_smoke_test.py` | all checks passed on **both** CPU and MPS |
| `verify/numerics_check.py` | RoPE bases match training; answers bit-identical run-to-run and SDPA vs. eager (`0.00e+00` on every reported value) |
| CI lint (`ruff`) + `compileall` | pass |

CPU and MPS produce identical answers (same presets, same confidences), so the GPU path is not a
numerical downgrade.

## Measured performance

One `predict()` call answering 4 questions on a short email, median of 10 after warm-up:

| checkpoint | CPU (28 threads) | MPS (Radeon Pro Vega II) |
|---|---|---|
| `laya` (421M) | 424 ms — 106 ms/question | **141 ms — 35 ms/question** |
| `laya-multilingual` (322M) | 195 ms — 49 ms/question | **117 ms — 29 ms/question** |
| `laya-typed-decisions` (421M) | 427 ms — 107 ms/question | **142 ms — 36 ms/question** |

The GPU is 1.7-3× faster than this 56-core CPU (3× on the two ModernBERT-large checkpoints, 1.7×
on the smaller multilingual one). The README's T4 reference is 32.8 ms/question for
`laya-multilingual`; MPS lands at 29 ms/question, i.e. T4-class latency on this hardware.

CPU thread scaling on `laya` (medians): 4 → 542 ms, 8 → 390 ms, **16 → 374 ms**, 28 → 433 ms,
56 → 927 ms. Around 16 threads is the sweet spot; the default 28 already over-subscribes, so
`OMP_NUM_THREADS=16` (or `torch.set_num_threads(16)`) is worth setting for CPU-only work.

Notes: the first MPS call pays ~13 s of Metal kernel compilation, so warm up before timing;
everything runs fp32 (bf16 autocast is CUDA-only in Laya).

## Using it from your own code

The checkpoints are already on disk, so point the router at the local roots and nothing touches
the network:

```python
import laya
from laya import Router

root = "/Users/threaded/projects/Laya/laya/models"
router = Router(models={"english":          root + "/laya",
                        "multilingual":     root + "/laya-multilingual",
                        "typed-decisions":  root + "/laya-typed-decisions"},
                preload=True)          # device=None → MPS here; pass device="cpu" to force CPU

res = router.predict({"body": "发票4411被重复扣款，请今天退款。"}, laya.triage_questions())
print(res["routing"]["model"], res["answers"]["intent"]["choice"])
```

`laya.load("<path>", device="mps")` works the same way for a single checkpoint. Leaving `device`
unset picks MPS, then CPU.

## Why these versions

* **torch 2.2.2** is the newest PyTorch with a macOS `x86_64` wheel. Both PyPI and
  `download.pytorch.org` stop at 2.2.2 (2.3.0 onward are Apple-Silicon-only), which is the whole
  constraint chain here.
* **transformers 4.57.6** is the last 4.x line. transformers 5.x requires torch ≥ 2.4, so it
  cannot be used with the only torch this CPU can install — hence the RoPE mapping above. 4.57.x
  is also the line that reads both ModernBERT and mmBERT checkpoints.
* **numpy < 2** because torch 2.2.2 was compiled against NumPy 1.x and fails to initialise under
  NumPy 2 (`Failed to initialize NumPy: _ARRAY_API not found`).
* **huggingface_hub < 1.0** is pinned by transformers 4.x.

If a newer torch is ever wanted, conda-forge still builds `osx-64` PyTorch (CPU/MKL) past 2.13,
which would allow transformers 5.x and remove the need for the RoPE mapping. That is an
alternative, not what is installed and verified here.

## Caveats

* No CUDA: fp16/bf16 mixed precision is unavailable, CPU and MPS both run fp32.
* MPS is a torch 2.2-era backend on Intel AMD hardware — it works and matches CPU numerics here,
  but it is not an NVIDIA-class path. `--device cpu` is the conservative choice.
* `models/` is ~2.3 GB and `.venv/` ~1 GB; both live in the repository and are gitignored, so
  they can be deleted and rebuilt with `./setup_laya.sh` without touching the working tree.
* No TensorFlow is installed on purpose (`USE_TF=0`): a TF install alongside torch can deadlock
  model construction, which the repo's tests already guard against.
