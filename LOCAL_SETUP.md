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
| `verify/edge_sweep.py` | inference paths the other suites miss: 12 questions in one pass, 20/77/120-option choice, odd input shapes, truncation, router lifecycle, CPU-vs-MPS agreement |
| `verify/checkpoints.py` + `verify/checkpoints.json` | checks the weights against recorded sha256 hashes (a truncated download is otherwise silently wrong) |
| `verify/soak_check.py` | 200 repeated calls for drift and RSS growth, reload/eviction cycles, concurrent calls from several threads. Uses `psutil` if installed and falls back to peak RSS otherwise |
| `examples/` | 41 worked examples as an eight-stage learning path, plus `run_all.sh` and the path doc |

`laya` is installed editable, so `import laya` from anywhere uses this checkout and picks up edits.

## Code changes that were required

Both are real portability bugs, not local hacks; each is covered by `laya/tests/test_portability.py`
(18 checks, including the CUDA-OOM fallback) which is wired into CI.

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
.venv/bin/python verify/edge_sweep.py
.venv/bin/python verify/checkpoints.py                           # weights vs recorded hashes
.venv/bin/python verify/soak_check.py                            # drift, RSS, concurrency
.venv/bin/python tests/test_local_e2e.py ./models                 # the repo's own e2e suite
./examples/run_all.sh                                            # every example, one log per run
```

Results on this machine:

| check | result |
|---|---|
| `laya/tests/test_local_e2e.py` (real weights, CPU) | **23 passed, 0 failed** — multilingual billing 8/8, all 11 routing languages correct |
| `laya/tests/test_router.py` | 106 passed, 0 failed |
| `laya/tests/test_criteria.py` | 34 passed, 0 failed |
| `laya/tests/test_portability.py` | 18 passed, 0 failed (includes the CUDA-OOM fallback) |
| `laya_smoke_test.py` | all checks passed on **both** CPU and MPS |
| `verify/numerics_check.py` | RoPE bases match training; answers bit-identical run-to-run and SDPA vs. eager (`0.00e+00` on every reported value) |
| `verify/edge_sweep.py` | all edge-case checks pass (25 here; the MPS checks skip where MPS is unavailable) |
| `verify/checkpoints.py` | 3/3 checkpoints match the recorded sha256; a deliberately corrupted copy is caught |
| `verify/soak_check.py` | 200/200 calls byte-identical with flat RSS; 6 reload cycles leave exactly 1 live agent and 1 live model (0 after `unload()`); 4-thread concurrency matches the single-threaded answer |
| `laya/tests/test_training.py` | 59 passed — proper-scoring-rule reward, TD(λ) targets, ECE, entropy confidence, collation, sequence building |
| `examples/run_all.sh` | **41 passed, 0 failed** (eight stages, 23-49 s each; the whole sweep is ~20 min) |
| CI lint (`ruff`) + `compileall` | pass |

CPU and MPS produce identical answers (same presets, same confidences), so the GPU path is not a
numerical downgrade.

## Measured performance

One `predict()` call answering 4 questions on a short email, median of 10 after warm-up:

| checkpoint | CPU (28 threads) | MPS (Radeon Pro Vega II) |
|---|---|---|
| `laya` (421M) | 453 ms — 113 ms/question | **146 ms — 37 ms/question** |
| `laya-multilingual` (322M) | 192 ms — 48 ms/question | **121 ms — 30 ms/question** |
| `laya-typed-decisions` (421M) | 441 ms — 110 ms/question | **145 ms — 36 ms/question** |

Re-measured on a quiet machine for the final verification pass; individual runs move by a few
percent with machine load (an earlier pass gave 424/195/427 ms on CPU and 141/117/142 ms on MPS).

The GPU is 1.6-3× faster than this 56-core CPU (3× on the two ModernBERT-large checkpoints, 1.6×
on the smaller multilingual one). The README's T4 reference is 32.8 ms/question for
`laya-multilingual`; MPS lands at 29 ms/question, i.e. T4-class latency on this hardware.

CPU thread scaling on `laya` (medians): 4 → 501 ms, 8 → 385 ms, **16 → 366 ms**, 28 → 415 ms,
56 → 1266 ms. Around 16 threads is the sweet spot; the default 28 already over-subscribes and 56
threads is markedly worse, so `OMP_NUM_THREADS=16` (or `torch.set_num_threads(16)`) is worth
setting for CPU-only work.

Notes: the first MPS call pays ~13 s of Metal kernel compilation, so warm up before timing;
everything runs fp32 (bf16 autocast is CUDA-only in Laya).

## Independent accuracy check against public data

Setting the machine up only proved the plumbing worked. The published tables were re-checked by
running the repo's own harnesses against public datasets — this is measured here, CPU only, not
copied from `BENCHMARKS.md`.

### MASSIVE intent, 20 options — `research/scripts/bench_local.py --langs 10 --per-lang 60 --skip-b`

| | english | multilingual | published, 51 languages (english / multilingual) |
|---|---|---|---|
| macro accuracy | 0.2500 | **0.3950** | 0.2269 / 0.3661 |
| macro ECE *(lower better)* | 0.7100 | **0.4008** | 0.7331 / 0.3869 |
| languages > 3x random | 4 / 10 | **8 / 10** | 23 / 51 / 45 / 51 |

Per language, `en` scores **0.783** — the published English figure exactly — and the English
checkpoint then collapses off English *while staying confident*: Amharic 0.100 at 0.949 mean
confidence, Arabic 0.133 at 0.897, Bengali 0.117 at 0.953, Greek 0.150 at 0.970. The multilingual
checkpoint holds those up (Arabic 0.450, Bengali 0.450, Greek 0.417, German 0.467), which is the
whole reason `Router` exists.

### Applications — `research/scripts/bench_apps.py` with `BENCH_N=80`

| suite | english | multilingual | typed-decisions | README (routed) |
|---|---|---|---|---|
| AG News, 4 labels | 0.963 | **0.975** | 0.963 | 0.950 |
| DAIR Emotion, 6 labels | 0.637 | 0.600 | **0.662** | 0.595 |
| Banking77, 77 labels at once | 0.412 | 0.500 | 0.425 | 0.425 |
| phishing email | 0.975 | **0.988** | 0.925 | — |
| email spam | **0.988** | 0.963 | 0.925 | — |
| guardrails (jailbreak) | 0.838 | **0.875** | 0.863 | 0.698 (injections) |
| moderation (toxicity) | 0.500 | 0.487 | 0.500 | — |
| support triage | 0.550 | 0.550 | 0.537 | — |

Banking77 lands on **0.425**, the number the README quotes for the 77-option case, and the
AG News / emotion figures agree with the published routed results within sampling noise (80 cases
per suite here). The README's honest limits reproduce too: `score` is the weakest primitive and
50+ options in one question is where Laya trails Jev.

Both tables were re-run end to end a second time (same seed, same cached datasets) and
reproduced **identically — every accuracy, F1 and ECE digit, and both macro aggregates**. The
runs are deterministic; only the ms/case figures move with machine load.

### Two fixes the harnesses needed to run here

* they hardcoded the model root to `~/laya_models` — now `LAYA_MODELS` (default unchanged), so
  `LAYA_MODELS=$PWD/models` points them at the checkpoints this repo already has;
* their `sys.path` insert pointed at a non-existent `research/laya`; it now points at the
  repository root.

Not verified: the fine-tuning notebook. It trains on 2xT4 and there is no CUDA here, so it was
read but never executed.

### What the soak check found

Repeated checkpoint loads push RSS up and it never comes back down — after 6 reload cycles the
process sat at 3.4 GB against 2.3 GB for a single resident checkpoint, and `unload()` did not
return it. That is macOS `malloc` keeping freed pages, not a leak: counting live objects shows
**exactly 1 `Agent` and 1 `DecisionModel` while cycling and 0 after `unload()`**, which is what
the check now asserts.

The same exercise threw up an API subtlety worth knowing: `Router.load()` returns early for a
checkpoint that is already resident, so it does not run eviction. `max_loaded` is enforced when a
checkpoint is *loaded*, not when one is touched — lower it and then load something new, or call
`unload()`, if residency has to drop immediately. That is now stated in `Router.load`'s docstring.

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
