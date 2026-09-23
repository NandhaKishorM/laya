# Laya benchmarks

Every checkpoint answered **byte-identical questions** in each run (fixed seed). Jev figures are **third-party published, never measured here** — no TypeSafe API access — so sample sizes and prompts differ; treat them as indicative.

| run | what | where |
|---|---|---|
| T4 Colab | typed-decisions, MASSIVE (14 langs), XNLI (15 langs), English suites, latency, option-order robustness, calibration repair | `research/results/t4_colab_benchmark.json` |
| CPU sweep | MASSIVE intent across **all 51 languages**; its typed-decisions part (`part_b`) covers the English checkpoint only | `research/results/cpu_51_language_sweep.json` |
| Applications | the seven workflow themes + the datasets where Jev numbers exist, all three checkpoints (laya 0.2.1, CPU, 400 cases per task, seed 13, 2026-09-19) | `research/results/app_benchmark_results.json` |


**Calibration columns in the CPU sweep predate the temperature clamp.** The 51-language ECE and mean-confidence figures were produced before #42 clamped temperatures to `[0.5, 5]`, so today's package reports different confidence for the affected buckets. Accuracy columns are unaffected, because a temperature-scaled softmax has the same argmax at every positive temperature.

The same 51 languages and 5,100 cases have now been re-run after the temperature clamp, in both regimes (`research/results/cpu_51_language_sweep_clamped.json`, [#208](https://github.com/NandhaKishorM/laya/issues/208)). Macro accuracy reproduces at **0.2269** exactly, and macro ECE moves **0.7331 → 0.5709**:

| | committed | re-run, raw temperatures | re-run, as served |
|---|---|---|---|
| macro accuracy | 0.2269 | 0.2269 | 0.2269 |
| macro ECE | 0.7331 | **0.7331** | 0.5709 |
| macro F1 | 0.2053 | 0.2053 | 0.2053 |
| mean confidence, `en` | 0.9989 | **0.9989** | 0.9582 |
| ECE, `en` | 0.1789 | **0.1789** | 0.1382 |

The raw-temperature column reproduces the committed file, so the only variable left is the clamp. `choice:11+` is the sole bucket it moves, and every case in this sweep is a 20-option question, so the clamp applies to all 5,100 — and lowers ECE in all 51 languages. `acc_at_50_coverage` is the one rank-quality column that uses the confidence values: macro 0.3004 → 0.3020, and `en` 0.94 → 0.98, so the flatter distribution selects a slightly better half rather than a worse one.

---

## Headline

| | Laya | Jev (published) |
|---|---|---|
| typed-decisions (2,000 decisions) | **0.766** | 0.727 |
| AG News (4 labels) | **0.953** | 0.910 |
| DAIR Emotion (6 labels) | **0.600** | 0.480 |
| ECE after temperature fitting | **0.081** | 0.246 |
| p50 latency, 1 question (T4) | **32.8 ms** | 236-276 ms |

---

## Languages

### All 51 MASSIVE languages — intent, 20 options (random = 0.050)

| | laya | laya-multilingual |
|---|---|---|
| macro accuracy | 0.2269 | **0.3661** |
| macro ECE *(lower better)* | 0.7331 | **0.3869** |
| languages clearing 3× random | 23 / 51 | **45 / 51** |

<details><summary><b>Per language (51)</b> — sorted by how much routing gains</summary>

| lang | laya | laya-multilingual | Δ | laya ECE | multilingual ECE |
|---|---|---|---|---|---|
| `th` | 0.080 | 0.480 | +0.400 | 0.881 | 0.336 |
| `ko` | 0.110 | 0.450 | +0.340 | 0.850 | 0.329 |
| `he` | 0.060 | 0.400 | +0.340 | 0.911 | 0.350 |
| `ur` | 0.070 | 0.400 | +0.330 | 0.883 | 0.311 |
| `hi` | 0.100 | 0.430 | +0.330 | 0.850 | 0.321 |
| `ar` | 0.110 | 0.400 | +0.290 | 0.800 | 0.341 |
| `pl` | 0.240 | 0.510 | +0.270 | 0.713 | 0.350 |
| `el` | 0.130 | 0.380 | +0.250 | 0.839 | 0.383 |
| `fa` | 0.140 | 0.390 | +0.250 | 0.820 | 0.399 |
| `ru` | 0.310 | 0.540 | +0.230 | 0.668 | 0.316 |
| `tr` | 0.140 | 0.370 | +0.230 | 0.788 | 0.417 |
| `lv` | 0.100 | 0.320 | +0.220 | 0.847 | 0.480 |
| `bn` | 0.080 | 0.290 | +0.210 | 0.865 | 0.408 |
| `nb` | 0.330 | 0.530 | +0.200 | 0.648 | 0.327 |
| `vi` | 0.060 | 0.260 | +0.200 | 0.891 | 0.521 |
| `az` | 0.100 | 0.300 | +0.200 | 0.825 | 0.368 |
| `hu` | 0.090 | 0.290 | +0.200 | 0.857 | 0.422 |
| `is` | 0.110 | 0.300 | +0.190 | 0.835 | 0.469 |
| `sv` | 0.380 | 0.570 | +0.190 | 0.596 | 0.276 |
| `km` | 0.000 | 0.180 | +0.180 | 0.952 | 0.412 |
| `ml` | 0.070 | 0.240 | +0.170 | 0.857 | 0.414 |
| `it` | 0.340 | 0.500 | +0.160 | 0.647 | 0.302 |
| `fi` | 0.130 | 0.290 | +0.160 | 0.849 | 0.436 |
| `ms` | 0.270 | 0.430 | +0.160 | 0.688 | 0.392 |
| `da` | 0.350 | 0.500 | +0.150 | 0.626 | 0.263 |
| `id` | 0.360 | 0.510 | +0.150 | 0.613 | 0.305 |
| `te` | 0.090 | 0.220 | +0.130 | 0.858 | 0.370 |
| `sl` | 0.200 | 0.330 | +0.130 | 0.756 | 0.433 |
| `jv` | 0.160 | 0.270 | +0.110 | 0.803 | 0.506 |
| `ta` | 0.120 | 0.230 | +0.110 | 0.822 | 0.397 |
| `ja` | 0.530 | 0.640 | +0.110 | 0.460 | 0.228 |
| `hy` | 0.050 | 0.150 | +0.100 | 0.835 | 0.506 |
| `zh-TW` | 0.460 | 0.540 | +0.080 | 0.520 | 0.327 |
| `de` | 0.420 | 0.500 | +0.080 | 0.558 | 0.301 |
| `tl` | 0.290 | 0.360 | +0.070 | 0.676 | 0.374 |
| `nl` | 0.390 | 0.450 | +0.060 | 0.591 | 0.378 |
| `af` | 0.290 | 0.350 | +0.060 | 0.687 | 0.484 |
| `my` | 0.060 | 0.120 | +0.060 | 0.861 | 0.455 |
| `sq` | 0.210 | 0.260 | +0.050 | 0.755 | 0.476 |
| `sw` | 0.130 | 0.180 | +0.050 | 0.828 | 0.549 |
| `cy` | 0.120 | 0.160 | +0.040 | 0.841 | 0.591 |
| `kn` | 0.110 | 0.150 | +0.040 | 0.842 | 0.437 |
| `es` | 0.510 | 0.530 | +0.020 | 0.480 | 0.275 |
| `ka` | 0.090 | 0.110 | +0.020 | 0.845 | 0.528 |
| `ro` | 0.330 | 0.350 | +0.020 | 0.658 | 0.404 |
| `zh-CN` | 0.620 | 0.630 | +0.010 | 0.376 | 0.212 |
| `am` | 0.120 | 0.110 | -0.010 | 0.825 | 0.463 |
| `pt` | 0.470 | 0.450 | -0.020 | 0.512 | 0.342 |
| `mn` | 0.130 | 0.100 | -0.030 | 0.837 | 0.558 |
| `fr` | 0.590 | 0.540 | -0.050 | 0.388 | 0.277 |
| `en` | 0.820 | 0.680 | -0.140 | 0.179 | 0.209 |

</details>

### English vs the rest

| task | | laya | laya-multilingual |
|---|---|---|---|
| MASSIVE intent — English | **0.783** | 0.657 |
| MASSIVE intent — other languages | 0.306 | **0.451** |
| MASSIVE scenario — English | **0.603** | 0.560 |
| MASSIVE scenario — other languages | 0.281 | **0.439** |
| XNLI — English | **0.860** | 0.843 |
| XNLI — other languages | 0.521 | **0.731** |

The English checkpoint does not degrade gracefully outside English — it collapses, and stays confident doing so. Khmer: **0.000 accuracy at 0.952 confidence**. Its mean confidence never drops below 0.885 at any accuracy level, so confidence gating cannot catch it — which is why routing happens *before* the forward pass.

---

## Themes — the application workflows

Each is real labelled data, 400 cases, all three checkpoints. *held out* means the source was **not** in Laya's training mix.

| theme | laya | laya-multilingual | laya-typed-decisions | data |
|---|---|---|---|---|
| Email spam | **0.993** | 0.993 | 0.958 | in training |
| Phishing | 0.980 | **0.993** | 0.940 | in training |
| LLM guardrails (jailbreak) | 0.708 | 0.755 | **0.762** | **held out** |
| Moderation (toxicity) | **0.530** | 0.525 | 0.530 | **held out** |
| RAG passage relevance | 0.625 | **0.657** | 0.625 | in training |
| Support triage (10-way queue) | 0.502 | **0.522** | 0.505 | in training |
| Model routing (domain) | 0.639 | 0.123 | **0.659** | held out |

**Where it is strong:** email spam 0.993 and phishing 0.993, both with ECE around 0.01 — production-grade, though both were in the training mix.

**Where it is weak:** moderation on held-out toxic-chat is 0.530 with macro-F1 0.400 — barely above chance on a balanced split. The demo Space has a Moderation tab; hand-picked examples work, real traffic does not. Guardrails at 0.708–0.762 is the honest jailbreak-detection number, consistent across two unrelated datasets (deepset prompt-injections measured 0.698 separately).

### On the public datasets where Jev numbers exist

| dataset | laya | laya-multilingual | laya-typed-decisions | Jev (published) |
|---|---|---|---|---|
| AG News (4 labels) | 0.950 | 0.930 | **0.953** | 0.910 |
| DAIR Emotion (6 labels) | 0.595 | 0.530 | **0.600** | 0.480 |
| banking77 (77 labels) | 0.425 | 0.425 | **0.492** | 0.870 |

banking77 is the one clear loss, and it is architectural: a choice question's options share a fixed `head_max_len` budget, so 77 labels get roughly 4 tokens each and stop being distinguishable. Both checkpoints score **exactly 0.425**, which is what you would expect from a budget ceiling rather than a capability gap. Keep choice questions under ~20 options.

---

## typed-decisions — 400 cases, 2,000 decisions

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` | **0.766** | 0.471 | 0.061 | 0.213 | 0.242 |
| `laya` | 0.361 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher ceiling* | *0.735* | *—* | *—* | *—* | *—* |
| *majority class* | *0.461* | *—* | *—* | *—* | *—* |
| *random guess* | *0.318* | *—* | *—* | *—* | *—* |

| workflow | laya-typed-decisions |
|---|---|
| agent trace observability | 0.730 |
| customer service | 0.764 |
| invoice processing | 0.804 |
| security incidents | 0.766 |

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461). All of the capability on this benchmark comes from fine-tuning.

---

## Speed (Tesla T4)

| questions per call | laya | laya-multilingual |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms | **72.3 ms** |
| 50 | 771.3 ms | **337.4 ms** |

103–332 questions/sec batched. Jev independently measured at 236-276 ms p50, so Laya answers one question roughly **6–7× faster**.

### Calibration

| | as shipped | temperature refit | 
|---|---|---|
| `laya` | 0.466 | **0.081** |
| `laya-multilingual` | 0.314 | **0.106** |

Both ship over-confident; `laya-multilingual` ships with no fitted temperatures at all. Refitting one temperature per (question type, option count) on held-out data is the single highest-value fix available, and takes ECE below Jev's measured 0.246.

### Option-order robustness

How often the answer changes when the options are permuted. Jev measured at 0.13.

| suite | laya | laya-multilingual |
|---|---|---|
| massive_intent.en | 0.150 | 0.230 |
| en.emotion | 0.040 | 0.090 |
| xnli.en | 0.000 | 0.015 |

At 20 options both are less order-stable than Jev — worth fixing with more aggressive option-order shuffling during training.


## Other hardware: GB10 and a laptop CPU

Contributed measurements from a router deployment (laya 0.3.5). They were taken through a small HTTP server wrapping `Agent.system_one`, not in-process, so every figure includes one HTTP round trip.

### NVIDIA GB10 (DGX Spark, aarch64), CUDA

`typed-decisions` checkpoint (1024 ctx), default dtype, torch 2.14.0+cu130. The GPU was shared with a resident 73 GB SGLang server and a whisper server. Each question is a 3-option `choice`, with 40 calls per row after warm-up. Loopback round trip to `/health` was 0.6 ms, so network is not in these numbers.

| questions per call | p50 | p95 |
|---|---|---|
| 1 | 100.2 ms | 169.3 ms |
| 5 | 137.7 ms | 162.4 ms |
| 10 | 159.3 ms | 243.0 ms |
| 50 | 443.1 ms | 464.6 ms |

Each extra question costs about **7.0 ms**, half the T4's ~14.9 ms. But one question is **slower** than the T4's 39.5 ms, because roughly 93 ms per call is fixed overhead that the GPU does not remove. We have not isolated where that overhead goes. On a GB10, batching questions into one call is where the speedup is.

On laya_router's 180 labelled requests (one tier question), accuracy on CUDA matched CPU to within one row per wording (0.700 vs 0.694, 0.656 vs 0.656, 0.611 vs 0.606). That is backend floating-point noise, not a change in behaviour.

Setup note for aarch64 without root: Triton JIT-compiles a CUDA shim with `gcc` on the first CUDA call, which fails with `Python.h: No such file or directory` if `python3-dev` is absent. Fetch the headers with `apt-get download libpython3.12-dev python3.12-dev`, unpack with `dpkg-deb -x` into a directory, and set `CPATH` to both `usr/include` and `usr/include/python3.12` under it.

### Laptop CPU (Ryzen 9 6900HX, avx2 only, WSL2)

**Pin inter-op threads to 1.** `system_one` runs one forward pass per call, so there is nothing for inter-op parallelism to overlap. On a three-question call over HTTP, on a busy host, torch's defaults (10 intra-op, 5 inter-op on 10 vCPUs) gave p50 **9,396 ms**. `torch.set_num_threads(8)` plus `torch.set_num_interop_threads(1)` brought it to **783 ms**, 12x faster with no code change.

With inter-op pinned, one question in-process on a quieter host:

| intra-op threads | p50 | p95 |
|---|---|---|
| 1 | 910 ms | 1,023 ms |
| 4 | 374 ms | 552 ms |
| 8 | **329 ms** | **378 ms** |
| 10 (every vCPU) | 388 ms | 708 ms |

The best setting is the physical core count plus a little, not one thread per vCPU. SMT siblings contend.

### Calibration on a routing task runs the other way

On laya_router's 180 requests (zero-shot, one 3-tier `choice`), nearly every configuration we measured was **under**-confident (the few exceptions were +0.01 to +0.06, and among the least accurate). Mean P(chosen) (the chosen option's probability, not the entropy-based `confidence` field) sat below accuracy, by −0.18 on the root checkpoint with example-led tier descriptions (0.562 vs 0.744) and by −0.19 on `typed-decisions` (0.501 vs 0.694). This is one task and one set of labels, so it does not contradict the over-confidence reported above. It does mean the direction of the miscalibration depends on the task, and a temperature fit on your own data is the right fix either way.

---

## Limits, stated plainly

- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400).
- **Keep `choice` questions under ~20 options.**
- **Both checkpoints ship over-confident.** Fit temperatures on your own data.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route.

---

## GPU fast path

`pip install laya[fast]` + `laya.load(..., fast=True)` replaces the encoder/head forward with fused
[TileLang](https://github.com/tile-ai/tilelang) kernels (GEMM+epilogue, GEMM+GEGLU, residual+LayerNorm,
in-place RoPE, sliding-window flash attention over the packed QKV buffer), bf16-resident weights and one
CUDA graph per (batch, length) bucket. Measured with `benchmarks/bench_fast.py --eval 1000` on an
RTX 4070 Ti SUPER, torch 2.11 + CUDA 13, tilelang 0.1.14; raw numbers in `benchmarks/results/`.

### Same answers

`benchmarks/parity_fast.py` answers a fixed, deterministic set of 60 states x up to 8 questions (the five presets over
12 texts in six languages, short and long) with the stock bf16-autocast forward, the fast path, and an fp32 forward as
the reference; every per-option probability from all three is in `benchmarks/results/parity_*.json`, so the comparison
can be re-checked without a GPU.

| checkpoint | type | n | max \|p_fast - p_stock\| | max \|p_fast - p_fp32\| | max \|p_stock - p_fp32\| | argmax fast = stock | fast = fp32 |
|---|---|---|---|---|---|---|---|
| laya | choice | 48 | 0.031 | **0.022** | 0.024 | 47/48 | 47/48 |
| laya | noul | 180 | 0.076 | **0.044** | 0.058 | 180/180 | 180/180 |
| laya | score | 60 | 0.015 | **0.011** | 0.017 | 59/60 | 60/60 |
| laya-multilingual | choice | 48 | 0.049 | **0.015** | 0.039 | 47/48 | 47/48 |
| laya-multilingual | noul | 180 | 0.037 | 0.046 | 0.045 | 180/180 | 179/180 |
| laya-multilingual | score | 60 | 0.010 | **0.009** | 0.009 | 59/60 | 59/60 |

The fast path is at least as close to the fp32 reference as the stock bf16 path is (the residual stream stays in fp32 in
both), and the two bf16 paths differ from each other only by bf16 accumulation order; the few argmax disagreements are
near-tie options, and on every one of them the fast path agrees with fp32. Dataset accuracy / ECE (AG News, dair-ai
emotion, 1,000 samples each) are identical within noise; see `benchmarks/bench_fast.py --eval 1000`.

### Latency, `agent.predict()` end to end (ms, incl. tokenization)

| checkpoint | case | stock | fast | speedup |
|---|---|---|---|---|
| laya (ModernBERT-large) | 1 question, 72 tok | 17.7 | 4.6 | **3.8×** |
| | 3 questions, 72 tok | 18.9 | 6.6 | 2.9× |
| | 30 questions, 72 tok | 43.2 | 35.7 | 1.2× |
| | 30 questions, 512 tok | 327.5 | 232.1 | 1.4× |
| laya-multilingual (mmBERT-base) | 1 question, 72 tok | 14.1 | 2.8 | **5.1×** |
| | 3 questions, 72 tok | 15.0 | 3.9 | 3.9× |
| | 30 questions, 72 tok | 22.2 | 17.8 | 1.2× |
| | 30 questions, 966 tok | 320.7 | 187.6 | 1.7× |
| laya-multilingual, AG News eval loop | 1 question / sample | 14.9 | 3.2 | 4.7× |

Small requests are launch-overhead bound in the stock path (≈200 kernels from Python per call); the CUDA
graph removes that. Large batches are GEMM bound; the fused kernels sit at ~80 TFLOPS there, on par with
cuBLAS, so the gain comes from the fused epilogues and the sliding-window attention (16× faster than SDPA
with a dense mask at L=1024). First use of a new length bucket compiles kernels (a few seconds, cached on
disk); inputs ≤256 tokens share one dynamic-shape kernel and never recompile.
