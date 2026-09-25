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

**The multilingual columns below are the re-run, not the committed file.** The committed `cpu_51_language_sweep.json` was measured at laya 0.2.0, and its `laya-multilingual` half does not reproduce on current code — 51 languages and 5,100 cases re-run under laya 0.2.0, 0.3.7 and 0.3.20, on x86 CPU and on an Apple M5 under both CPU and MPS, all give **0.4008** macro accuracy and match the committed file on only **6 of 51** languages, while the `laya` half reproduces exactly on all of them. Three machines agreeing settled it ([#208](https://github.com/NandhaKishorM/laya/issues/208)). The table therefore prints the refreshed numbers for both checkpoints, from `research/results/cpu_51_language_sweep_refreshed.json`, which records the environment and keeps the superseded committed columns beside the new ones:

| | committed (`laya-multilingual`) | refreshed |
|---|---|---|
| macro accuracy | 0.3661 | **0.4008** |
| macro ECE | 0.3869 | **0.3911** |
| languages clearing 3× random | 45 / 51 | **48 / 51** |

The direction is consistent rather than noise: of the 16 languages that move by 0.05 or more, **every one moves up** (`bn` 0.29 → 0.45, `kn` 0.15 → 0.30, `hy` 0.15 → 0.25), and none move down by that much. `laya`'s columns are unchanged by the refresh — they reproduce the committed file to the last stored digit — so the accuracy spread between the two checkpoints is wider than the committed table suggested, and `en` remains the one language where `laya` wins (0.820 against 0.710).

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
| macro accuracy | 0.2269 | **0.4008** |
| macro ECE *(lower better)* | 0.5709 | **0.3911** |
| languages clearing 3x random | 23 / 51 | **48 / 51** |

<details><summary><b>Per language (51)</b> — sorted by how much routing gains</summary>

| lang | laya | laya-multilingual | Δ | laya ECE | multilingual ECE |
|---|---|---|---|---|---|
| `th` | 0.080 | 0.480 | +0.400 | 0.718 | 0.356 |
| `bn` | 0.080 | 0.450 | +0.370 | 0.627 | 0.354 |
| `fa` | 0.140 | 0.510 | +0.370 | 0.633 | 0.298 |
| `hi` | 0.100 | 0.460 | +0.360 | 0.642 | 0.368 |
| `ko` | 0.110 | 0.470 | +0.360 | 0.680 | 0.311 |
| `ar` | 0.110 | 0.460 | +0.350 | 0.580 | 0.315 |
| `ur` | 0.070 | 0.420 | +0.350 | 0.643 | 0.336 |
| `el` | 0.130 | 0.440 | +0.310 | 0.679 | 0.430 |
| `he` | 0.060 | 0.370 | +0.310 | 0.738 | 0.419 |
| `vi` | 0.060 | 0.340 | +0.280 | 0.690 | 0.464 |
| `hu` | 0.090 | 0.360 | +0.270 | 0.678 | 0.401 |
| `az` | 0.100 | 0.360 | +0.260 | 0.598 | 0.381 |
| `pl` | 0.240 | 0.500 | +0.260 | 0.570 | 0.353 |
| `tr` | 0.140 | 0.400 | +0.260 | 0.587 | 0.400 |
| `ru` | 0.310 | 0.570 | +0.260 | 0.595 | 0.296 |
| `is` | 0.110 | 0.350 | +0.240 | 0.664 | 0.452 |
| `nb` | 0.330 | 0.560 | +0.230 | 0.518 | 0.294 |
| `fi` | 0.130 | 0.340 | +0.210 | 0.719 | 0.392 |
| `ml` | 0.070 | 0.280 | +0.210 | 0.601 | 0.443 |
| `lv` | 0.100 | 0.310 | +0.210 | 0.666 | 0.472 |
| `hy` | 0.050 | 0.250 | +0.200 | 0.571 | 0.513 |
| `km` | 0.000 | 0.200 | +0.200 | 0.705 | 0.445 |
| `kn` | 0.110 | 0.300 | +0.190 | 0.590 | 0.388 |
| `ta` | 0.120 | 0.310 | +0.190 | 0.554 | 0.448 |
| `it` | 0.340 | 0.520 | +0.180 | 0.516 | 0.343 |
| `da` | 0.350 | 0.520 | +0.170 | 0.526 | 0.315 |
| `sl` | 0.200 | 0.370 | +0.170 | 0.599 | 0.434 |
| `id` | 0.360 | 0.510 | +0.150 | 0.487 | 0.323 |
| `zh-TW` | 0.460 | 0.610 | +0.150 | 0.434 | 0.266 |
| `jv` | 0.160 | 0.300 | +0.140 | 0.670 | 0.484 |
| `ms` | 0.270 | 0.410 | +0.140 | 0.506 | 0.436 |
| `te` | 0.090 | 0.220 | +0.130 | 0.669 | 0.497 |
| `ja` | 0.530 | 0.640 | +0.110 | 0.366 | 0.234 |
| `sv` | 0.380 | 0.490 | +0.110 | 0.487 | 0.341 |
| `my` | 0.060 | 0.160 | +0.100 | 0.580 | 0.477 |
| `sw` | 0.130 | 0.230 | +0.100 | 0.632 | 0.492 |
| `sq` | 0.210 | 0.300 | +0.090 | 0.603 | 0.487 |
| `de` | 0.420 | 0.500 | +0.080 | 0.452 | 0.336 |
| `nl` | 0.390 | 0.470 | +0.080 | 0.502 | 0.397 |
| `es` | 0.510 | 0.580 | +0.070 | 0.422 | 0.258 |
| `af` | 0.290 | 0.350 | +0.060 | 0.559 | 0.439 |
| `ka` | 0.090 | 0.150 | +0.060 | 0.610 | 0.542 |
| `tl` | 0.290 | 0.350 | +0.060 | 0.512 | 0.430 |
| `ro` | 0.330 | 0.370 | +0.040 | 0.550 | 0.397 |
| `pt` | 0.470 | 0.500 | +0.030 | 0.432 | 0.341 |
| `zh-CN` | 0.620 | 0.650 | +0.030 | 0.320 | 0.219 |
| `am` | 0.120 | 0.150 | +0.030 | 0.680 | 0.484 |
| `mn` | 0.130 | 0.160 | +0.030 | 0.670 | 0.577 |
| `cy` | 0.120 | 0.130 | +0.010 | 0.645 | 0.587 |
| `fr` | 0.590 | 0.600 | +0.010 | 0.299 | 0.248 |
| `en` | 0.820 | 0.710 | -0.110 | 0.138 | 0.233 |

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

The English checkpoint does not degrade gracefully outside English — it collapses, and stays confident doing so. Khmer: **0.000 accuracy at 0.952 confidence** raw, **0.705** as served after the clamp. Its mean confidence never drops below 0.885 (raw; 0.621 as served) at any accuracy level, so confidence gating cannot catch it — which is why routing happens *before* the forward pass.

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
| `laya-multilingual` | 0.352 | 0.328 | 0.463 | 0.314 | 0.760 |
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

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.352 against 0.461). All of the capability on this benchmark comes from fine-tuning.

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


## Other hardware: GB10, a laptop CPU, and an Intel Arc

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

### Intel Arc B390 (torch 2.14.0+xpu), XPU — before/after vs CPU

`english` checkpoint (421M, ModernBERT-large), in-process `agent.predict()`, one 2-option `choice` question (~90 tokens), 40 calls per row after 5 warm-ups. XPU row at the default bf16 with autocast (XPU autocast supports bf16/fp16 only); CPU row fp32, pinned as recommended above (intra-op 8, inter-op 1). Rows measured on the same laptop; CPU rows are stable across sessions (p95 within ~10% of p50).

| questions per call | CPU p50 | XPU p50 | XPU p95 | speedup (p50) |
|---|---|---|---|---|
| 1 | 288.2 ms | **29.7 ms** | 30.4 ms | 9.7x |
| 3 | 730.6 ms | **45.4 ms** | 46.8 ms | 16.1x |
| 10 | 2608.7 ms | **96.9 ms** | 102.5 ms | 26.9x |

CPU scales roughly linearly with question count (288.2 -> 2608.7 ms, 9.1x for 10x the questions), while XPU scales sub-linearly (29.7 -> 96.9 ms, 3.3x), so the speedup widens from ~10x to ~27x. The XPU p95 stays within ~6% of its p50 on every row (30.4, 46.8, 102.5). At one question the Arc B390 is slightly faster than the T4's 32.8 ms p50 above.

### Calibration on a routing task runs the other way

On laya_router's 180 requests (zero-shot, one 3-tier `choice`), nearly every configuration we measured was **under**-confident (the few exceptions were +0.01 to +0.06, and among the least accurate). Mean P(chosen) (the chosen option's probability, not the entropy-based `confidence` field) sat below accuracy, by −0.18 on the root checkpoint with example-led tier descriptions (0.562 vs 0.744) and by −0.19 on `typed-decisions` (0.501 vs 0.694). This is one task and one set of labels, so it does not contradict the over-confidence reported above. It does mean the direction of the miscalibration depends on the task, and a temperature fit on your own data is the right fix either way.

## Server CPU: AMD EPYC 9R14, 4 cores, Linux

`research/scripts/bench_latency.py` ran in-process and unchanged at v0.3.20 on an AWS `m7a.xlarge`: 4 physical cores (no SMT), 16 GiB RAM. The run used `OMP_NUM_THREADS=4`, `device="cpu"`, fp32, torch 2.14.0, transformers 5.17.0, Python 3.14.4, and checkpoints at revision `55cf4c4`. Questions alternate a 3-option `choice` and a `noul`. Each row is 10 timed calls after 2 warm-up calls. Raw results: `research/results/latency_cpu_m7a_xlarge_20260924.json`.

| checkpoint | 1 question | 5 | 10 | 50 | cold load |
|---|---|---|---|---|---|
| english | 580 ms | 3,072 ms | 6,244 ms | 35,969 ms | 4.4 s |
| multilingual | 193 ms | 912 ms | 1,842 ms | 11,157 ms | 2.5 s |
| typed-decisions | 584 ms | 2,819 ms | 6,031 ms | 35,653 ms | 0.5 s |

Values are p50. p95 is within 2% of p50 on every row. Up to 10 questions, each question costs about 600 ms on `english` and `typed-decisions` and about 185 ms on `multilingual`. At 50 questions, the cost per question rises by 15–20% on all three. Batching questions saves little on CPU, unlike the GB10 above. Cold load depends on the OS file cache, so treat that column as approximate. Peak memory for the whole script, with up to five checkpoints loaded at once, was 9.3 GiB (maximum RSS).

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
| laya | noul | 180 | 0.076 | **0.043** | 0.058 | 180/180 | 180/180 |
| laya | score | 60 | 0.015 | **0.011** | 0.017 | 59/60 | 60/60 |
| laya-multilingual | choice | 48 | 0.049 | **0.015** | 0.039 | 47/48 | 47/48 |
| laya-multilingual | noul | 180 | 0.037 | 0.045 | 0.045 | 180/180 | 179/180 |
| laya-multilingual | score | 60 | 0.010 | **0.009** | 0.009 | 59/60 | 59/60 |

The fast path stays close to the fp32 reference on every row — at most **0.046** away, against 0.058 for the stock
path on the same row — and no row is more than **0.076** from stock. The residual stream stays in fp32 in both, and
the two bf16 paths differ from each other only by bf16 accumulation order; the few argmax disagreements are near-tie
options, and on every one of them the fast path agrees with fp32. One row is the exception to the stronger reading
that used to be printed here: on `laya-multilingual` `noul` the stock bf16 path is marginally closer to fp32 than the
fast path is (0.0446 against 0.0455), so this table does not show that the fast path is never further from fp32.
Dataset accuracy / ECE (AG News, dair-ai emotion, 1,000 samples each) are identical within noise; see
`benchmarks/bench_fast.py --eval 1000`.

### fp16

The fast path runs in the agent's autocast dtype when `accelerate()` is called, so an agent set to fp16
(`agent.dtype = torch.float16`, or the CUDA autocast override proposed for #443) gets fp16 kernels and fp16
weights; the residual stream and every accumulation stay fp32 in both dtypes. Same fixed set,
`parity_fast.py --dtype fp16 | bf16`, RTX 4070 Ti SUPER; per-option probabilities in
`benchmarks/results/parity_*_rtx4070.json` (the bf16 columns are the table above, plus a bf16 run of
`laya-typed-decisions`):

| checkpoint | type | n | max \|p_fast - p_fp32\| bf16 | max \|p_fast - p_fp32\| fp16 | argmax fast = fp32, bf16 | fp16 |
|---|---|---|---|---|---|---|
| laya | choice | 48 | 0.022 | **0.004** | 47/48 | 48/48 |
| laya | noul | 180 | 0.043 | **0.005** | 180/180 | 180/180 |
| laya | score | 60 | 0.011 | **0.003** | 60/60 | 60/60 |
| laya-multilingual | choice | 48 | 0.015 | **0.002** | 47/48 | 48/48 |
| laya-multilingual | noul | 180 | 0.045 | **0.009** | 179/180 | 180/180 |
| laya-multilingual | score | 60 | 0.009 | **0.001** | 59/60 | 60/60 |
| laya-typed-decisions | choice | 48 | 0.019 | **0.002** | 47/48 | 47/48 |
| laya-typed-decisions | noul | 180 | 0.023 | **0.005** | 180/180 | 180/180 |
| laya-typed-decisions | score | 60 | 0.009 | **0.001** | 60/60 | 60/60 |

In fp16 the fast path is 3-10x closer to fp32 than in bf16 and agrees with the fp16 stock path on every argmax
(864/864; the most it moves a probability against fp16 stock is 0.009). The one fp16 disagreement with fp32 is a
`laya-typed-decisions` choice question whose top two options are 0.001 apart in fp32; the fp16 stock path flips it too.
`agent.predict()` latency shows no consistent difference between the dtypes: on every case of the table below, on
both checkpoints, fp16 and bf16 are within 10% of each other in both directions (single runs of 50 iterations), for
stock and fast alike.

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
