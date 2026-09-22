# Laya benchmarks

Every checkpoint answered **byte-identical questions** in each run (fixed seed). Jev figures are **third-party published, never measured here** — no TypeSafe API access — so sample sizes and prompts differ; treat them as indicative.

| run | what | where |
|---|---|---|
| T4 Colab | typed-decisions, MASSIVE (14 langs), XNLI (15 langs), English suites, latency, option-order robustness, calibration repair | `research/results/t4_colab_benchmark.json` |
| CPU sweep | MASSIVE intent across **all 51 languages**, typed-decisions on all three checkpoints | `research/results/cpu_51_language_sweep.json` |
| Applications | the six workflow themes + the datasets where Jev numbers exist, all three checkpoints | `research/results/app_benchmark.json` |

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

### Arabic regression suite (local CPU)

The Arabic suite uses 36 AI-authored synthetic messages: 12 parallel scenarios in
standard Arabic, Emirati Arabic, and mixed Arabic-English. Each message is tested
with Arabic and English question wording (72 evaluations, 216 typed decisions).
The examples and labels have **not been native-speaker reviewed**. These small,
related samples measure regression behavior, not general Arabic proficiency.

Measured on 2026-09-22 using the multilingual checkpoint at Hugging Face revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`, on macOS arm64 CPU:

| Question language | Message group | Category accuracy | Refund accuracy | Urgency MAE (0–2, lower better) |
|---|---|---:|---:|---:|
| Arabic | Standard Arabic | 91.7% | 91.7% | 0.586 |
| Arabic | Emirati Arabic | 83.3% | 83.3% | 0.618 |
| Arabic | Mixed | 83.3% | 58.3% | 0.727 |
| English | Standard Arabic | 75.0% | 75.0% | 0.700 |
| English | Emirati Arabic | 75.0% | 75.0% | 0.607 |
| English | Mixed | 83.3% | 75.0% | 0.618 |

All 72 evaluations routed to multilingual without a locale override. Routing
success is separate from prediction accuracy. The category majority baseline is
25%; the refund majority baseline is 75%; always predicting urgency 1 gives MAE
0.667. **Arabic-question refund detection on mixed text falls below baseline.**
Urgency prediction is also weak. Arabic wording improves category accuracy on
this corpus, but does not consistently improve every decision type.

The [raw report](research/results/arabic_triage_cpu.json) contains all predictions,
labels, calibration metrics, warm latency, configuration, and source hashes.
See the [runner and reproduction commands](research/README.md#arabic-regression-benchmark).
No weights or temperatures were fitted to these cases.

### Cross-workflow Arabic routing comparison

80 synthetic messages cover intent, sentiment, moderation, relevance, and urgency
in standard Arabic, Emirati, mixed text, and English controls. Each is evaluated
with Arabic and English question wording under two routing policies. There are
160 decisions per policy; the predictions are reused where both policies select
the same checkpoint. This is a routing comparison using identical weights and
the current inference runtime, not a retrained model comparison.

| Question language | Task | Original routing | Arabic-aware routing |
|---|---|---:|---:|
| Arabic | Intent accuracy | 100% | 100% |
| Arabic | Sentiment accuracy | 87.5% | 100% |
| Arabic | Moderation accuracy | 81.3% | 87.5% |
| Arabic | Relevance accuracy | 93.8% | 87.5% |
| Arabic | Urgency MAE (lower better) | 0.640 | 0.553 |
| English | Intent accuracy | 93.8% | 93.8% |
| English | Sentiment accuracy | 93.8% | 100% |
| English | Moderation accuracy | 93.8% | 93.8% |
| English | Relevance accuracy | 93.8% | 87.5% |
| English | Urgency MAE (lower better) | 0.580 | 0.641 |

Each row aggregates 16 cases, including English controls. The expected routing
policy matches 100% of cases after the fix versus 50% with Arabic questions and
75% with English questions before it. **Prediction quality does not improve
uniformly:** relevance regresses under both question languages, and urgency error
increases with English questions. The routing fix ensures Arabic reaches the
multilingual model, but does not resolve all model-quality gaps. Do not infer a
general quality gain from routing accuracy. No workflow-specific checkpoint
selection was fitted to these results.

This corpus was authored after a small triage prompt exploration; it was not
used for prompt selection or training. It is still synthetic, unreviewed, and
too small to qualify production behavior (four cases per task/message group).
Native-speaker-reviewed, representative evaluation and separate training data
are needed for broader model improvement.

[Raw paired results](research/results/arabic_workflows_cpu.json) include every
answer, per-group metrics and baselines, checkpoint revisions, and source hashes.
[Reproduction commands](research/README.md#cross-workflow-arabic-comparison).

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

---

## Limits, stated plainly

- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400).
- **Keep `choice` questions under ~20 options.**
- **Both checkpoints ship over-confident.** Fit temperatures on your own data.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route.
