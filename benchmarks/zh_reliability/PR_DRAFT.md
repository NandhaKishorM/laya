# feat(benchmarks): add Chinese decision evaluation, calibration and head tuning

## What and why

Add an opt-in local workflow for assessing Chinese `choice`/`noul` decisions on the real multilingual Laya checkpoint. It preserves unrounded decision logits, separates train/dev/calibration/test by source group, fits type-wise temperatures without double scaling, and supports frozen-encoder supervised head tuning with validated resume and an Agent-loadable export. The runtime API is unchanged.

The public-data preparer downloads fixed MASSIVE 1.1 Chinese/English artifacts, retains CC BY 4.0 attribution and upstream intent-review evidence, and derives an alarm-intent subset without changing source utterances or labels. Local human-review flags stay false. Source/template overlaps, conflicting labels and contested annotations remain in diagnostics. The 80 agent-authored fixtures remain separate regression diagnostics.

Training now supports `--checkpoint-every` (default 1). It schedules dev selection and immutable recovery checkpoints by successful update count, includes the final update, and logs every update. This made the measured 198-update run practical with the available disk: 18 saved candidates at interval 11 require about 3 GiB instead of 33 GiB. Resume binds the interval while preserving default compatibility.

## Measured public-data experiment

The fixed source-reviewed subset contains 471 utterances: 261 train, 53 dev, 64 calibration and 93 test. Each utterance has choice/noul views; the test therefore has 186 correlated decisions in 89 heuristic source groups. The 79 excluded alarm rows are retained with reasons.

The protocol was fixed before predictions: seed 20260924; multilingual revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`; FP16; max_len=512/head_max_len=192; micro-batch 1, accumulation 8; AdamW lr=1e-4, weight_decay=.01; 198 successful head updates on TITAN RTX. Dev NLL was checked only at steps 11,22,...,198 and selected step 198. Two initial scaler overflows were skipped; final cursor was epoch 3/index 16 (zero-based), so this is slightly over three data passes. Encoder/action parameters stayed unchanged; decision parameters changed. Export/reload raw-logit difference on the training card was zero.

Calibration used 64 examples per type with the normal minimum of 30. All four test conditions were then evaluated on RTX 2080 Ti after weights and temperatures were locked:

| Condition | Accuracy | Task macro F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| E0: Original checkpoint | 0.532258 | 0.440840 | 1.383747 | 0.730071 | 0.366488 |
| E1: Original + calibration | 0.532258 | 0.440840 | 0.785033 | 0.506898 | 0.197529 |
| E2: Head tuning, T=1 | 0.758065 | 0.729674 | 0.595625 | 0.350510 | 0.099407 |
| E3: Head tuning + calibration | 0.758065 | 0.729674 | 0.576033 | 0.332620 | 0.060256 |

E3−E0 accuracy changed by **22.58 percentage points**, with a paired source-group bootstrap 95% interval of [15.16, 29.67] points. Against the calibrated baseline, E3−E1 NLL changed by -0.209000 [-0.278481, -0.130405]. These are conditional, exploratory intervals from 1,000 cluster resamples, not uncertainty over repeated training or calibration fits.

The further calibration gain after tuning is smaller: E3−E2 NLL is -0.019592 [-0.059509, 0.030108]; the interval crosses zero. Base noul calibration hit T=5. The tuned choice/noul temperatures were 0.531014 and 0.718293. Temperature fitting did not change classifications. By type, choice accuracy was 63.44%→68.82%; noul was 43.01%→82.80%.

Timing used 10 warmup and 186 measured single-decision batches per stage. E0/E2 end-to-end p50 was 52.556/56.141 ms, p95 71.509/84.011 ms, with 1507.7 MiB peak allocated in both. This does not show a speedup. Actual test inputs were 43–60 tokens with no truncation; max_len=512 is a configured limit, not a long-input stress result. GPUs were not exclusively reserved.

## Validation and limits

* `python -m unittest discover -s tests -p 'test_zh_*.py' -v`: 56 passed, one opt-in checkpoint test skipped. Tiny-model tests cover frozen weights, optimizer updates, interval/tail checkpoints, exact interrupted/resumed updates, corruption refusal and API parity. Fabricated-source tests cover public-data conversion, evidence, grouping, pinned downloads and decoding.
* Required runtime Ruff and compileall gates passed; the same checks cover the benchmark modules. No public API changes were made.
* Seven real GPU stages completed successfully. Aggregation verified source/model/data/split hashes, unique paired test IDs, probabilities rebuilt from raw logits, calibration bindings, unchanged raw logits across temperature changes, and official API parity for all four conditions.
* This is one seed on a filtered public human-localized alarm subset, not full MASSIVE, natural customer-service data or general Chinese capability evidence. There is no independent local human review; unknown semantic dependencies and base-model exposure to MASSIVE/SLURP remain possible.
* English, other domains, score/action output quality, full-encoder tuning and long-document stress are unmeasured. Shared decision parameters may affect untested tasks. No deployment-readiness claim follows.
* Earlier fixture-only GPU smoke, FP16/FP32 cross-card checks and real resume parity are retained as engineering validation, separate from these public-data metrics. The dataset is not sent to hosted inference.

Commands and definitions are in `README.md`. The committed `reproduce_massive.sh`, `massive_protocol.json` and `aggregate_massive.py` reproduce and verify the seven-stage protocol from prepared data and a pinned local model. Each new run binds its actual source paths and hashes before predictions. Recorded local artifacts are under `runs/massive-experiment-v1/`; source data and attribution are under `runs/massive-alarm-data/`. Generated data, models and results remain ignored and are not prerequisites shipped in this PR.

## Suggested review split

A: data preparation, validation, metrics, calibration, runtime adapter and evaluation CLI/tests. B: supervised head tuning, interval checkpoints, resume/export and corresponding CLI/tests. The shared CLI/test files can be split by commands/classes without duplicating the data or evaluation implementation.
