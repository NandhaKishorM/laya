# Benchmarks and known limits

Use the [full benchmark report](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
for run conditions, per-language tables, scripts, and raw-result paths. This guide explains what
those measurements can tell you when deciding whether Laya fits a workload. The numbers are
observations for the stated datasets and hardware, not accuracy or latency guarantees for new
inputs.

## Read a result in its own setting

| Result in the report | What it shows | What it does not show |
|---|---|---|
| 0.766 accuracy on 2,000 typed decisions | The **fine-tuned** `laya-typed-decisions` checkpoint on that workflow benchmark | Zero-shot accuracy of the base checkpoints: they scored 0.361 and 0.352 on the same benchmark |
| 0.3661 macro accuracy on 51-language MASSIVE intent | `laya-multilingual` on 20-option intent questions, averaged across languages | Accuracy for every language or a different set of labels; the per-language results vary widely |
| 0.530 accuracy on held-out moderation | The tested checkpoints on a balanced toxic-chat split | Reliable moderation on live traffic; this is a weak result for that task |
| 32.8 ms p50 for one question | `laya-multilingual` on the measured Tesla T4 run | CPU latency, cold checkpoint loading, or tail latency on your server |

The report includes published Jev figures for context. Those figures were **not measured in the
same run**: sample sizes and prompts differ, so the cross-system columns are indicative rather
than a controlled head-to-head result. The 51-language sweep's original ECE and mean-confidence
columns also predate a temperature clamp. Use the post-clamp comparison in the
[full report](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) and its
[raw results](https://github.com/NandhaKishorM/laya/blob/main/research/results/cpu_51_language_sweep_clamped.json)
when discussing current confidence; its accuracy is unchanged.

## Limits to test before adopting a decision

- **Many choices share one token budget.** `head_max_len` is 192 tokens for the default English
  checkpoint and 256 for the multilingual and typed-decisions checkpoints. On banking77's
  77-label choice, the measured accuracies were 0.425, 0.425, and 0.492 respectively. The
  benchmark report recommends keeping ordinary `choice` questions under about 20 options.
  For larger catalogs, test a larger head budget or narrow candidates first with
  [`predict_shortlist`](https://github.com/NandhaKishorM/laya/blob/main/README.md#honest-limits).
  Raising the budget can leave less room for the state, so check both option text and state
  truncation on representative inputs.
- **Language routing matters.** In the 51-language sweep, the English checkpoint's macro
  accuracy was 0.2269 and the multilingual checkpoint's was 0.3661 on the same intent task.
  English itself favored the English checkpoint. Test the languages and mixed-language inputs
  your application actually receives; inspect the Router's decision instead of assuming that
  one checkpoint works equally well everywhere.
- **Confidence needs local calibration.** On one T4 benchmark, temperature fitting on held-out
  data moved ECE from 0.466 to 0.081 for `laya` and from 0.314 to 0.106 for
  `laya-multilingual`. Another measured routing task was *under*-confident, so the direction
  of the error is task-dependent. When setting an escalation threshold, evaluate
  `answer_confidence` against outcomes from your own held-out data. It is the probability of
  the reported answer; the separate `confidence` field measures distribution concentration
  for `choice` and `score`.
- **Some tasks remain weak.** Held-out moderation scored 0.530 accuracy and 0.400 macro-F1;
  the report also calls ordinal `score` the weakest primitive on its SST-5 test. Test each
  question type and task separately. A strong spam result does not establish moderation
  quality.
- **Option order can change a choice.** On the 20-option English MASSIVE intent test,
  permuting options changed the selected answer in 0.150 of English-checkpoint cases and
  0.230 of multilingual-checkpoint cases. Repeat important evaluations with alternate orderings
  before relying on a close ranking.

## Evaluate your own workload

1. **Freeze representative cases.** Keep real input shapes, languages, option counts, and
   ground-truth decisions. Split cases used to choose instructions or thresholds from cases
   reserved for the final evaluation.
2. **Record the exact setup.** Include the Laya and checkpoint revisions, selected route,
   `max_len`, `head_max_len`, option wording and order, device, dtype, and relevant thread
   settings. A checkpoint or prompt change creates a new experiment.
3. **Measure decision quality by task.** Report accuracy and per-class results for `choice`;
   error by level for `score`; and false positives and false negatives for `noul`. Check
   calibration and coverage at your proposed `answer_confidence` threshold. Keep results by
   language and option count so a good aggregate cannot hide a failing slice.
4. **Probe the boundaries.** Reorder options, include long states, and test the largest catalog
   you expect. Confirm the answer and routing metadata, and check whether truncation removes
   evidence the question needs.
5. **Measure latency on the target machine.** Separate warm inference from checkpoint loading,
   and report p50 and p95 for your actual questions per call. The
   [latency script](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_latency.py)
   is a starting point; it expects local checkpoints under `~/laya_models` and writes its
   result into `research/latency_benchmark_results.json`.

If a slice fails, try a clearer question, fewer candidate labels, a checkpoint better suited
to that task, or fine-tuning and calibration on separate training data. Re-run the held-out
evaluation after each change before promoting it.
