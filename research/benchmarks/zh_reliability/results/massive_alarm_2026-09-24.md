# MASSIVE Chinese alarm calibration evaluation

Measurement date: **2026-09-24**. Execution base: `23a17522aa4942da6cce53a995a275760320b691`; contribution: `37cc58465da5cdabddf1ff543234397d666ad546`.

Historical inference was executed on 2026-09-24 from base commit 23a17522aa4942da6cce53a995a275760320b691 with a dirty contribution worktree subsequently recorded in 37cc58465da5cdabddf1ff543234397d666ad546. The recorded execution source-file hashes are authoritative for the measured code. These measurements were not rerun on the later upstream rebase or evaluation-only refactor.

E0 and E1 use the same multilingual checkpoint and raw decision logits. E1 fits one temperature per question type on calibration data. No model weights are updated.

| Condition | Slice | n | Accuracy | Macro F1 | NLL | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | overall | 186 | 0.532258 | 0.440840 | 1.383747 | 0.730071 | 0.366488 |
| E0 | choice | 93 | 0.634409 | 0.580928 | 1.296043 | 0.520825 | 0.253426 |
| E0 | noul | 93 | 0.430108 | 0.300752 | 1.471450 | 0.939317 | 0.494611 |
| E1 | overall | 186 | 0.532258 | 0.440840 | 0.785033 | 0.506898 | 0.197529 |
| E1 | choice | 93 | 0.634409 | 0.580928 | 0.837271 | 0.474032 | 0.169905 |
| E1 | noul | 93 | 0.430108 | 0.300752 | 0.732796 | 0.539765 | 0.289635 |

Positive scalar temperature preserves argmax: accuracy and F1 are unchanged.

| Type | Calibration n | Temperature | Bound reached |
|---|---:|---:|---|
| choice | 64 | 3.135504 | False |
| noul | 64 | 5.000000 | True |

Paired differences are E1 minus E0; negative NLL, Brier and ECE differences are better.

| Metric | Difference | 95% group bootstrap interval |
|---|---:|---|
| accuracy | +0.000000 | [+0.000000, +0.000000] |
| macro_f1 | +0.000000 | [+0.000000, +0.000000] |
| nll | -0.598713 | [-0.865774, -0.369099] |
| brier | -0.223173 | [-0.298826, -0.152890] |
| ece | -0.168958 | [-0.224995, -0.098547] |

## Scope and limits

- Test: 93 source utterances, 89 heuristic groups, 186 correlated decisions. Calibration: 64 source utterances, 62 groups, 128 decisions. The paired question types are not independent samples.
- This is a derived MASSIVE 1.1 alarm subset with upstream human intent-review evidence, not the full benchmark, naturally occurring customer-service logs, or independent local human review.
- One fixed split/seed; 1,000 group bootstrap draws. Intervals condition on the fixed checkpoint and fitted temperatures, omit calibration uncertainty, and are descriptive without multiplicity correction.
- Unknown checkpoint exposure to MASSIVE/SLURP; grouping may miss semantic dependencies. These results do not establish general Chinese improvement or unchanged behavior on other tasks.
- The baseline noul fit reaches T=5, the permitted upper bound. Calibration settings were fixed without choosing on test. Risk/coverage does not improve at every coverage.
- Configured max_len=512, head_max_len=192, FP16. Historical test inputs were at most 60 tokens; this is not a 512-token stress measurement.
- The compact evidence omits source text and retains source IDs/hashes. It can recompute numerical results and check pairing; full source-schema/annotation checks require the prepared public dataset.
