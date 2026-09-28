# research/evals

Datasets, baselines and thresholds for the `laya-evals` harness and its CI gate.

- `fixture.jsonl`: a tiny hand-written set that exercises `choice`, `score` and `noul`
  across a few tags. It exists to prove the format and to run the harness in tests. It is
  **not** a quality claim and its accuracy has no meaning.
- `dataset.template.jsonl`: the format, with comments. Start here.
- `thresholds.json`: the tolerances the scheduled gate allows against the committed
  baselines in `research/results/`.
- `check_regression.py`: adapts a `research/eval/laya_eval.py` report into
  `laya.evals.EvalReport` and compares it to a committed baseline.
- `act_head_eval.py`: scores `answer.action.act_probability` against labelled
  decisions in a report, as the bar a future act-head retraining has to beat.
  Pure arithmetic over a report `laya-evals run --json` already wrote: no second
  model load, and no change to model, routing or `laya.evals` code. The current
  checkpoints never learned a useful act head, so this measures, it does not
  fix, and a number here is not a claim the signal is informative today.
- `test_act_head_eval.py`: weight-free tests for the above. Like the
  `research/eval/test_*.py` suites, run it directly; CI does not collect it.

## Format

One JSON object per line (JSONL). Blank lines and lines starting with `#` are ignored.

| field | required | meaning |
|---|---|---|
| `state` | yes | the text, email, ticket or JSON document to decide on |
| `questions` | yes | a Laya question dict, exactly as `Router.predict` accepts |
| `expected` | yes | ground truth keyed by question id: a label for `choice`, a number for `score`, `true`/`false` for `noul` |
| `tags` | no | strings to slice by |
| `language` | no | BCP-47-ish code, to slice by language |
| `model` | no | force a checkpoint for this row; `--model` overrides it |

## Use

```bash
laya-evals validate research/evals/fixture.jsonl
laya-evals run data.jsonl --model english --min-accuracy 0.8 --max-ece 0.05 --slice language
laya-evals run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 --json out.json
```

`run` exits non-zero when a threshold or a baseline tolerance fails, so it drops into CI
unchanged. `laya eval ...` is the same thing through the main CLI.

Adding a dataset: point `--baseline` at a report you have reviewed, keep the tolerances in
`thresholds.json`, and commit both beside the dataset, so a quality change is a reviewable
diff.

## Act-head diagnostic

```bash
laya-evals run data.jsonl --model english --json report.json
python research/evals/act_head_eval.py report.json --markdown act.md
```

Answers one question: does `act_probability` rank correct decisions above incorrect ones?
It reports the scorable count, the correct/incorrect split, the tie-correct ROC-AUC of
`act_probability`, the same AUC for the calibrated `confidence` on **exactly** the same
decisions as a control, and the observed min/max/unique of `act_probability` so a signal
pinned at one value is visible rather than averaged away.

Cases are bucketed explicitly and nothing is folded into a denominator: a `score` answer has
no binary `correct` and is skipped with a count, a missing or non-numeric
`act_probability` is reported as missing rather than read as `0.0`, and an AUC is `null`
with a reason when either class is absent, since ROC-AUC is undefined there.

The AUC is the Mann-Whitney U form with mid-ranks for ties, which agrees with
`sklearn.metrics.roc_auc_score` and needs only numpy. Ties are not a rounding detail here:
the current signal is constant, so a tie-correct implementation is the difference between
reporting a failure and reporting 0.5 as if it were a measurement.

## The real labelled set

The maintainer's 396-decision English set and the 51-language sweep are the authoritative
numbers. They are not committed here yet; drop a JSONL in this directory and a baseline
report beside it and the gate will pick both up. The scheduled workflow currently runs the
MASSIVE English suite through `research/eval/laya_eval.py` against
`research/results/eval_english_51_languages.json`.

## Swedish support diagnostic

`swedish_support.jsonl` is a small, hand-written diagnostic set of 30 synthetic
Swedish support messages. It has five examples for each of six request categories
and labels three independent decisions per message: category, urgency and an
explicit threat to cancel. The score levels run from 0 (not urgent) to 4 (critical).
The cancellation question is deliberately separate from category: a technical
problem can include a cancellation threat without becoming a cancellation request.

The Latin-script detector now has Swedish-specific vocabulary. A model-free routing
pass selected the multilingual checkpoint for all 30 messages and named Swedish for
29; one billing message received a Spanish language guess but still reached the right
checkpoint. The regression checks also cover Swedish text with diacritics stripped.
Language identification remains best-effort; pass `lang="sv-SE"` when the caller
knows the language and needs deterministic routing.

Validate the file, then run it against the multilingual checkpoint:

```bash
laya-evals validate research/evals/swedish_support.jsonl
laya-evals run research/evals/swedish_support.jsonl --model multilingual --device cpu \
    --revision multilingual=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
    --slice qid --slice tag --json research/results/swedish_support_multilingual_cpu.json
```

First CPU run (Laya 0.3.21, multilingual revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, 2026-09-28):

| decision | result |
|---|---:|
| Category accuracy (30 cases) | 0.6333 |
| Urgency mean absolute error (0–4 scale) | 1.0024 |
| Cancellation-threat accuracy (30 cases) | 0.4333 |
| Overall ECE | 0.3591 |
| CPU latency, p50 / p95 per case | 88.4 / 91.7 ms |

The cancellation result is driven by false positives: all 3 explicit threats were
detected, but 17 of the 27 non-threat messages were also marked positive. This is a
clear failure signal for that question on this sample. The full machine-readable
report is [`swedish_support_multilingual_cpu.json`](../results/swedish_support_multilingual_cpu.json).
Category accuracy was strongest for delivery and cancellation (4/5 each) and weakest
for sales inquiries (1/5). This points to useful follow-up cases around Swedish
product-plan and pricing language, but the set needs independent review before any
prompt or model changes are evaluated against it.

This is a first diagnostic, not a representative Swedish test set: the messages are
synthetic, the labels have not been independently adjudicated, and 30 cases are too
few to support a general quality claim. Do not tune on it and present the same cases
as held-out results. The JSON report records per-question metrics and every decision
so errors can be inspected. No baseline or CI gate is set until a checkpoint run has
been reviewed; the existing 51-language MASSIVE result (`sv`, 100 examples, 20
choices) measures a different task and should not be compared directly.

For context, the committed multilingual MASSIVE sweep reports Swedish accuracy 0.57,
macro-F1 0.5240 and ECE 0.2757 on those 100 test utterances (CPU, 20 choices).
That is a useful existing language-level reference, but it does not measure support
triage, urgency or cancellation detection. The source is
[`cpu_51_language_sweep.json`](../results/cpu_51_language_sweep.json); a fresh run of
`swedish_support.jsonl` is still needed before reporting results for these tasks.
