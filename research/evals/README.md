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
and labels three independent decisions per message: category, urgency and churn risk
(an expressed intention or conditional threat to leave the service). The score levels run from 0 (not urgent) to 4 (critical).
The churn question is deliberately separate from category: a technical problem can
include a cancellation threat without becoming a cancellation request.
The urgency rubric describes concrete impact and deadline conditions at each level;
the instruction asks the model to judge those facts rather than a customer's tone or
word choice. Score criteria should stand alone as ordered descriptions of distinct
situations, as recommended by TypeSafe's [Score guidance](https://docs.typesafe.ai/primitives/score.md).

The Latin-script detector now has Swedish-specific vocabulary, including a conservative
two- or three-word path for short support fragments such as `Ingen åtkomst` and
`Glömt lösenord`. The current model-free routing pass selects the multilingual checkpoint
and names Swedish for all 30 messages. An earlier pass named Swedish for 29; adding support
vocabulary resolved the remaining Spanish guess on a billing message. Short Nordic greetings and acknowledgements such as `Hej`, `Ja`, `Nej`, `Jo` and `Tack` route
to the multilingual checkpoint without being misidentified specifically as Swedish. Regression checks cover Swedish with and without diacritics, short fragments, and Danish,
Norwegian and English controls. These routing examples informed the detector changes and are
development diagnostics, not an independent quality estimate. Language identification remains best-effort; pass
`lang="sv-SE"` when the caller knows the language and needs deterministic routing.

As a separate routing diagnostic, the MASSIVE test split at dataset revision
`940fd47a81eaa7f2cc7b129674d945d618ac38c2` contains 2,974 examples each for Swedish, Danish,
and English. The current router sent 2,586 Swedish and 1,589 Danish examples to the multilingual
checkpoint, and all 2,974 English examples to the English checkpoint. It explicitly named Swedish
for 901 examples; many of the remaining Swedish examples were still routed multilingual from
non-English letters or shared Nordic wording. This voice-assistant set is a language-routing
control, not an evaluation of support quality. Reproduce these model-free counts with
`python research/evals/swedish_route_diagnostic.py`; the script pins the dataset revision and does
not load model weights.

Validate the file, then run it against the multilingual checkpoint:

```bash
laya-evals validate research/evals/swedish_support.jsonl
laya-evals run research/evals/swedish_support.jsonl --model multilingual --device cpu \
    --revision multilingual=55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 \
    --slice qid --slice tag --json research/results/swedish_support_multilingual_cpu.json
```

Initial CPU run with the first urgency rubric (Laya 0.3.21, multilingual revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, 2026-09-28):

| decision | result |
|---|---:|
| Category accuracy (30 cases) | 0.6333 |
| Urgency mean absolute error (0–4 scale) | 1.0024 |
| Cancellation-threat accuracy (30 cases) | 0.4333 |
| Overall ECE | 0.3591 |
| CPU latency, p50 / p95 per case | 88.4 / 91.7 ms |

After clarifying the Swedish urgency criteria and rerunning the same 30 examples
against the same pinned checkpoint, urgency MAE was 0.9356 (0–4), category accuracy
remained 0.6333, and cancellation-threat accuracy remained 0.4333. The updated run's
CPU latency was 159.4 / 166.9 ms (p50 / p95). The score MAE is not directly comparable
to the initial run because the rubric wording changed; this remains a small synthetic
development set, not an independent quality estimate. Its full report is
[`swedish_support_multilingual_cpu.json`](../results/swedish_support_multilingual_cpu.json);
the initial report is retained as
[`swedish_support_multilingual_cpu_initial_rubric.json`](../results/swedish_support_multilingual_cpu_initial_rubric.json).

The original cancellation question had a clear failure signal: all 3 explicit threats
were detected, but 17 of the 27 non-threat messages were also marked positive (13/30
accuracy). In a follow-up, the target was clarified as churn risk: either an expressed
intention to leave or a conditional cancellation threat, while questions about terms and
hypotheticals remain negative. Updating the two direct-cancellation labels accordingly
and adding explicit positive/negative criteria raised accuracy to 24/30 (80%) on the
same synthetic cases. However, several severe service problems without any expressed
intent were still false positives. A further wording change that explicitly excluded
severity and frustration fell to 14/30 (46.7%). The two follow-up reports are preserved
as [`churn revision 1`](../results/swedish_support_multilingual_cpu_churn_revision.json)
and [`churn revision 2`](../results/swedish_support_multilingual_cpu_churn_revision2.json).
This prompt sensitivity means the diagnostic does not establish reliable churn detection;
review the target labels and wording with a Swedish support specialist and evaluate on
independently collected, adjudicated tickets before relying on it.
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
[`cpu_51_language_sweep.json`](../results/cpu_51_language_sweep.json); the separate
support diagnostic has been rerun with the clarified rubric as described above.
