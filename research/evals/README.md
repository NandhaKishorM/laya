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
