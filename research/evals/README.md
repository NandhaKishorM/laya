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

## The real labelled set

The maintainer's 396-decision English set and the 51-language sweep are the authoritative
numbers. They are not committed here yet; drop a JSONL in this directory and a baseline
report beside it and the gate will pick both up. The scheduled workflow currently runs the
MASSIVE English suite through `research/eval/laya_eval.py` against
`research/results/eval_english_51_languages.json`.
