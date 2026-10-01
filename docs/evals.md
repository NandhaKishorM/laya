# Evaluation harness

`laya.evals` turns a labelled dataset into a repeatable score, and a baseline into a
pass/fail gate, so a quality change is a reviewable diff instead of a hand-check.

The metric math and the dataset parser are pure Python plus numpy and never import torch, so
they run with no weights. Running a dataset against a checkpoint needs the checkpoint and
takes its normal load time.

## Quickstart

```bash
# check the format without a model
laya-evals validate research/evals/fixture.jsonl

# score a labelled set on one checkpoint, with thresholds and a baseline
laya-evals run data.jsonl --model english --device cpu \
    --min-accuracy 0.8 --max-ece 0.05 --score-within 0.25 --slice language \
    --json report.json --markdown report.md

# compare a saved report to a baseline
laya-evals compare report.json --baseline baseline.json --tolerance choice_accuracy=0.02
```

`laya eval ...` is the same thing through the main CLI, so `laya eval validate data.jsonl`
works too.

Exit codes: `0` on success, `1` when a threshold or a baseline tolerance fails, `2` on a
usage error. `run` prints the overall metrics and any requested slices to stdout, and writes
the full report and a Markdown summary when `--json` / `--markdown` are given.

## Attributing shortlist errors

For a labelled high-cardinality choice set, `laya.evals_shortlist.evaluate_shortlist`
uses the existing `predict_shortlist` path and the regular evaluation harness. It
answers two separate questions: did retrieval keep the gold label, and did Laya
choose it when it was present? This is an opt-in Python API for choice labels;
ordinary `laya-evals run` reports are unchanged.

```python
import laya
from laya.evals import Dataset
from laya.evals_shortlist import evaluate_shortlist
from laya.shortlist import embed_fn_from_agent

agent = laya.load()
dataset_path = "intents.jsonl"
dataset = Dataset.from_jsonl(dataset_path)
report = evaluate_shortlist(
    agent, dataset, embed_fn_from_agent(agent), k=20,
    checkpoint_id="my-checkpoint@revision", embedder_id="my-encoder@revision",
    dataset_path=dataset_path,
)
print(report.overall)
print(report.cases[0]["shortlist_status"])
```

Use the same embedding function and checkpoint as the deployment being measured.
The two identifiers are supplied by the caller and should name immutable revisions;
the report cannot infer the weights behind an arbitrary callable. `dataset_path`
records the file's SHA256 alongside the existing question fingerprint. Each case
keeps the actual shortlist labels and one of `correct`, `retrieval_miss`, or
`decision_miss`. `shortlist_recall_at_k` is the fraction of gold labels retained.
`shortlist_accuracy_on_recalled` is correct decisions divided by retained cases;
it is omitted when none were retained. The existing `choice_accuracy` remains
end-to-end accuracy over all cases, including retrieval misses. The shortlist
metrics appear in the same language, model, question and tag slices. Request
latency includes embedding and the decision call; the report does not isolate
stage timings. With `k >= n`, the original question passes through and retrieval
recall is 1 without calling the embedder.

This does not reproduce the BANKING77 results in [issue #102](https://github.com/NandhaKishorM/laya/issues/102):
those numbers depend on its dataset, checkpoint and bi-encoder. This API makes the
same kind of diagnosis repeatable on a caller's own labelled set.

## Evaluating an ONNX export

`run --onnx PATH` scores an exported ONNX model through `ONNXAgent` instead of the torch
Router, so an ONNX deployment (including an INT8 copy from `scripts/export_onnx.py --quantize`)
gets gated by the same thresholds and baselines as the torch path:

```bash
python scripts/export_onnx.py --model convaiinnovations/laya --output laya.onnx --quantize
laya-evals run data.jsonl --onnx laya.int8.onnx --max-ece 0.05
```

`--model` names the checkpoint the export came from — a Hub id or local path, not a Router
short name like `english`, since there is no Router on this path (default
`convaiinnovations/laya`). Its config and tokenizer are loaded from there. The agent serves one checkpoint, so a dataset row
whose `model` field names a different one fails with a clear error rather than being silently
answered by the wrong model; `--device` does not apply. `--batch-size` uses the agent's batch
API when it has one and falls back to one call per state otherwise; `--sort-by-length` is forwarded
to that batch API, which the per-state fallback has no group to reorder. Pass `--calibration PATH`
to load a fitted calibration map onto `ONNXAgent`, so calibration gates such as `--max-ece`
evaluate against calibrated probabilities. The report's `config` block records the `onnx` path
and `calibration` path (when set).

Measured on `research/evals/fixture.jsonl` (12 labelled rows, English checkpoint, CPU):

| runner | choice_acc | noul_acc | score_mae | ece | mean_conf | p50 ms |
|---|---|---|---|---|---|---|
| torch Router | 0.75 | 1.00 | 1.3418 | 0.1596 | 0.7304 | 116.8 |
| `--onnx` fp32 | 0.75 | 1.00 | 1.3418 | 0.1596 | 0.7304 | 66.3 |
| `--onnx` int8 | 0.75 | 1.00 | 1.3512 | 0.1658 | 0.7304 | 46.3 |

The fp32 export reproduces the torch numbers exactly, and the quantized copy moves `score_mae`
by 0.009 and `ece` by 0.006 — the kind of drift `compare --tolerance` is meant to gate.

## Dataset format

One JSON object per line (JSONL). Blank lines and lines starting with `#` are ignored.

| field | required | meaning |
|---|---|---|
| `state` | yes | text, email, ticket or JSON document to decide on |
| `questions` | yes | a Laya question dict, exactly as `Router.predict` accepts |
| `expected` | yes | ground truth keyed by question id: a label for `choice`, a number for `score`, `true`/`false` for `noul` |
| `tags` | no | strings to slice by |
| `language` | no | a code to slice by |
| `model` | no | force a checkpoint for this row; `--model` overrides it. A row that forces nothing is labelled with whatever checkpoint the `Router` answered with |

`research/evals/dataset.template.jsonl` has a commented example.

## Metrics

Each metric is computed per answer where it applies and aggregated over the dataset:

| metric | applies to | meaning |
|---|---|---|
| `choice_accuracy` | `choice` | fraction whose chosen label matches |
| `noul_accuracy` | `noul` | fraction whose boolean (probability >= 0.5) matches |
| `score_mae` | `score` | mean absolute error |
| `score_within_<tol>` | `score` | fraction within an absolute tolerance |
| `ece` | any answer with a confidence | expected calibration error, 15 bins, computed on `answer["answer_confidence"]`, the calibrated probability Laya reports on every answer type |
| `mean_confidence` | any answer with a confidence | mean reported `answer["answer_confidence"]` |
| `latency_p50_ms`, `latency_p95_ms` | per request | wall time each request waited, informational -- see [batching](#batching-and-timing) |
| `cost_per_decision_p50_ms`, `cost_per_decision_p95_ms` | per decision | a call's wall time divided by the rows it carried, informational |

Add `ScoreWithin(0.25)` to the evaluator list for a tolerance metric; the default set is
`choice_accuracy`, `noul_accuracy`, `score_mae`, `mean_confidence`, plus `ece`. From the CLI the
same thing is one flag: `laya-evals run data.jsonl --score-within 0.25` reports `score_within_0.25`
beside the defaults, and the flag repeats, so `--score-within 0.25 --score-within 0.5` reports both.

A tolerance metric needs a `score` answer with a numeric label, so on a dataset without one it has
no value: `run` names the metric it could not compute instead of publishing a silent zero, and a
`--min` / `--max` gate naming that metric fails as missing. The tolerances a run was asked for are
recorded in the report's `config` block, so a reviewed baseline says which columns it expects.

## Batching and timing

`--batch-size N` scores up to N consecutive rows that share a checkpoint and a question schema in
one call. Both timing metrics come from the same measurements and answer different questions:
every row of a batch returns when the batch does, so its `latency` is the whole call, while its
`cost_per_decision` is `1/N` of it. Batching therefore *raises* `latency_*` and *lowers*
`cost_per_decision_*` on an unchanged set of decisions, and `--max latency_p50_ms=...` asks whether
requests were served fast, not whether the run was cheap. With no `--batch-size` the two agree.

`compare` ignores any `*_ms` metric unless a tolerance names it, so these never fail a baseline on
timing noise. What the harness actually did -- the batch size asked for, the runner shape it
resolved to, how many rows shared a call, and the largest chunk -- is recorded in the report's
`config.timing`, because the flag alone does not say whether anything was batched. Those counters
record the calls issued, not the calls that returned: with `laya-evals run --on-error skip`, a
chunk whose call raised still counts in `rows_grouped` and `max_chunk`, next to its entries in
`config.errored`. The default is `--on-error fail`, which re-raises instead of publishing a report
whose metrics cover only the calls that came back. The two `*_ms` metrics count only the calls that
returned, so a failed call never contributes a latency it did not measure.

### Grouping the rows inside a batch

`--sort-by-length` groups similarly sized rows into the same forward pass, so each pass pads to a
shorter maximum instead of to the longest row in it. It is the shape of the calls, not their
answers: results come back in the same order and score identically, which is why `research/` can
report 2.15x over 10,000 tickets with no decision changing.

There has to be more than one pass to reorder, so it takes effect only with a `--batch-size N`
below the number of rows the run groups. `config.timing` keeps the two claims apart:
`sort_by_length` is what the command line said, `sort_by_length_sent` is what reached the runner.
A run with no `--batch-size` asks for something that cannot happen, and says so with
`sent: false`; a runner whose `predict_batch` predates the knob is scored unsorted rather than
raising `TypeError` halfway through a long run.

### The abstention gate at a threshold

`--min-confidence T` forwards core's opt-in abstention threshold (#361) to every call the run
makes, so `Router` and `ONNXAgent` mark answers whose `answer_confidence` falls below `T` with
`low_confidence: True` before the harness sees them. Unlike grouping, this changes the answers
that score: the same run at `T=0` and `T=0.7` is a different experiment, and a `precision@coverage`
sweep is a series of these, not a single baseline drifting.

The accepted range is core's `laya.confidence.check_min_confidence` -- `[0.0, 1.0]`, finite, not
a bool -- rather than a copy here, so a value the gate itself would reject fails as a usage error
(exit 2) before any checkpoint loads. `0.0` is a legal ask: it is the control arm for a
`precision@coverage` sweep, and a check that dropped it would hide the sweep's own floor.

A runner whose `predict` or (for a batched run) whose `predict_batch` predates the gate is
**refused with a named `EvalError`**, not scored without the threshold. Silently dropping a
scoring control is the class of lie this harness exists to prevent: the report would publish a
`precision@coverage` figure for a policy that never ran. `config.timing` records both the ask and
the fact: `min_confidence` is the threshold that was requested, `min_confidence_sent` says whether
any call this run made actually carried it.

## Slices

`compare` and `run` report overall numbers and, for `--slice language|model|qid|tag`, the same
metrics per slice value, so a regression in one language or one question is visible without
reading the aggregate. The `model` slice holds the checkpoint that answered each row: the
`Router`'s own choice per request, or the runner's `model` for a runner that does not route.

### Opt-in slice gates

The overall baseline gate can pass while a smaller language or question slice regresses. To make
one reviewed slice a CI requirement, save a JSON policy such as `gates.json`:

```json
{
  "version": 1,
  "rules": [
    {"slice": {"language": "zh"}, "metric": "choice_accuracy",
     "min_count": 50, "max_drop": 0.05},
    {"slice": {"qid": "intent"}, "metric": "ece",
     "min_count": 50, "max": 0.10}
  ]
}
```

```bash
laya-evals run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 \
    --gate-policy gates.json --json report.json
laya-evals compare report.json --baseline baseline.json \
    --tolerance choice_accuracy=0.02 --gate-policy gates.json
```

Each rule selects exactly one `language`, `model`, `qid`, or `tag` value and names the metric
exactly as it appears in the slice report. It has a positive `min_count` and exactly one limit:
`min` or `max` checks the candidate value; `max_drop` permits at most that decrease from the
baseline; `max_increase` permits at most that increase. The latter two require `--baseline`.
The count is the number of scored answers for that metric in the selected slice, in **both**
reports for a relative rule. For `ece`, it is the number of answers with a finite confidence and
boolean `correct` value. A missing slice or metric, too few scored answers, or skipped/errored
cases fails the opted-in gate. Relative rules also require both reports to carry matching run
identities, so missing evidence cannot appear as a pass. A measured regression reports the slice,
metric, counts, values, and limit. Invalid policy syntax exits 2 before a checkpoint loads; a
quality failure exits 1. The policy is recorded in `config.gate_policy` of a `run --json` report.
`compare --gate-policy` applies the policy supplied on that command line to the saved measurements.
If it differs from the report's recorded policy, `compare` says so; an explicit re-check under a
new policy does not change the policy under which the original run was made.

The regular overall comparison still applies, including its tolerance and legacy-baseline
behavior. Without `--gate-policy`, slice reporting and comparison behave as before.

## Run identity

`run` records what it measured in the report's `config` block, so the artifact a reviewer reads
is reviewable on its own:

| key | meaning |
|---|---|
| `schema` | the report shape, `laya-evals-report/1`, so a consumer can refuse one it cannot read |
| `dataset` | the path as typed -- a name, not a hash |
| `dataset_sha256` | the sha256 of the dataset bytes that were parsed |
| `questions_sha256` | a fingerprint of the question schema: every question's id, type, `instructions` and `criteria`, over the whole dataset |
| `laya_version` | the `laya` that computed the numbers |
| `thresholds` | the gate this run applied: `min`, `max` and `baseline_tolerance` |
| `gate_policy` | the optional slice gate policy applied by `run --gate-policy` |
| `revisions` | the commit each checkpoint that answered was loaded from (see [below](#baseline-and-ci-gate)) |

`dataset` is a path, and a path is not an identity: a dataset can be edited in place, moved, or
refetched under the same name, and a CI cache can hand two runs the same filename and different
bytes. `questions_sha256` covers what was *asked* rather than how many rows there were, so adding
states to an unchanged question set leaves the fingerprint alone -- `dataset_sha256` still moves,
and adding a row is a change to the data, not to the question.

It covers `instructions` too, because the instruction text is the prompt. `build_sequence` renders
`"<type> question: <instructions>"` into the tokenized head, `Agent` refuses a question without one
("add the text the model should answer"), and Laya's own question identity already counts it:
`Router._question_schema` and this harness's batch grouping both key on the whole questions dict,
and `tests/test_router_batch.py` pins that rewording `instructions` alone moves a row into its own
batch group. So does a reworded instruction still compare equal to a baseline? No -- and that is
the point. "Judge whether a refund is justified" and "Be conservative and only approve explicit
refund requests" ask different questions, and the metric gate can only notice when the difference
happens to move a number further than the tolerance you named. Naming a `choice` option is the
same argument: `criteria` is the decision space, and the metamorphic checks in
`research/eval/metamorphic.py` exist because renaming a label flips answers.

Nothing about the instruction text is normalized except the one step the engine itself applies: a
non-string `instructions` is hashed as `json.dumps(ins, ensure_ascii=False)`, matching
`Agent._to_internal`. So whitespace and wording both count, and a rewording that a human considers
a copy edit is treated as a new experiment. That is the honest default -- the alternative is a
similarity heuristic standing between a run and its baseline, and no evaluation system in common
use has one.

Nothing time-bearing is recorded, so a report is still byte-reproducible for a fixed runner.

`REPORT_SCHEMA`, `questions_fingerprint(dataset)` and `file_fingerprint(path)` are public, so a
caller driving `laya.evals.evaluate` directly gets the same identity a CLI run does.

## Baseline and CI gate

- Keep the dataset, a baseline report (`--json` output you have reviewed), and the tolerances
  together, committed, so a change is a reviewable diff. `--tolerance METRIC=VALUE` is the
  maximum absolute drift allowed for that metric.
- `laya-evals run ... --baseline baseline.json --tolerance ...` exits non-zero on drift, so it
  drops into CI unchanged. `laya.evals.EvalReport.compare` and `assert_regression` expose the
  same logic for tests.

The metric gate answers "did the numbers move". It cannot answer "were these the same numbers",
because `compare` reads `overall` and only `overall` -- so a baseline recorded against one dataset
would pass a candidate scored on another, with identical arithmetic. `EvalReport.comparable_to`
closes that: it compares `schema`, `dataset_sha256` and `questions_sha256`, and `run --baseline`
and `compare` still print every delta, then fail with a non-zero exit naming the key and both
values:

```text
FAIL: baseline is not comparable: dataset_sha256 (dataset bytes): baseline is <sha>, this run is <sha>
```

A key missing on either side is *unknown*, not a conflict, so every report written before the
identity existed keeps comparing exactly as it did. That includes the scheduled gate's baseline
below, which comes from `research/eval/` and has no `config.schema` at all.

Two CI surfaces use this:

- a weight-free job in `.github/workflows/ci.yml` runs `tests/test_evals.py` and
  `tests/test_evals_api.py`, so metric math, dataset parsing and the CLI are covered on every
  PR without downloading a checkpoint;
- `.github/workflows/evals.yml` runs weekly, before a release and on demand: it evaluates the
  English checkpoint on the MASSIVE English suite and compares to
  `research/results/eval_english_51_languages.json` with the tolerances in
  `research/evals/thresholds.json`. It uploads the report as an artifact and does not block a
  PR.

The harness is deterministic for a fixed checkpoint revision, so a report is reproducible.
`run` records the dataset, model and device, plus the [timing](#batching-and-timing) facts of the
run, in the report's `config` block, and `revisions`: the commit each checkpoint that answered was
actually loaded from. `--revision <SHA>` pins that commit for every checkpoint the run loads, and
`--revision english=<SHA>` pins one checkpoint (repeatable) — which is the form an auto-routing run
wants, since the three checkpoints are three repositories and one commit cannot exist in all of
them. Left unpinned, the run takes the checkpoint's default branch and the report still says which
commit answered, so a baseline drift can be attributed to the weights or to the code.
`laya/revisions.py` publishes reviewed commit SHAs in `PINNED_REVISIONS` for callers who want to
opt in. With `--onnx`, only a bare `--revision <SHA>` applies, to the config and tokenizer download.

## Adding the real labelled set

Drop a JSONL in `research/evals/` and a reviewed baseline beside it, then point a workflow (or
`research/evals/check_regression.py`) at both. The format is the same as the fixture; nothing in
the harness knows about MASSIVE.
