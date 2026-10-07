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
to that batch API; the per-state fallback has no group to reorder. Pass `--calibration PATH`
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
| `ece` | any answer with a confidence | expected calibration error, 15 bins, computed on the column `laya.evals._answer_confidence` reads -- `answer["answer_confidence"]` where the answer carries it, and a fallback where it does not; see [which confidence a metric reads](#which-confidence-a-metric-reads) |
| `brier` | any answer with a confidence and a known label | Brier score of confidence as P(correct), `mean((confidence - correct)**2)`; lower is better |
| `aurc` | any answer with a confidence and a known label | area under the risk--coverage curve: one risk value per distinct confidence level, each weighted by the answers that level spans; lower is better, and rewards a confidence that *ranks* right from wrong rather than just being calibrated |
| `selective_accuracy@50`, `selective_accuracy@80` | any answer with a confidence and a known label | accuracy over the answers a confidence threshold at the 50% / 80% coverage point accepts -- what abstaining on the least-confident tail buys. A threshold cannot split a group of equal confidences, so this can cover more than the named fraction; see [coverage cuts](#coverage-cuts-and-ties) |
| `mean_confidence` | any answer with a confidence | mean of the same column -- `answer["answer_confidence"]` where the answer carries it |
| `latency_p50_ms`, `latency_p95_ms` | per request | wall time each request waited, informational -- see [batching](#batching-and-timing) |
| `cost_per_decision_p50_ms`, `cost_per_decision_p95_ms` | per decision | a call's wall time divided by the rows it carried, informational |

### Which confidence a metric reads

Every row above that takes a confidence (`ece`, `brier`, `aurc`, both `selective_accuracy@*`,
`mean_confidence`) gets its column from `laya.evals._answer_confidence`, which prefers
`answer["answer_confidence"]` -- the probability of the answer being reported, the quantity temperature
scaling fits. It is not a claim that the number is right as shipped: both base checkpoints are
over-confident and `laya-multilingual` ships no fitted temperatures at all -- see the README's
[Calibration](https://github.com/NandhaKishorM/laya#calibration) -- which is what `ece` measures
rather than assumes.

An answer that carries no `answer_confidence` falls through, in order, to `confidence`, then
`max(p, 1 - p)` for a `noul`, then `max(probabilities)`. Those are different quantities. On `choice`
and `score` the `confidence` field is normalized entropy, which moves with the option count (#394)
rather than with how right the answer is; `max(probabilities)` is the mass on the top option, equal to
the reported answer's probability only when the reported answer is the argmax.

The shape that actually arrives without the field is the strict Jev wire contract: `LAYA_JEV_STRICT`
drops `answer_confidence` from every answer (`laya/serve.py::_project_jev_strict`, and the flag's row
in [the HTTP API page](http-api.md)), so a report run over recorded strict responses calibrates the
entropy number. Measured on three labelled rows -- `choice` carrying `answer_confidence` 0.95 against
an entropy `confidence` of 0.7887, `score` 0.90 against 0.6410, `noul` 0.87 -- the same dataset scores
`mean_confidence` 0.9067 and `ece` 0.0900 over the full payloads and 0.7666 and 0.1707 over the strict
projection. Two reports are comparable only when their answers carry the same field;
`tests/test_evals.py` pins both paths so a change to that fallback order has to be made deliberately.

### Coverage cuts and ties

Both coverage metrics cut on a confidence **threshold**, and a threshold accepts every answer at its
own confidence. So a cut never splits a group of answers that share one: when `coverage * n` falls
inside such a group, every member of the group is accepted. The number of answers behind the figure
is therefore the group's upper edge rather than the named fraction -- `selective_accuracy@50` over a
slice whose confidences are all equal is that slice's own accuracy, not the better half of it. The
count the gate prints (`n=` in a rule's failure message) is the slice's size, not the accepted size,
so a very wide group is not visible from the message alone.

Ties are the normal case rather than a corner: a fitted temperature can leave a bucket reporting a
point mass, which `laya.common.answer_confidence` records of the shipped `choice:11+`, and a real
checkpoint produced a six-row group at exactly 1.0 out of twelve answers. Cutting at a row index
instead made both metrics depend on the order the dataset arrived in -- the same rows, shuffled,
moved `selective_accuracy@50` between 0.000 and 1.000.

`aurc` integrates one risk value per distinct level, weighted by the answers that level spans, so it
remains an area under the risk--coverage curve rather than an average of unevenly sized points.

Two consequences worth planning for:

* **A number can move in either direction, by more than a reordering could.** Where a group
  straddles the cut, the threshold reading differs from every row-index reading of the same data:
  measured over 400,001 tie-shaped datasets, up to 0.500 for `selective_accuracy@50` and 0.351
  for `aurc`. On the twelve-answer shape above -- six correct, all at confidence 1.0 -- `aurc` moves
  0.327 (0.173 to 0.500). A gate that was passing may fail, and one that was failing may pass; the
  previous verdict depended on row order, including for an absolute `min` or `max` limit, which is
  refused by nothing because it reads a single run.
* **Regenerate committed baselines.** `config.coverage_metric_definition` records which definition
  produced a report. Comparing a coverage metric across two definitions is refused on **both** the
  gates that subtract a baseline -- `--baseline --tolerance` (`EvalReport.compare`) and a relative
  rule (`max_drop` / `max_increase`) under `--gate-policy` -- and on **either** side being stale,
  not just the baseline: a candidate produced by an older `laya` carries a row-order artifact that
  can read *better* than the truth, so gating it against a correctly regenerated baseline would
  pass a regression that the correctly scored report fails. Without that refusal a
  stale baseline hides a real regression: a slice recorded at 0.033 under the old definition reads
  0.517 under this one, so a candidate that genuinely dropped 0.217 would clear a `max_drop` of
  0.05. `ece` and `brier` do not cut and stay comparable.

With no ties in the data there is one level per answer, and both metrics are exactly what they have
always been -- bit-identical, not merely close.


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
`low_confidence: True` and `abstention: "abstained"` before the harness sees them. The gate is a
*reporting* control, not a scoring one: `apply_confidence_gate` leaves
`answer["choice"] / ["noul"] / ["score"]` as the raw argmax and `_aggregate` reads only those
keys, so every accuracy, calibration and coverage number -- `ece`, `brier`, `aurc`,
`selective_accuracy@NN` -- is the same at `T=0` and `T=0.7`. What changes is the report's
`config.timing.min_confidence` and `config.timing.min_confidence_sent`, and any caller who acts
on the flag downstream of the harness.

The accepted range is core's `laya.confidence.check_min_confidence` -- `[0.0, 1.0]`, finite, not
a bool -- rather than a copy here, so a value the gate itself would reject fails as a usage error
(exit 2) before any checkpoint loads. `0.0` is a legal ask: it is the control arm for an
abstention sweep and the value `flag_low_confidence` treats as a no-op, so a check that dropped
it would hide which arm actually ran.

A runner whose `predict` or (for a batched run) whose `predict_batch` predates the gate is
**refused with a named `EvalError`**, not run without the threshold. Silently dropping it would
let `report.config["timing"]["min_confidence"]` name a threshold the harness never applied -- the
class of lie this harness exists to prevent, even though the metric numbers stay identical.
`config.timing` records both the ask and the fact: `min_confidence` is the threshold that was
requested, `min_confidence_sent` says whether any call this run made actually carried it.

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
| `coverage_metric_definition` | which definition of `aurc` / `selective_accuracy@*` produced this report (see [coverage cuts](#coverage-cuts-and-ties)). A relative gate rule on either refuses a baseline recorded under a different one, rather than subtracting numbers that do not mean the same thing |
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

## Evidence inspection

`evidence` reads — never writes — the artifacts a fine-tune/eval run already persists and says
what evidence exists, what is missing, and what is insufficient. It loads no model weights and
does not require torch.

```bash
laya-evals evidence --checkpoint ./my-checkpoint [--report report.json]
```

A checkpoint directory must contain `rl_agent_config.json`. Calibration evidence is derived from
the persisted `training.laya_train_calibration` block: a legacy checkpoint with no training
metadata is `UNKNOWN`; a question type with zero items is `MISSING`; a type whose entry records
any upstream `issues` text (not fitted, below `MIN_TYPE_N`/`CALIB_WARN_N`, clamped, unchanged
fit) is `INSUFFICIENT`; a clean fit is `PRESENT`. The helper trusts the persisted `issues`
written by #933 instead of keeping a second numeric threshold source.

The eval report's identity fields (`schema`, `dataset_sha256`, `questions_sha256`, `laya_version`)
are reported with the same semantics as `run`. The checkpoint↔report relationship is claimed only
as a deterministic conflict: if both artifacts expose the same `dataset_sha256`/`questions_sha256`
with different values, it is `INCOMPARABLE`. Everything else is `UNKNOWN` — matching `laya_version`
alone never proves a match, and differing `laya_version` alone never proves a conflict.
