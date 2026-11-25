"""Model-free checks for paired agreement and case-level uncertainty."""
from copy import deepcopy
import collections
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from laya import _agreement, confidence, evals


def run_document(run_id, rows, *, teacher=True):
    cases = []
    for index, labels in enumerate(rows):
        questions = {
            "q%d" % q: {"type": "choice", "instructions": "Choose a route.",
                        "criteria": {"a": "Accept", "b": "Defer"}}
            for q in range(len(labels))
        }
        cases.append({
            "id": "case-%d" % index,
            "state": {"text": "request-%d" % index, "context": ["original"]},
            "questions": questions,
            "answers": dict(zip(questions, labels)),
            "tags": ["routing"], "language": "en",
        })
    return {"schema": "laya-agreement-run/1", "run_id": run_id,
            "provenance": {"model": "teacher" if teacher else "student",
                           "revision": "pinned", "settings": {"temperature": 0.5}},
            "cases": cases}


def documents(reference=(("a",), ("b",)), repeat=None, student=None):
    return (run_document("reference", reference),
            run_document("repeat", repeat if repeat is not None else reference),
            run_document("student", student if student is not None else reference,
                         teacher=False))


def compare(runs, **kwargs):
    return evals.compare_agreement(*runs, **kwargs)


def test_paired_rates_use_every_decision_and_preserve_abstention_denominator():
    runs = documents((("a", "a", "a"), ("b", "a")),
                     (("a", "b", "a"), ("b", "a")),
                     (("a", "b", None), ("b", "b")))
    report = compare(runs)
    assert report["schema"] == "laya-agreement-report/1"
    overall = report["overall"]
    assert overall["n_cases"] == 2
    assert overall["n_decisions"] == 5
    assert overall["teacher_repeat_agreement"] == pytest.approx(4 / 5)
    assert overall["student_teacher_agreement"] == pytest.approx(2 / 5)
    assert overall["student_coverage"] == pytest.approx(4 / 5)
    assert overall["paired_agreement_delta"] == pytest.approx(-2 / 5)
    assert len(report["cases"]) == 5
    observed = {(row["case_id"], row["qid"]):
                (row["teacher_reference"], row["teacher_repeat"], row["student"],
                 row["accepted"], row["delta"]) for row in report["cases"]}
    assert observed == {
        ("case-0", "q0"): ("a", "a", "a", True, 0),
        ("case-0", "q1"): ("a", "b", "b", True, 0),
        ("case-0", "q2"): ("a", "a", None, False, -1),
        ("case-1", "q0"): ("b", "b", "b", True, 0),
        ("case-1", "q1"): ("a", "a", "b", True, -1),
    }
    assert report["slices"]["qid"]["q2"]["n_decisions"] == 1
    assert report["slices"]["qid"]["q2"]["student_coverage"] == 0.0
    assert not {"accuracy", "choice_accuracy", "ece", "brier", "aurc"} & overall.keys()
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("repeat,student,agreement", [("a", "a", 1.0), ("b", "b", 0.0)])
def test_perfect_and_zero_agreement(repeat, student, agreement):
    overall = compare(documents((("a",), ("a",)),
                                ((repeat,), (repeat,)),
                                ((student,), (student,))))["overall"]
    assert overall["teacher_repeat_agreement"] == agreement
    assert overall["student_teacher_agreement"] == agreement
    assert overall["paired_agreement_delta"] == 0.0
    assert tuple(overall["paired_delta_ci95"]) == (0.0, 0.0)


def test_student_abstention_is_not_a_missing_case_or_an_agreeing_answer():
    overall = compare(documents(student=((None,), (None,))))["overall"]
    assert overall["n_cases"] == overall["n_decisions"] == 2
    assert overall["student_coverage"] == 0.0
    assert overall["student_teacher_agreement"] == 0.0
    assert overall["paired_agreement_delta"] == -1.0
    assert tuple(overall["paired_delta_ci95"]) == (-1.0, -1.0)


def test_alignment_uses_case_ids_and_comparison_does_not_mutate_inputs():
    runs = documents((("a",), ("b",), ("a",)),
                     (("a",), ("a",), ("b",)),
                     (("b",), ("b",), ("a",)))
    original = deepcopy(runs)
    expected = compare(runs, seed=31)["overall"]
    assert runs == original
    runs[0]["cases"].reverse()
    runs[1]["cases"] = runs[1]["cases"][1:] + runs[1]["cases"][:1]
    assert compare(runs, seed=31)["overall"] == expected


def test_one_case_has_no_interval_even_with_many_questions():
    overall = compare(documents((("a",) * 30,)))["overall"]
    assert overall["n_cases"] == 1
    assert overall["n_decisions"] == 30
    assert overall["paired_delta_ci95"] is None


def test_bootstrap_resamples_whole_cases_not_correlated_questions():
    # There are just two independent cases. Resampling two cases can give only
    # delta -1, 0 or +1, regardless of how many questions each case contains.
    runs = documents((("a",) * 25, ("a",) * 25),
                     (("a",) * 25, ("b",) * 25),
                     (("b",) * 25, ("a",) * 25))
    report = compare(runs, bootstrap_samples=2000, seed=17)
    overall = report["overall"]
    assert overall["paired_agreement_delta"] == 0.0
    assert tuple(overall["paired_delta_ci95"]) == (-1.0, 1.0)
    assert compare(runs, bootstrap_samples=2000, seed=17) == report


def test_bootstrap_preserves_decision_weights_with_unequal_case_sizes():
    runs = documents((("a",) * 100, ("a",), ("a",), ("a",)),
                     (("b",) * 100, ("a",), ("a",), ("a",)),
                     (("a",) * 100, ("b",), ("b",), ("b",)))
    overall = compare(runs, bootstrap_samples=5000, seed=23)["overall"]
    assert overall["paired_agreement_delta"] == pytest.approx(97 / 103)
    # K large-case draws follows Binomial(4, 1/4). The 2.5th/97.5th percentiles
    # have K=0/3: ratios -4/4 and (300-1)/(300+1), not case-mean rates -1/0.5.
    assert overall["paired_delta_ci95"] == pytest.approx([-1.0, 299 / 301])


def test_the_bootstrap_batch_split_is_invisible_in_the_published_interval():
    """`paired_delta_ci95` depends on the records, `bootstrap_samples` and `seed`, and on nothing else.

    `_agreement._interval` draws its replicate indices in batches so the index matrix stays under
    131072 `int64` -- 1 MiB -- and the comment there says exactly that: "batching bounds the index
    matrix memory". It has to stay true that nothing else moves: the report publishes
    `bootstrap_samples` and `seed` as the identity of the interval, so a rewrite that reseeds per
    batch -- one line, and every other number in the report still validates -- silently republishes a
    different 95% interval under the same recorded configuration.

    A batch boundary lands on a row boundary of the same row-major stream, so the replicates are the
    ones one matrix of the whole draw would have produced, and the equality below is exact rather
    than `pytest.approx` for that reason: both sides ask the same numpy, in the same process, for the
    same stream and the same percentile method, so a numpy that changed either would move both.

    The shape is sized from the constant read out of `_interval`'s own source -- 120 cases and
    `5 * (constant // 120) + 1` samples cross the split 6 times here -- so the test cannot quietly
    stop exercising the split if the constant moves; it fails with the remedy in the message.

    What this does NOT pin is the constant's value: shrinking it (more batches, same stream) and
    drawing the matrix flat then reshaping it both leave the interval exactly where they are, and
    both were measured to do so. Only a rewrite that changes which draws land in which replicate
    moves it. Removing the batching altogether is the one equivalent-looking change this test does
    fail, deliberately: the guard belongs to the mechanism, so the message names both remedies --
    keep a memory budget, or delete this test along with the split.
    """
    cases = 120
    budget = re.search(r"(\d+) // len\(clusters\)", inspect.getsource(_agreement._interval))
    assert budget is not None, (
        "`_interval` no longer splits its bootstrap draw by a row budget, so the split this test "
        "guards is gone: either keep a memory budget here, or delete this test together with the "
        "batching -- but do not leave a claim of a guarded invariant standing on nothing.")
    constant = int(budget.group(1))
    batch = constant // cases
    samples = min(100000, 5 * batch + 1)
    assert samples > batch, ("this shape no longer crosses the batch split (%d samples against a "
                             "%d-row batch): raise `cases` so the draw still spans more than one batch"
                             % (samples, batch))
    # 1..4 questions per case, so the clusters carry unequal decision weights, and the repeat and
    # student runs disagree on a pattern that is periodic in neither the case nor the question index.
    reference = tuple(tuple("a" for _ in range(1 + i % 4)) for i in range(cases))

    def labels(modulus):
        return tuple(tuple("a" if (i + j) % modulus == 0 else "b"
                           for j in range(1 + i % 4)) for i in range(cases))

    report = compare(documents(reference, labels(4), labels(3)), bootstrap_samples=samples, seed=12)
    published = report["overall"]["paired_delta_ci95"]
    assert report["overall"]["n_cases"] == cases
    assert report["overall"]["n_decisions"] == sum(len(row) for row in reference)

    totals = {}
    for case in report["cases"]:
        cluster = totals.setdefault(case["case_id"], [0, 0])
        cluster[0] += case["delta"]
        cluster[1] += 1
    matrix = np.asarray([totals[case_id] for case_id in sorted(totals)], dtype=np.int64)
    indices = np.random.default_rng(12).integers(len(matrix), size=(samples, len(matrix)))
    sums = matrix[indices].sum(axis=1)
    one_shot = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5]).tolist()
    assert published == one_shot, ("the batched draw and one draw of the same %d replicates disagree, "
                                   "so the batching is no longer invisible: batched=%r one-shot=%r"
                                   % (samples, published, one_shot))


@pytest.mark.parametrize("side", [0, 1, 2])
@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_case_sets_must_match_exactly(side, change):
    runs = documents()
    if change == "missing":
        runs[side]["cases"].pop()
    elif change == "duplicate":
        runs[side]["cases"].append(deepcopy(runs[side]["cases"][0]))
    else:
        runs[side]["cases"][0]["id"] = "unrelated"
    with pytest.raises(ValueError):
        compare(runs)


@pytest.mark.parametrize("change", ["state", "prompt", "criteria_order", "criteria_text", "qid"])
def test_input_and_question_drift_are_rejected(change):
    runs = documents()
    case = runs[1]["cases"][0]
    question = case["questions"]["q0"]
    if change == "state":
        case["state"]["context"].append("changed")
    elif change == "prompt":
        question["instructions"] = "Choose the opposite route."
    elif change == "criteria_order":
        question["criteria"] = {"b": "Defer", "a": "Accept"}
    elif change == "criteria_text":
        question["criteria"]["a"] = "Reject"
    else:
        case["questions"]["other"] = case["questions"].pop("q0")
        case["answers"]["other"] = case["answers"].pop("q0")
    with pytest.raises(ValueError):
        compare(runs)


@pytest.mark.parametrize("key,value", [("model", "other"), ("revision", "other-pin"),
                                       ("settings", {"temperature": 0.9})])
def test_teacher_configuration_must_match(key, value):
    runs = documents()
    runs[1]["provenance"][key] = value
    with pytest.raises(ValueError):
        compare(runs)


def test_teacher_seed_may_differ():
    runs = documents()
    runs[0]["provenance"]["seed"] = 5
    runs[1]["provenance"]["seed"] = 8
    assert compare(runs)["overall"]["teacher_repeat_agreement"] == 1.0


def test_unknown_teacher_revision_remains_unknown_and_disclosed():
    runs = documents()
    for run in runs[:2]:
        run["provenance"]["revision"] = None
    report = compare(runs)
    assert report["config"]["provenance"]["reference"]["revision"] is None
    assert report["config"]["provenance"]["repeat"]["revision"] is None
    assert any("revision" in note.lower() for note in report["limitations"])


@pytest.mark.parametrize("name,value", [("bootstrap_samples", 99), ("bootstrap_samples", 100001),
    ("bootstrap_samples", True), ("bootstrap_samples", 100.5), ("bootstrap_samples", float("inf")),
    ("seed", -1), ("seed", True), ("seed", 1.5), ("seed", float("nan"))])
def test_bootstrap_options_require_bounded_integer_values(name, value):
    with pytest.raises(ValueError):
        compare(documents(), **{name: value})


@pytest.mark.parametrize("change", ["schema", "empty", "score", "teacher_abstention",
                                  "unknown_label", "missing_answer", "extra_answer", "nonfinite"])
def test_malformed_runs_are_rejected(change):
    runs = documents()
    case = runs[0]["cases"][0]
    if change == "schema":
        runs[0]["schema"] = "laya-agreement-run/999"
    elif change == "empty":
        for run in runs:
            run["cases"] = []
    elif change == "score":
        case["questions"]["q0"]["type"] = "score"
    elif change == "teacher_abstention":
        case["answers"]["q0"] = None
    elif change == "unknown_label":
        case["answers"]["q0"] = "unlisted"
    elif change == "missing_answer":
        case["answers"].pop("q0")
    elif change == "extra_answer":
        case["answers"]["other"] = "a"
    else:
        for run in runs:
            run["provenance"]["settings"]["temperature"] = float("nan")
    with pytest.raises(ValueError):
        compare(runs)


def _deep(value, dict_factory, list_factory):
    """Rebuild every mapping as `dict_factory(...)` and every list as `list_factory(...)`.

    Scalars are left alone: these arms are about the container spellings a caller reaches for
    when it wants a JSON object whose key order it controls, not about scalar subclasses.
    """
    if isinstance(value, dict):
        return dict_factory({k: _deep(v, dict_factory, list_factory) for k, v in value.items()})
    if isinstance(value, list):
        return list_factory([_deep(v, dict_factory, list_factory) for v in value])
    return value


class SubDict(dict):
    pass


class SubList(list):
    pass


@pytest.mark.parametrize("flavour", ["ordered", "pairs_hook", "defaultdict", "counter",
                                     "subdict", "sublist"])
def test_container_subclasses_encode_as_the_mapping_they_wrap(flavour):
    """`_json` matched containers by exact type, so it refused payloads `json.dumps` encodes
    byte-for-byte as the plain mapping. `object_pairs_hook=OrderedDict` is how a caller reads a
    saved run *and* keeps the key order this module fingerprints, and `Counter`/`defaultdict` are
    what a caller builds criteria and settings from; all of them reached
    "unsupported value of type OrderedDict" while every `isinstance(..., dict)` guard in `_run`,
    `_provenance` and `_labels` had already accepted the same object."""
    expected = compare(documents())
    runs = documents()
    if flavour == "ordered":
        runs = [_deep(run, collections.OrderedDict, list) for run in runs]
    elif flavour == "pairs_hook":
        runs = [json.loads(json.dumps(run), object_pairs_hook=collections.OrderedDict)
                for run in runs]
    elif flavour == "defaultdict":
        runs = [_deep(run, lambda d: collections.defaultdict(dict, d), list) for run in runs]
    elif flavour == "counter":
        for run in runs:
            for case in run["cases"]:
                for question in case["questions"].values():
                    question["criteria"] = collections.Counter(question["criteria"])
    elif flavour == "subdict":
        runs = [_deep(run, SubDict, list) for run in runs]
    else:
        runs = [_deep(run, dict, SubList) for run in runs]
    report = compare(runs)
    assert report == expected
    assert report["config"]["input_fingerprints"] == expected["config"]["input_fingerprints"]


@pytest.mark.parametrize("value,typed", [(("x", "y"), "tuple"), ({"x", "y"}, "set"),
                                         (np.float64(0.5), "float64"), (np.int64(7), "int64"),
                                         (np.str_("teacher"), "str_")])
def test_scalars_stay_matched_by_exact_type(value, typed):
    """The other half of the rule: a tuple is an array to `json.dumps` and a set is not
    encodable, so exactness there protects the encoding; a scalar subclass would silently
    substitute a type the writer did not declare (`np.int64` and `np.bool_` do not encode at
    all). Containers are the only match relaxed."""
    runs = documents()
    runs[0]["provenance"]["settings"]["temperature"] = value
    with pytest.raises(evals.EvalError,
                       match="must be strict JSON: unsupported value of type %s" % typed):
        compare(runs)


def test_a_non_string_key_inside_a_subclass_is_still_refused():
    runs = documents()
    runs[0]["provenance"]["settings"] = collections.OrderedDict([(1, 0.5)])
    with pytest.raises(evals.EvalError, match="must be strict JSON: object keys must be strings"):
        compare(runs)


def test_ordered_spellings_do_not_blind_the_order_check():
    # `_align` compares the *encoding*, so a criteria mapping spelled as an OrderedDict in both
    # runs must still fail when the two runs disagree on option order.
    runs = [_deep(run, collections.OrderedDict, list) for run in documents()]
    question = runs[1]["cases"][0]["questions"]["q0"]
    question["criteria"] = collections.OrderedDict(reversed(list(question["criteria"].items())))
    with pytest.raises(evals.EvalError, match="including option order"):
        compare(runs)


_DEFAULT_RESULT = object()


class RecordingRunner:
    def __init__(self, result=_DEFAULT_RESULT, mutate=False):
        self.calls = []
        self.result = result
        self.mutate = mutate

    def predict(self, state, questions, model=None):
        self.calls.append((deepcopy(state), deepcopy(questions), model))
        answers = {qid: {"type": "choice", "choice": "a"} for qid in questions}
        if self.mutate:
            state["context"].append("runner mutation")
            questions["q0"]["criteria"]["a"] = "runner mutation"
        return {"answers": answers} if self.result is _DEFAULT_RESULT else self.result


def evaluate(runner, runs, **kwargs):
    return evals.evaluate_agreement(runner, runs[0], runs[1],
                                   student_provenance=runs[2]["provenance"], **kwargs)


def test_evaluate_forwards_model_and_isolates_inputs_from_runner_mutation():
    runs = documents()
    original = deepcopy(runs)
    runner = RecordingRunner(mutate=True)
    report = evaluate(runner, runs, model="local-checkpoint")
    assert runs == original
    assert len(runner.calls) == 2
    assert all(call[2] == "local-checkpoint" for call in runner.calls)
    assert report["overall"]["student_teacher_agreement"] == 0.5


def test_evaluate_treats_abstained_argmax_as_uncovered():
    runner = RecordingRunner({"answers": {"q0": {"type": "choice", "choice": "a",
                                               "abstention": confidence.GATE_ABSTAINED}}})
    overall = evaluate(runner, documents())["overall"]
    assert overall["student_coverage"] == overall["student_teacher_agreement"] == 0.0


def test_the_gate_writes_the_marker_and_the_report_reads_it():
    """Abstention is `laya.confidence`'s vocabulary, so this result comes from that gate.

    The fixture above types the state by hand, which pins nothing about the field a gated runner
    actually returns: renaming the state, renaming the key, or reading the key as "any marker
    present" all leave a hand-typed test green while the report counts a declined answer as
    covered. Here the gate produces all three states and the report has to single out one of them.
    """
    answers = {
        "q0": {"type": "choice", "choice": "a", "answer_confidence": 0.9},   # clears the gate
        "q1": {"type": "choice", "choice": "a", "answer_confidence": 0.2},   # falls below it
        "q2": {"type": "choice", "choice": "a"},                             # nothing to compare
    }
    confidence.apply_confidence_gate([{"answers": answers}], min_confidence=0.5)
    assert [answers[qid]["abstention"] for qid in ("q0", "q1", "q2")] == list(confidence.GATE_STATES)
    declined = [qid for qid in answers if answers[qid]["abstention"] == confidence.GATE_ABSTAINED]
    assert declined == ["q1"]

    report = evaluate(RecordingRunner({"answers": answers}), documents((("a", "a", "a"),)))
    overall = report["overall"]
    assert overall["n_decisions"] == 3
    assert overall["student_coverage"] == pytest.approx(2 / 3)
    assert overall["student_teacher_agreement"] == pytest.approx(2 / 3)
    assert overall["teacher_repeat_agreement"] == pytest.approx(1.0)
    assert overall["paired_agreement_delta"] == pytest.approx(-1 / 3)
    assert [(case["qid"], case["student"], case["accepted"]) for case in report["cases"]] == [
        ("q0", "a", True), ("q1", None, False), ("q2", "a", True),
    ]


@pytest.mark.parametrize("result", [None, "invalid", {}, {"answers": {}},
    {"answers": {"q0": {"type": "choice", "choice": "unlisted"}}},
    {"answers": {"q0": {"type": "score", "score": 0.4}}},
    {"answers": {"q0": {"type": "choice", "choice": None}}}])
def test_evaluate_rejects_invalid_runner_results(result):
    with pytest.raises(ValueError):
        evaluate(RecordingRunner(result), documents())


@pytest.mark.parametrize("invalid", ["alignment", "bootstrap", "provenance", "null_state"])
def test_evaluate_validates_every_input_before_first_runner_call(invalid):
    runs = documents()
    kwargs = {}
    if invalid == "alignment":
        runs[1]["cases"][-1]["state"] = "changed final case"
    elif invalid == "bootstrap":
        kwargs["bootstrap_samples"] = True
    elif invalid == "null_state":
        for run in runs:
            run["cases"][-1]["state"] = None
    else:
        runs[2]["provenance"]["settings"] = {"temperature": float("inf")}
    runner = RecordingRunner()
    with pytest.raises(ValueError):
        evaluate(runner, runs, **kwargs)
    assert runner.calls == []


def cli(*args):
    return subprocess.run([sys.executable, "-m", "laya.evals_cli", "agreement", *map(str, args)],
                          cwd=Path(__file__).resolve().parents[1], capture_output=True,
                          text=True, encoding="utf-8", timeout=30)


def saved_runs(tmp_path):
    runs = documents()
    for run in runs:
        run["cases"][0]["state"] = "退款申请"
    paths = [tmp_path / (run["run_id"] + ".json") for run in runs]
    for path, run in zip(paths, runs):
        path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    return paths, runs


def test_cli_saved_run_replay_matches_public_api(tmp_path):
    paths, runs = saved_runs(tmp_path)
    expected = compare(runs, bootstrap_samples=100, seed=12)
    args = [*paths, "--bootstrap-samples", "100", "--seed", "12"]
    result = cli(*args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    output = tmp_path / "结果.json"
    result = cli(*args, "--json", output)
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_cli_reads_bom_prefixed_saved_runs(tmp_path):
    """A run re-saved through an editor or a spreadsheet export is the same run with three bytes in front.

    The three agreement reads opened `utf-8`, so a UTF-8 BOM stayed as U+FEFF and `json.load` refused
    a document that is valid JSON, answering `cannot read agreement run ... Unexpected UTF-8 BOM
    (decode using utf-8-sig)` -- a refusal naming the decoding this CLI could have asked for itself.
    `laya/cli.py`, `laya/train.py` and `laya/evidence.py` open a caller's file with `utf-8-sig` for
    exactly this reason, and the byte-for-byte equality below is the point: the BOM changes nothing
    about the numbers, so it must change nothing about the verdict.
    """
    paths, runs = saved_runs(tmp_path)
    for path in paths:
        path.write_text("﻿" + path.read_text(encoding="utf-8"), encoding="utf-8")
    expected = compare(runs, bootstrap_samples=100, seed=12)
    result = cli(*paths, "--bootstrap-samples", "100", "--seed", "12")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("invalid", ["duplicate_key", "nan", "malformed", "nested", "missing", "misaligned"])
def test_cli_refuses_bad_inputs_without_a_report(tmp_path, invalid):
    paths, runs = saved_runs(tmp_path)
    if invalid == "duplicate_key":
        # Reject ambiguity even when the duplicate has the same value.
        raw = paths[0].read_text(encoding="utf-8")
        paths[0].write_text(raw.replace('"schema":', '"schema":"laya-agreement-run/1","schema":', 1),
                            encoding="utf-8")
    elif invalid == "nan":
        runs[0]["provenance"]["settings"]["temperature"] = float("nan")
        paths[0].write_text(json.dumps(runs[0]), encoding="utf-8")
    elif invalid == "malformed":
        paths[0].write_text("{broken", encoding="utf-8")
    elif invalid == "nested":
        paths[0].write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
    elif invalid == "missing":
        paths[0].unlink()
    else:
        runs[2]["cases"].pop()
        paths[2].write_text(json.dumps(runs[2]), encoding="utf-8")
    output = tmp_path / "report.json"
    result = cli(*paths, "--json", output)
    assert result.returncode == 2
    assert "laya-evals:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
