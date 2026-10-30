"""Offline tests for the layered evaluation harness.

No checkpoint and no network: every function under test is pure, so this runs in
CI next to the other suites. The model-dependent paths (`score_cases`,
`run_language`) are exercised by `tests/test_local_e2e.py` when a checkpoint is
present.

Run: python research/eval/test_laya_eval.py
"""
import gzip
import os
import random
import sys
import tempfile
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.eval.laya_eval import (  # noqa: E402
    DATASET, ECE_BINS, INSTRUCTIONS, N_OPTS, SEED, available_languages, build_case, build_suite,
    ece, input_fingerprint, load_language, macro_f1, render_label, summarise,
    temperature_for, validate_release_report,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


# --------------------------------------------------------------- label rendering
check("render/underscores become spaces", render_label("alarm_set"), "alarm set")
check("render/dots become colon-space", render_label("news.»"), "news: »".replace("»", "»"))
check("render/query_definition", render_label("qa_definition"), "qa definition")
check("render/no underscores left", "_" in render_label("a_b_c"), False)
# must match bench_local.py:176 exactly
check("render/upstream parity", render_label("datetime_query"), "datetime query")


# ------------------------------------------------------------------ one case
_rng = random.Random(SEED)
_state, _q, _gold_idx = build_case("wake me up at five am", "alarm_set",
                                   ["a", "b", "c"], _rng, 4)
check("case/state key is utterance", list(_state), ["utterance"])
check("case/single question id", list(_q), ["intent"])
check("case/type is choice", _q["intent"]["type"], "choice")
check("case/instructions fixed", _q["intent"]["instructions"], INSTRUCTIONS)
check("case/option count", len(_q["intent"]["criteria"]), 4)
check("case/gold is present", "alarm_set" in _q["intent"]["criteria"], True)
check("case/gold index points at gold",
      list(_q["intent"]["criteria"])[_gold_idx], "alarm_set")


# ------------------------------------------------- determinism and seed parity
ROWS = [{"text": "t%d" % i, "label_text": "lab%d" % (i % 7)} for i in range(30)]
LABELS = sorted({r["label_text"] for r in ROWS})

a, b, c = build_suite(ROWS, LABELS, 10, 5, SEED)
d, e, f = build_suite(ROWS, LABELS, 10, 5, SEED)
check("suite/same seed same cases", a == d, True)
check("suite/same seed same gold", b == e, True)
check("suite/same seed same options", c == f, True)

g, h, i = build_suite(ROWS, LABELS, 10, 5, SEED + 1)
check_true("suite/different seed different options", c != i)

# a fresh Random(SEED) per language is what upstream does; two languages with the
# same row content must therefore produce identical option sets
ROWS_B = [dict(r) for r in ROWS]
_b1 = build_suite(ROWS, LABELS, 10, 5, SEED)[2]
_b2 = build_suite(ROWS_B, LABELS, 10, 5, SEED)[2]
check("suite/rng reset per language", _b1, _b2)

check("suite/per_lang caps cases", len(a), 10)
check("suite/gold within options",
      all(0 <= gi < len(opts) for gi, opts in zip(b, c)), True)
check("suite/every gold is in its option set",
      all(opts[gi] == ROWS[i]["label_text"] for i, (gi, opts) in enumerate(zip(b, c))), True)
check("suite/options are distinct",
      all(len(set(o)) == len(o) for o in c), True)
check("suite/n_opts respected", all(len(o) == 5 for o in c), True)

# more options requested than labels available: must not crash or duplicate
tiny = [{"text": "x", "label_text": "only"}]
_tc, _tg, _to = build_suite(tiny, ["only"], 1, 20, SEED)
check("suite/fewer labels than n_opts", len(_to[0]), 1)


# --------------------------------------------------------------------- metrics
import math  # noqa: E402

check_true("ece/empty is nan", math.isnan(ece([], [])))
check("ece/perfect on one bin", round(ece([0.95] * 50, [1.0] * 50), 4), 0.05)
check("ece/all wrong and confident", round(ece([0.95] * 50, [0.0] * 50), 4), 0.95)
check("ece/normalised by count", round(ece([0.9] * 100, [1.0] * 100), 4),
      round(ece([0.9] * 1000, [1.0] * 1000), 4))
check("ece/one sample", round(ece([1.0], [1.0]), 4), 0.0)

# Bin boundary: the first bin is closed at the bottom, so conf == 0.0 IS counted. This
# harness tested `conf > lo` for every bin until the divergence was found, which made it
# the only one of the four implementations that binned differently --
# `laya.common.ece_score`, `research/scripts/bench_local.py` and
# `research/scripts/build_benchmark_nb.py` all settled on this boundary in #39.
check("ece/conf==0.0 is binned (matches the other three)",
      round(ece([0.0, 0.0], [1.0, 1.0]), 4), 1.0)
check("ece/conf==0.0 carries its bin weight",
      round(ece([0.0, 1.0], [1.0, 1.0]), 4), 0.5)
check("ece/conf==0.0 and correct costs nothing",
      round(ece([0.0, 0.0], [0.0, 0.0]), 4), 0.0)
check_true("ece/conf slightly above 0 IS binned",
           ece([1e-9, 1e-9], [1.0, 1.0]) > 0.0)

check_true("f1/empty is nan", math.isnan(macro_f1([], [])))
check("f1/perfect", macro_f1([0, 1, 2], [0, 1, 2]), 1.0)
check("f1/all wrong", round(macro_f1([0, 1], [1, 0]), 4), 0.0)
check("f1/disjoint labels", round(macro_f1([0], [1]), 4), 0.0)

_s = summarise([0.9, 0.8, 0.7, 0.6], [1.0, 1.0, 0.0, 0.0], [0, 1, 2, 3], [0, 1, 3, 2])
check("summary/n", _s["n"], 4)
check("summary/accuracy", _s["accuracy"], 0.5)
check("summary/mean_confidence", _s["mean_confidence"], 0.75)
check("summary/acc_at_50_coverage takes the confident half",
      _s["acc_at_50_coverage"], 1.0)
check("summary/empty", summarise([], [], [], []), {"n": 0})


# ------------------------------------------------------- temperature selection
class _FakeAgent:
    temperature_by_options = {"choice:11+": 0.5}
    temperature = [1.0, 1.0, 1.0]
    temperature_by_options_raw = {"choice:11+": 0.10058280825614929}
    temperature_raw = [1.0, 1.0, 1.0]


_fa = _FakeAgent()
from laya.common import QTYPES  # noqa: E402

check("temp/clamped path is used by default",
      temperature_for(_fa, QTYPES["choice"], 20), 0.5)
check("temp/raw path under unclamped",
      round(temperature_for(_fa, QTYPES["choice"], 20, unclamped=True), 6), 0.100583)
check("temp/falls back to the per-type list",
      temperature_for(_fa, QTYPES["choice"], 3), 1.0)
check("temp/raw falls back too",
      temperature_for(_fa, QTYPES["choice"], 3, unclamped=True), 1.0)


# ------------------------------------------------------------------- constants
check("const/seed matches upstream", SEED, 13)
check("const/n_opts matches upstream", N_OPTS, 20)
check("const/bins", ECE_BINS, 15)
check("const/instructions match bench_local.py",
      INSTRUCTIONS, "What is the user asking for in `utterance`?")


# ------------------------------------------- the #208 before/after re-run file
# research/results/cpu_51_language_sweep_clamped.json records the committed sweep,
# the pre-clamp re-run and the served-temperature re-run side by side. It is only
# useful if it still agrees with the committed file, so that agreement is a test.
import json  # noqa: E402

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "research", "results")


def _load(name):
    with open(os.path.join(_RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


_rerun = _load("cpu_51_language_sweep_clamped.json")
_sweep = _load("cpu_51_language_sweep.json")["part_a"]["by_model"]["english"]
_langs = _sweep["per_language"]

# Both files round each per-language figure to 4 decimals (bench_local.py:138-144), so
# agreement has to be judged at that resolution: two files can disagree by 1 in the
# last stored digit for reasons that have nothing to do with the temperatures, and
# three languages do (am, el, ro) because the committed run used torch 2.8.0 and this
# one 2.14.0. The macro figures match exactly, which is why that is the headline.
_QUANT = 1.5e-4


def _close(a, b, tol=_QUANT):
    return abs(a - b) <= tol

check("208/one entry per language", len(_rerun["per_language"]), len(_langs))
check("208/case count is languages x per_lang",
      _rerun["config"]["n_cases"],
      _rerun["config"]["languages"] * _rerun["config"]["per_lang"])
check("208/macro accuracy copied from the committed sweep",
      _rerun["macro"]["committed"]["accuracy"], _sweep["macro_accuracy"])
check("208/macro ece copied from the committed sweep",
      _rerun["macro"]["committed"]["ece"], _sweep["macro_ece"])

# accuracy is argmax of a temperature-scaled softmax, so it cannot move with T
check_true("208/accuracy identical in all three regimes",
           all(len({_rerun["per_language"][lg][r]["accuracy"] for r in
                    ("committed", "unclamped_rerun", "clamped_rerun")}) == 1
               for lg in _langs))
check_true("208/macro_f1 identical in all three regimes",
           all(len({_rerun["per_language"][lg][r]["macro_f1"] for r in
                    ("committed", "unclamped_rerun", "clamped_rerun")}) == 1
               for lg in _langs))

# the committed calibration columns must match the unclamped re-run, and only the
# clamped re-run may differ -- that is the whole claim
check_true("208/ece matches the committed file in the unclamped re-run",
           all(_close(_rerun["per_language"][lg]["unclamped_rerun"]["ece"],
                      _langs[lg]["ece"]) for lg in _langs))
check_true("208/ece differs from the committed file in the clamped re-run",
           all(not _close(_rerun["per_language"][lg]["clamped_rerun"]["ece"],
                          _langs[lg]["ece"]) for lg in _langs))
check_true("208/mean_confidence matches in the unclamped re-run",
           all(_close(_rerun["per_language"][lg]["unclamped_rerun"]["mean_confidence"],
                      _langs[lg]["mean_confidence"]) for lg in _langs))
check_true("208/mean_confidence differs in the clamped re-run",
           all(not _close(_rerun["per_language"][lg]["clamped_rerun"]["mean_confidence"],
                          _langs[lg]["mean_confidence"]) for lg in _langs))

# the unclamped re-run is a reproduction, not an approximation: name the tolerance
# so a future change that widens it has to say so
check_true("208/unclamped reproduction agrees in at least 48 of 51 languages",
           sum(1 for lg in _langs
               if _close(_rerun["per_language"][lg]["unclamped_rerun"]["ece"], _langs[lg]["ece"])) >= 48)

# the clamp can lower ECE everywhere without lowering rank quality, so this is a
# guard against the misleading "the clamp makes it worse" reading
check_true("208/the clamp lowers macro ece",
           _rerun["macro"]["clamped_rerun"]["macro_ece"]
           < _rerun["macro"]["unclamped_rerun"]["macro_ece"])
check_true("208/every per-language delta is reported",
           all("delta_ece" in v and "delta_mean_confidence" in v
               for v in _rerun["per_language"].values()))
check("208/only choice:11+ is the clamped bucket",
      _rerun["temperature_choice_11_plus"]["unclamped_rerun"], 0.10058280825614929)
check("208/the served bucket is 0.5",
      _rerun["temperature_choice_11_plus"]["clamped_rerun"], 0.5)


# --------------------------------------------------------- release-run evidence
_model_sha, _dataset_sha, _source_sha = "a" * 40, "b" * 40, "c" * 40
_record = {
    "lang": "en", "index": 0, "state": {"utterance": "book a flight"},
    "instructions": INSTRUCTIONS, "options": ["book_flight", "cancel"],
    "option_texts": ["book flight", "cancel"], "gold_index": 0,
    "gold_label": "book_flight", "pred_index": 0, "pred_label": "book_flight",
    "correct": 1, "confidence": 0.9, "probability": 0.9, "p_gold": 0.9,
    "temperature": 1.0,
}
_release = {
    "config": {
        "requested_model_revision": _model_sha, "model_revision": _model_sha,
        "dataset_revision": _dataset_sha, "source_revision": _source_sha,
        "source_dirty": False, "per_lang": 1, "n_opts": 2,
        "environment": {"python": "3.11", "torch": "2", "transformers": "5", "datasets": "5"},
    },
    "report": {"en": {"n": 1, "accuracy": 1.0, "macro_f1": 1.0,
                      "ece": 0.1, "mean_confidence": 0.9,
                      "acc_at_50_coverage": 1.0, "temperature": 1.0,
                      "input_sha256": input_fingerprint([_record])}},
    "cases": [_record],
}
check("release/complete report accepted", validate_release_report(_release, ["en"]), [])
_changed = deepcopy(_record)
_changed["pred_index"] = 1
check("release/predictions do not define input identity",
      input_fingerprint([_changed]), input_fingerprint([_record]))
_changed["options"] = list(reversed(_changed["options"]))
check_true("release/option order defines input identity",
           input_fingerprint([_changed]) != input_fingerprint([_record]))
_changed = deepcopy(_release)
_changed["config"]["dataset_revision"] = None
check_true("release/missing dataset revision rejected",
           any("dataset_revision" in e for e in validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["config"]["model_revision"] = "d" * 40
check_true("release/resolved model revision mismatch rejected",
           any("differs" in e for e in validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["config"]["source_dirty"] = True
check_true("release/dirty checkout rejected", bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["cases"] = []
check_true("release/missing case rejected", bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["cases"][0]["state"]["utterance"] = "cancel the flight"
check_true("release/changed input rejected", bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["report"]["en"] = {"error": "dataset unavailable"}
check_true("release/failed language rejected", bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["report"]["en"]["accuracy"] = 0.0
check_true("release/incorrect accuracy rejected", bool(validate_release_report(_changed, ["en"])))

for _metric in ("accuracy", "macro_f1", "ece", "mean_confidence", "acc_at_50_coverage"):
    _changed = deepcopy(_release)
    _changed["report"]["en"][_metric] -= 0.0001
    check_true("release/%s last-digit drift rejected" % _metric,
               any(_metric in e for e in validate_release_report(_changed, ["en"])))
    _changed = deepcopy(_release)
    _changed["report"]["en"].pop(_metric)
    check_true("release/missing %s rejected" % _metric,
               any(_metric in e for e in validate_release_report(_changed, ["en"])))

for _field in _record:
    _changed = deepcopy(_release)
    _changed["cases"][0].pop(_field)
    check_true("release/missing case %s rejected without raising" % _field,
               bool(validate_release_report(_changed, ["en"])))

for _field, _value in (
        ("confidence", float("nan")), ("probability", float("inf")),
        ("p_gold", -0.1), ("confidence", 1.1), ("confidence", True),
        ("temperature", 0.0), ("temperature", float("-inf")),
        ("gold_index", True), ("pred_index", -1), ("index", False),
        ("correct", True), ("pred_label", "unrelated"),
        ("options", ["cancel", "cancel"]), ("option_texts", ["book flight", None]),
        ("instructions", None), ("probability", 0.8), ("p_gold", 0.95),
        ("p_gold", 0.8)):
    _changed = deepcopy(_release)
    _changed["cases"][0][_field] = _value
    check_true("release/invalid case %s=%r rejected" % (_field, _value),
               bool(validate_release_report(_changed, ["en"])))

_changed = deepcopy(_release)
_changed["cases"][0].update(confidence=0.4, probability=0.4, p_gold=0.4)
check_true("release/top probability below uniform rejected", bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["cases"][0].update(pred_index=1, pred_label="cancel", correct=0,
                            confidence=0.5, probability=0.5, p_gold=0.5)
check_true("release/argmax cannot skip an earlier tied gold",
           any("probabilities disagree" in error for error in validate_release_report(_changed, ["en"])))
_changed["cases"][0].update(confidence=0.9, probability=0.9, p_gold=0.5)
check_true("release/distinct gold and chosen probabilities cannot sum above one",
           bool(validate_release_report(_changed, ["en"])))
_changed = deepcopy(_release)
_changed["cases"][0].update(gold_index=1, gold_label="cancel", correct=0,
                            confidence=0.5, probability=0.5, p_gold=0.5)
_changed["report"]["en"].update(accuracy=0.0, macro_f1=0.0, ece=0.5, mean_confidence=0.5,
                              acc_at_50_coverage=0.0, input_sha256=input_fingerprint(_changed["cases"]))
check("release/argmax may choose the first option before a tied gold",
      validate_release_report(_changed, ["en"]), [])
for _n_opts, _chosen, _gold_probability, _valid in ((2, 0.6, 0.3, False),
                                                  (3, 0.4, 0.01, False), (3, 0.4, 0.2, True)):
    _changed = deepcopy(_release)
    _changed["config"]["n_opts"] = _n_opts
    _options = ["book_flight", "cancel", "other"][:_n_opts]
    _changed["cases"][0].update(gold_index=1, gold_label="cancel", correct=0,
                                options=_options, option_texts=[render_label(key) for key in _options],
                                confidence=_chosen, probability=_chosen, p_gold=_gold_probability)
    _changed["report"]["en"].update(accuracy=0.0, macro_f1=0.0, ece=_chosen, mean_confidence=_chosen,
                                  acc_at_50_coverage=0.0, input_sha256=input_fingerprint(_changed["cases"]))
    _errors = validate_release_report(_changed, ["en"])
    check_true("release/%d options can contain the remaining probability mass=%s" % (_n_opts, _valid),
               not _errors if _valid else any("probabilities disagree" in error for error in _errors))
for _field, _value in (("ece", float("nan")), ("mean_confidence", float("inf")),
                       ("accuracy", True), ("temperature", -1.0)):
    _changed = deepcopy(_release)
    _changed["report"]["en"][_field] = _value
    check_true("release/invalid report %s rejected" % _field,
               bool(validate_release_report(_changed, ["en"])))
for _field, _value in (("config", None), ("report", []), ("cases", [None])):
    _changed = deepcopy(_release)
    _changed[_field] = _value
    check_true("release/invalid %s container rejected without raising" % _field,
               bool(validate_release_report(_changed, ["en"])))

check("release/committed 100-case report reconstructs all metrics",
      validate_release_report(_load("massive_en_cpu_release_20261001.json"), ["en"]), [])

_dataset_calls = []
_fake_datasets = SimpleNamespace(
    load_dataset=lambda *args, **kw: (_dataset_calls.append((args, kw)) or
                                     [{"text": "hello", "label_text": "greet"}]),
    get_dataset_config_names=lambda *args, **kw: (_dataset_calls.append((args, kw)) or
                                                  ["default", "en"]),
)
with patch.dict(sys.modules, {"datasets": _fake_datasets}):
    with tempfile.TemporaryDirectory() as _tmp:
        _archive = os.path.join(_tmp, "en.json.gz")
        _source_rows = [{"text": "¿hola?", "label_text": "greet", "ignored": 1},
                        {"text": "book a flight", "label_text": "book_flight", "ignored": 2}]
        with gzip.open(_archive, "wt", encoding="utf-8") as _fh:
            for _row in _source_rows:
                _fh.write(json.dumps(_row, ensure_ascii=False) + "\n")
        with patch("huggingface_hub.hf_hub_download", return_value=_archive) as _download:
            check("release/pinned dataset rows preserve order and project only model fields",
                  load_language("en", revision=_dataset_sha),
                  [{"text": row["text"], "label_text": row["label_text"]} for row in _source_rows])
        check("release/pinned dataset file has immutable Hub identity", _download.call_args.kwargs,
              {"repo_id": DATASET, "filename": "test/en.json.gz",
               "repo_type": "dataset", "revision": _dataset_sha})
    check("release/pinned language list", available_languages(_dataset_sha), ["en"])
    check("release/language list receives the dataset pin", _dataset_calls,
          [((DATASET,), {"revision": _dataset_sha})])
    with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("requested revision unavailable")):
        try:
            load_language("en", revision=_dataset_sha)
        except RuntimeError as _exc:
            check_true("release/unavailable pin raises a readable load error",
                       "could not load" in str(_exc) and "requested revision unavailable" in str(_exc))
        else:
            check_true("release/unavailable pin raises a readable load error", False)
    check("release/unavailable pin never falls back to datasets", len(_dataset_calls), 1)
    check("release/unpinned dataset rows use the existing loader", load_language("en"),
          [{"text": "hello", "label_text": "greet"}])
    check("release/unpinned dataset loader receives no revision", _dataset_calls[-1],
          ((DATASET, "en"), {"split": "test"}))

import laya  # noqa: E402
from research.eval import laya_eval as harness  # noqa: E402

# Two adjacent confidences can round into one ECE bin or one coverage tie. Exercise
# the report writer, not just the validator, so the evidence keeps the precision it scored.
import numpy as np  # noqa: E402

_precision_rows = [{"text": "first", "label_text": "book_flight"},
                   {"text": "second", "label_text": "cancel"}]
_precision_gold = build_suite(_precision_rows, ["book_flight", "cancel"], 2, 2)[1]
for _name, _confidences, _corrects in (
        ("ECE bin boundary", [0.66666664, 0.66666669], [1, 0]),
        ("near-tied coverage cutoff", [0.70000003, 0.70000004], [0, 1])):
    _probabilities = []
    for _gold, _confidence, _correct in zip(_precision_gold, _confidences, _corrects):
        _pred = _gold if _correct else 1 - _gold
        _probability = np.full(2, 1.0 - _confidence)
        _probability[_pred] = _confidence
        _probabilities.append(_probability)
    with patch.object(harness, "load_language", return_value=_precision_rows), \
            patch.object(harness, "score_cases", return_value=[np.zeros(2), np.zeros(2)]), \
            patch.object(harness, "temperature_for", return_value=1.0), \
            patch.object(harness, "softmax_t", side_effect=_probabilities):
        _precision = harness.run_language(None, "en", 2, 2)
    check("release/%s keeps scored confidences" % _name,
          [row["confidence"] for row in _precision["cases"]], _confidences)
    _precision_payload = deepcopy(_release)
    _precision_payload["config"]["per_lang"] = 2
    _precision_payload["report"]["en"] = _precision["report"]
    _precision_payload["cases"] = _precision["cases"]
    check("release/%s metrics reconstruct exactly" % _name,
          validate_release_report(_precision_payload, ["en"]), [])
    _metric = "ece" if _name == "ECE bin boundary" else "acc_at_50_coverage"
    _rounded = summarise([round(value, 6) for value in _confidences], _corrects,
                         _precision_gold, [row["pred_index"] for row in _precision["cases"]])
    check_true("release/%s is lost at six decimals" % _name,
               _rounded[_metric] != _precision["report"][_metric])

with patch.object(harness, "load_language", return_value=_precision_rows), \
        patch.object(harness, "score_cases", return_value=[np.zeros(2), np.zeros(2)]), \
        patch.object(harness, "temperature_for", return_value=1e-8), \
        patch.object(harness, "softmax_t", side_effect=_probabilities):
    _precision = harness.run_language(None, "en", 2, 2, unclamped=True)
_precision_payload = deepcopy(_release)
_precision_payload["config"]["per_lang"] = 2
_precision_payload["report"]["en"] = _precision["report"]
_precision_payload["cases"] = _precision["cases"]
check_true("release/tiny positive raw temperatures retain their precision and validate",
           _precision["report"]["temperature"] == 1e-8
           and all(row["temperature"] == 1e-8 for row in _precision["cases"])
           and not validate_release_report(_precision_payload, ["en"]))

with redirect_stderr(StringIO()):
    try:
        harness.main(["--require-complete", "--out", "report.json",
                      "--model-revision", "main", "--dataset-revision", _dataset_sha])
    except SystemExit as _exc:
        check("release/invalid revision rejected before model load", _exc.code, 2)
    else:
        check_true("release/invalid revision rejected before model load", False)

with patch.object(laya, "__file__", os.path.join(os.path.dirname(harness._REPO_ROOT),
                                               "other-checkout", "laya", "__init__.py")), \
        patch.object(harness, "source_revision", return_value=(_source_sha, False)), \
        patch.object(laya, "load") as _load_model, redirect_stderr(StringIO()) as _stderr:
    try:
        harness.main(["--require-complete", "--out", "report.json",
                      "--model-revision", _model_sha, "--dataset-revision", _dataset_sha])
    except SystemExit as _exc:
        check("release/another checkout's package rejected", _exc.code, 2)
    else:
        check_true("release/another checkout's package rejected", False)
    check_true("release/package provenance rejected before model load",
               not _load_model.called and "same checkout" in _stderr.getvalue())

_agent = SimpleNamespace(
    revision=_model_sha, device="cpu", cfg={"max_len": 512, "head_max_len": 192},
    model=SimpleNamespace(eval=lambda: None), temperature_by_options={},
    temperature_by_options_raw={},
)
with tempfile.TemporaryDirectory() as _tmp:
    _out = os.path.join(_tmp, "report.json")
    with patch.object(laya, "load", return_value=_agent) as _load_model, \
            patch.object(harness, "run_language", return_value={
                "report": deepcopy(_release["report"]["en"]), "cases": [deepcopy(_record)]}) as _run, \
            patch.object(harness, "source_revision", return_value=(_source_sha, False)), \
            patch.object(harness, "package_version", return_value="1"):
        _rc = harness.main(["--model-revision", _model_sha,
                            "--dataset-revision", _dataset_sha, "--per-lang", "1",
                            "--n-opts", "2", "--out", _out, "--require-complete"])
    check("release/CLI writes a complete report", _rc, 0)
    check("release/CLI pins model load", _load_model.call_args.kwargs["revision"], _model_sha)
    check("release/CLI pins dataset run", _run.call_args.args[-1], _dataset_sha)
    with open(_out, encoding="utf-8") as _fh:
        _written = json.load(_fh)
    check("release/CLI records code revision", _written["config"]["source_revision"], _source_sha)

    with patch.object(laya, "load", return_value=_agent), \
            patch.object(harness, "run_language", side_effect=RuntimeError("dataset unavailable")), \
            patch.object(harness, "source_revision", return_value=(_source_sha, False)), \
            patch.object(harness, "package_version", return_value="1"):
        _rc = harness.main(["--model-revision", _model_sha,
                            "--dataset-revision", _dataset_sha, "--per-lang", "1",
                            "--n-opts", "2", "--out", _out, "--require-complete"])
    with open(_out, encoding="utf-8") as _fh:
        _failed = json.load(_fh)
    check("release/failed run exits nonzero", _rc, 1)
    check("release/failed run keeps diagnostic artifact",
          _failed["report"]["en"]["error"], "dataset unavailable")

    _incomplete = deepcopy(_record)
    _incomplete.pop("state")
    with patch.object(laya, "load", return_value=_agent), \
            patch.object(harness, "run_language", return_value={
                "report": deepcopy(_release["report"]["en"]), "cases": [_incomplete]}), \
            patch.object(harness, "source_revision", return_value=(_source_sha, False)), \
            patch.object(harness, "package_version", return_value="1"):
        _rc = harness.main(["--model-revision", _model_sha,
                            "--dataset-revision", _dataset_sha, "--per-lang", "1",
                            "--n-opts", "2", "--out", _out, "--require-complete"])
    with open(_out, encoding="utf-8") as _fh:
        _failed = json.load(_fh)
    check("release/invalid case exits nonzero without raising", _rc, 1)
    check_true("release/invalid case keeps diagnostic artifact", "state" not in _failed["cases"][0])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
