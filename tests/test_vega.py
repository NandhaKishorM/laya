"""`laya.Agent(vegaml=True)`: vega's engine behind laya's surface (laya/vega.py).

Two layers, so the suite is useful on every machine that runs it:

* **Weight-free**, always on: the question translation, the gold-label mapping, the answer
  fold (the "False"/"True" key folding that once scored every yes/no task at 0.000), the
  calibration padding, the automatic route policy, the Router's switching rule, and the
  dispatch in `Agent.__new__` / `laya.load`, including the error a 3.9 interpreter gets.
  These run without vegaml, torch weights, or the network.
* **Model-backed**, only when `LAYA_VEGA_TESTS=1` and vegaml is importable: the real 0.8B
  universal reader answering text, the fitted path end to end (fit, calibrate, save, reload,
  identical answers and routes), and the band taking over the route. `LAYA_VEGA_MEDIA_TESTS=1`
  adds the image path, which pulls the vision tower.

Run: python -m pytest tests/test_vega.py -q
"""
import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import vega as lv  # noqa: E402

VEGA = lv.is_available()
MODEL_TESTS = VEGA and os.environ.get("LAYA_VEGA_TESTS") == "1"
MEDIA_TESTS = MODEL_TESTS and os.environ.get("LAYA_VEGA_MEDIA_TESTS") == "1"

Q = {
    "dept": {"type": "choice", "instructions": "Which department should handle this?",
             "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                          "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?",
              "labels": {"false": "stays", "true": "threatens to leave"}},
}
STATE = "We were billed twice for March. Refund the duplicate today or we cancel our plan."


# ----------------------------------------------------------------------------- translation
def test_questions_translate_into_vegaml_vocabulary():
    vq = lv._to_vega_questions(Q)
    assert vq["churn"]["type"] == "boolean"
    assert vq["churn"]["criteria"] == {"false": "stays", "true": "threatens to leave"}
    assert vq["urgency"] == {"instructions": Q["urgency"]["instructions"], "type": "score",
                             "criteria": ["not urgent", "soon", "blocking"]}
    assert vq["dept"]["type"] == "choice" and vq["dept"]["criteria"] == Q["dept"]["criteria"]


def test_list_form_choice_and_unknown_type():
    vq = lv._to_vega_questions({"k": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}})
    assert vq["k"]["criteria"] == {"a": None, "b": None}
    with pytest.raises(ValueError):
        lv._to_vega_questions({"k": {"type": "ranking", "instructions": "x"}})


def test_labels_for_each_type():
    assert lv._labels_for(Q["churn"]) == ["false", "true"]
    assert lv._labels_for(Q["urgency"]) == ["0", "1", "2"]
    assert lv._labels_for(Q["dept"]) == ["billing", "technical", "other"]


def test_gold_label_accepts_every_laya_answer_shape():
    assert lv._gold_label(Q["churn"], True) == "true"
    assert lv._gold_label(Q["churn"], {"label": "yes"}) == "true"
    assert lv._gold_label(Q["churn"], {"noul": 0.2}) == "false"
    assert lv._gold_label(Q["urgency"], 2) == "2"
    assert lv._gold_label(Q["urgency"], "blocking") == "2"
    assert lv._gold_label(Q["dept"], {"probabilities": {"billing": 0.1, "technical": 0.9}}) == "technical"
    assert lv._gold_label(Q["dept"], {"label": "other"}) == "other"


# ----------------------------------------------------------------------------- the fold
def _engine_answer(**kw):
    base = {"confidence": 0.9, "prediction_set": None, "abstain": False, "unbound": False,
            "readout": "engine"}
    base.update(kw)
    return base


def test_fold_lowercases_the_engines_boolean_keys():
    # the engine keys a boolean's probabilities "False"/"True"; laya spells them lower case
    a = lv._fold(_engine_answer(probs={"False": 0.94, "True": 0.06}, boolean=0.06,
                                prediction_set=[False]), Q["churn"])
    assert set(a["probabilities"]) == {"false", "true"}
    assert a["noul"] == pytest.approx(0.06)
    assert a["prediction_set"] == ["false"]
    assert a["answer_confidence"] == pytest.approx(0.9)
    assert a["type"] == "noul"


def test_fold_choice_and_score_carry_laya_fields():
    c = lv._fold(_engine_answer(probs={"billing": 0.7, "technical": 0.2, "other": 0.1},
                                choice="billing", prediction_set=["billing"]), Q["dept"])
    assert c["choice"] == "billing" and c["prediction_set"] == ["billing"]
    s = lv._fold(_engine_answer(probs={"0": 0.1, "1": 0.8, "2": 0.1}, score=1,
                                expected_score=1.0), Q["urgency"])
    assert s["score"] == 1 and s["legend"] == {"0": "not urgent", "1": "soon", "2": "blocking"}
    assert s["expected_score"] == 1.0


def test_fold_confidence_is_entropy_and_answer_confidence_is_max_p():
    a = lv._fold(_engine_answer(probs={"a": 0.5, "b": 0.5}, choice="a"),
                 {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}})
    assert a["confidence"] == pytest.approx(0.0, abs=1e-6)        # an even split has no information
    assert a["answer_confidence"] == pytest.approx(0.9)            # what the engine reported
    assert lv._entropy_confidence({"a": 1.0, "b": 0.0}) == 1.0


# ----------------------------------------------------------------------------- padding
class _FakeCalibration:
    def __init__(self, labels):
        self.questions = {"q": {"labels": labels, "temperature": 1.0, "qhat": 0.4}}


def test_padding_fills_fitted_options_the_answer_never_mentioned():
    cal = _FakeCalibration(list("ABCDEFGHI"))
    narrow = {"answers": {"q": {"probs": {"A": 0.7, "B": 0.2, "C": 0.1}}}}
    out = lv._pad_for_calibration(narrow, cal)
    assert set(out["answers"]["q"]["probs"]) == set("ABCDEFGHI")
    assert out["answers"]["q"]["probs"]["G"] == 0.0
    assert out["answers"]["q"]["probs"]["A"] == 0.7
    assert narrow["answers"]["q"]["probs"].keys() == {"A", "B", "C"}       # input untouched
    wide = {"answers": {"q": {"probs": {k: 1 / 9 for k in "ABCDEFGHI"}}}}
    assert lv._pad_for_calibration(wide, cal)["answers"]["q"] is wide["answers"]["q"]


# ----------------------------------------------------------------------------- the route
def _bare_agent():
    """A VegaAgent with no engine, enough to exercise `_route`."""
    a = object.__new__(lv.VegaAgent)
    a.alpha, a.risk = "0.1", "0.1"
    return a


def test_route_is_auto_only_when_everything_agrees():
    a = _bare_agent()
    folded = {"abstain": False, "unbound": False, "prediction_set": ["x"], "answer_confidence": 0.8}
    r = a._route({}, folded, None)
    assert r["route"] == "auto" and "singleton" in r["route_reason"]


@pytest.mark.parametrize("field, value, word", [
    ("abstain", True, "abstains"),
    ("unbound", True, "unbound"),
    ("prediction_set", ["x", "y"], "holds 2 options"),
])
def test_route_reviews_on_each_engine_signal(field, value, word):
    a = _bare_agent()
    folded = {"abstain": False, "unbound": False, "prediction_set": ["x"], "answer_confidence": 0.8}
    folded[field] = value
    r = a._route({}, folded, None)
    assert r["route"] == "review" and word in r["route_reason"]


def test_route_prefers_a_fitted_band_and_reviews_unvouched_heads():
    a = _bare_agent()
    banded = a._route({"route": "review", "route_reason": "below the line"},
                      {"abstain": False, "unbound": False, "prediction_set": ["x"]}, None)
    assert banded["route"] == "review" and banded["route_reason"].startswith("band:")
    # a head answer carries no abstain and no set; with no band nothing can vouch for it
    head = a._route({}, {"abstain": None, "unbound": None, "prediction_set": None,
                         "answer_confidence": 0.99}, None)
    assert head["route"] == "review" and "no fitted band" in head["route_reason"]


def test_min_confidence_is_applied_on_top_only_when_given():
    a = _bare_agent()
    folded = {"abstain": False, "unbound": False, "prediction_set": ["x"], "answer_confidence": 0.8}
    assert a._route({}, folded, 0.7)["route"] == "auto"
    r = a._route({}, folded, 0.9)
    assert r["route"] == "review" and "min_confidence" in r["route_reason"]


# ----------------------------------------------------------------------------- switching
class _StubAgent:
    def __init__(self, name):
        self.name = name
        self.vega = type("V", (), {"heads": {}})()

    def predict(self, state=None, questions=None, **kw):
        return {"answers": {q: {"choice": self.name} for q in (questions or {})}}


def test_router_rule_in_order():
    fast, slow = _StubAgent("fast"), _StubAgent("vega")
    r = lv.Router(fast=fast, vega=slow, fast_context_tokens=8)
    q = {"dept": Q["dept"]}
    assert r.choose("I was charged twice", q)[0] == "fast"
    assert r.choose("I was charged twice", q, image=object())[0] == "vega"
    assert r.choose("x " * 100, q)[0] == "vega"                        # longer than 8 tokens
    assert r.choose("मुझे दो बार चार्ज किया गया", q)[0] == "vega"      # non-Latin script
    assert r.choose("I was charged twice", q, lang="de")[0] == "vega"
    slow.vega.heads["dept"] = (None, 0.9)
    assert r.choose("I was charged twice", q)[0] == "vega"             # a fitted head wins
    out = r.predict("I was charged twice", q, model="fast")
    assert out["routing"] == {"model": "fast", "reason": "explicit model=fast"}
    assert out["answers"]["dept"]["choice"] == "fast"
    with pytest.raises(ValueError):
        r.predict(questions=q, image=object(), model="fast")


def test_router_without_a_fast_model_always_picks_vega():
    r = lv.Router(fast=None, vega=_StubAgent("vega"))
    assert r.choose("hello", {"dept": Q["dept"]})[0] == "vega"


# ----------------------------------------------------------------------------- dispatch
def test_agent_new_does_not_divert_without_the_argument():
    for value in (None, False):
        obj = laya.Agent.__new__(laya.Agent, vegaml=value)
        assert type(obj) is laya.Agent


@pytest.mark.skipif(VEGA, reason="vegaml is installed here; the ImportError path needs it absent")
def test_without_vegaml_the_argument_fails_with_the_install_hint():
    with pytest.raises(ImportError, match=r"pip install -U laya"):
        laya.Agent(vegaml=True)
    with pytest.raises(ImportError, match=r"pip install -U laya"):
        laya.load(vegaml=True)


def test_vega_agent_is_exported_and_is_an_agent_subclass():
    assert laya.VegaAgent is lv.VegaAgent
    assert issubclass(lv.VegaAgent, laya.Agent)
    assert "VegaAgent" in laya.__all__


# ----------------------------------------------------------------------------- with weights
needs_model = pytest.mark.skipif(
    not MODEL_TESTS, reason="set LAYA_VEGA_TESTS=1 with vegaml installed (downloads the 0.8B reader)")


@pytest.fixture(scope="module")
def agent():
    if not MODEL_TESTS:
        pytest.skip("model tests off")
    return laya.Agent(vegaml=True)


@needs_model
def test_dispatch_builds_a_vega_agent(agent):
    assert type(agent) is lv.VegaAgent and isinstance(agent, laya.Agent)
    assert type(laya.load(vegaml=True)) is lv.VegaAgent
    assert agent.context_tokens >= 8192


@needs_model
def test_text_answers_carry_laya_and_vega_fields(agent):
    out = agent.predict(STATE, Q)
    for qid, q in Q.items():
        a = out["answers"][qid]
        assert set(lv._labels_for(q)) == set(a["probabilities"])
        assert a["route"] in ("auto", "review") and a["route_reason"]
        assert a["abstain"] in (True, False) and a["unbound"] in (True, False)
        assert a["prediction_set"] is None or isinstance(a["prediction_set"], list)
        assert 0.0 <= a["answer_confidence"] <= 1.0
    assert "choice" in out["answers"]["dept"] and "score" in out["answers"]["urgency"]
    assert "noul" in out["answers"]["churn"]
    assert out["usage"]["input_tokens"] > 0 and out["modality"] == "text"
    assert agent.predict(STATE, {}) == {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                                        "latency_ms": 0.0, "modality": "text"}
    with pytest.raises(ValueError):
        agent.predict(STATE, Q, image="x.png")


@needs_model
def test_batch_long_and_thresholds(agent):
    assert len(agent.predict_batch([STATE, "The app crashes on login."], Q)) == 2
    lg = agent.predict_long(STATE * 20, Q)
    assert lg["usage"]["windows"] == 1 and lg["long"]["applied"] is False
    assert lg["usage"]["input_tokens"] > 500
    assert agent.thresholds()["policy"] == "conformal set + abstain + unbound"


def _synthetic_rows(n=24):
    """Labelled rows a head can learn: two departments, two kinds of complaint."""
    rows = []
    bills = ["Charged twice for %s, refund the duplicate." % m for m in
             ("March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December", "January", "February")]
    bugs = ["The app crashes when I open %s." % s for s in
            ("settings", "the inbox", "a report", "the dashboard", "search", "billing history",
             "my profile", "notifications", "the export page", "the login screen", "the map",
             "attachments")]
    for i in range(n // 2):
        rows.append({"state": bills[i], "questions": {"dept": Q["dept"]}, "gold": {"dept": "billing"}})
        rows.append({"state": bugs[i], "questions": {"dept": Q["dept"]}, "expected": {"dept": "technical"}})
    return rows


@needs_model
def test_fit_calibrate_save_reload_roundtrip():
    a = laya.Agent(vegaml=True)
    rows = _synthetic_rows()
    rep = a.fit(rows)
    assert rep["n_rows"] == len(rows) and "dept" in rep["heads"]
    assert "calibration" in rep and rep["calibration"]["target_precision"] == 0.95
    assert a.calibration is not None and a.thresholds()["policy"] == "band"
    out = a.predict(rows[0]["state"], {"dept": Q["dept"]})
    assert out["calibrated"] is True
    assert out["answers"]["dept"]["route_reason"].startswith("band:")
    with tempfile.TemporaryDirectory() as d:
        saved = a.save_calibration(d)
        assert saved["calibrated"] and os.path.exists(os.path.join(d, "heads.npz"))
        b = laya.Agent(vegaml=True)
        loaded = b.load_calibration(d)
        assert loaded["calibrated"] and set(loaded["heads"]) == set(saved["heads"])
        again = b.predict(rows[0]["state"], {"dept": Q["dept"]})
    assert again["answers"]["dept"]["probabilities"] == out["answers"]["dept"]["probabilities"]
    assert again["answers"]["dept"]["route"] == out["answers"]["dept"]["route"]


@pytest.mark.skipif(not MEDIA_TESTS, reason="set LAYA_VEGA_MEDIA_TESTS=1 (pulls the vision tower)")
def test_image_path(agent):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (320, 200), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 300, 180], outline="black", width=4)
    d.text((40, 80), "INVOICE  TOTAL DUE  $1,240.00", fill="black")
    q = {"kind": {"type": "choice", "instructions": "What kind of document is this?",
                  "criteria": {"invoice": "an invoice or bill", "photo": "a photograph",
                               "form": "a blank form"}}}
    out = agent.predict(image=img, questions=q)
    assert out["modality"] == "image" and out["answers"]["kind"]["readout"] == "engine"
    assert out["answers"]["kind"]["choice"] in q["kind"]["criteria"]
    assert agent.predict_image_text(img, q)["modality"] == "image_withheld"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
