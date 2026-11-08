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


def test_a_malformed_question_is_refused_by_layas_own_check():
    # the translation reads `q["type"]` and `q["instructions"]` directly: a missing key used to
    # surface as a bare `KeyError: 'type'` and a non-dict as `TypeError: string indices must be
    # integers`, naming neither the question nor what to fix, on fit as well as predict.
    # vega's own unknown-type message already names the question; the check changes only the wording
    with pytest.raises(ValueError):
        lv._to_vega_questions({"dept": {"type": "ranking", "instructions": "x"}})
    with pytest.raises(ValueError, match=r"question 'dept': definition must be a dict"):
        lv._to_vega_questions({"dept": "the billing team"})
    with pytest.raises(ValueError, match="question 'dept': no 'instructions'"):
        lv._to_vega_questions({"dept": {"type": "choice", "criteria": Q["dept"]["criteria"]}})
    with pytest.raises(ValueError, match="question 'urgency': a score question takes 'criteria'"):
        lv._to_vega_questions({"urgency": {"type": "score", "instructions": "x"}})
    with pytest.raises(ValueError, match=r"question 'dept': a choice question takes 'criteria'"):
        lv._to_vega_questions({"dept": {"type": "choice", "instructions": "x"}})


def test_the_question_check_keeps_every_shape_the_translation_supports():
    vq = lv._to_vega_questions(Q)
    assert vq["churn"]["criteria"] == {"false": "stays", "true": "threatens to leave"}, "noul labels"
    assert vq["urgency"]["criteria"] == ["not urgent", "soon", "blocking"], "score levels"
    assert vq["dept"]["type"] == "choice" and vq["dept"]["criteria"] == Q["dept"]["criteria"]
    bare = {"type": "noul", "instructions": "x"}
    assert lv._to_vega_questions({"k": bare})["k"] == {"instructions": "x", "type": "boolean"}
    ints = {"type": "choice", "instructions": "x", "criteria": [1, 2]}
    assert lv._to_vega_questions({"k": ints})["k"]["criteria"] == {"1": None, "2": None}, \
        "laya allows number labels; vega stringifies them into answer keys"


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


def test_gold_label_reads_a_yes_no_number_the_way_the_fine_tune_does():
    # `str(1.0)` is "1.0", which is not one of the true-words, so a float yes/no used to train the
    # head on the false side of its own label. `target_from_expected` takes 1 and 1.0 as true.
    assert lv._gold_label(Q["churn"], 1.0) == "true"
    assert lv._gold_label(Q["churn"], 0.0) == "false"
    assert lv._gold_label(Q["churn"], "YES") == "true"
    assert lv._gold_label(Q["churn"], {"label": 1}) == "true"
    for unreadable in ("maybe", "", None, 0.5, 2, "11", {"probabilities": {}}):
        with pytest.raises(ValueError, match="noul gold"):
            lv._gold_label(Q["churn"], unreadable)


def test_gold_label_refuses_a_score_level_the_question_does_not_have():
    assert lv._gold_label(Q["urgency"], "2") == "2"
    assert lv._gold_label(Q["urgency"], 2.0) == "2"
    for outside in (9, "9", -1):
        with pytest.raises(ValueError, match=r"level index in \[0, 2\]"):
            lv._gold_label(Q["urgency"], outside)
    with pytest.raises(ValueError, match="one whole level"):
        lv._gold_label(Q["urgency"], 1.5)           # used to truncate to level 1
    with pytest.raises(ValueError, match="one of \\['not urgent', 'soon', 'blocking'\\]"):
        lv._gold_label(Q["urgency"], "urgent")      # used to raise list.index(x): x not in list
    with pytest.raises(ValueError, match="not True"):
        lv._gold_label(Q["urgency"], True)          # used to read as level 1


def test_gold_label_refuses_a_choice_label_the_question_never_offered():
    assert lv._gold_label(Q["dept"], "other") == "other"
    assert lv._gold_label(Q["dept"], {"probabilities": {"billing": 1.0}}) == "billing"
    for unknown in ("nope", "Billing", None, True, 1):
        with pytest.raises(ValueError, match="choice gold"):
            lv._gold_label(Q["dept"], unknown)


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


def test_fold_refuses_an_answer_number_that_is_not_a_finite_number():
    # `float()` reads True as 1.0 and lets nan/inf through, and every comparison against a NaN
    # answers False: the answer used to clear a min_confidence gate of 0.9 and leave as route
    # "auto", which is the one field this module tells callers to gate on.
    for unreadable in (float("nan"), float("inf"), float("-inf"), True, "high", None, 10 ** 400):
        with pytest.raises(ValueError, match="the probability of 'billing'"):
            lv._fold(_engine_answer(probs={"billing": unreadable, "technical": 0.0}), Q["dept"])
    for unreadable in (float("nan"), float("inf"), True, "high"):
        with pytest.raises(ValueError, match="the engine's confidence"):
            lv._fold(_engine_answer(probs={"billing": 0.5, "technical": 0.5}, confidence=unreadable),
                     Q["dept"])
    # a numeric string is a number here, as `float()` has always read it on every laya path
    readable = lv._fold(_engine_answer(probs={"billing": 0.5, "technical": 0.5}, confidence="0.9"),
                        Q["dept"])
    assert readable["answer_confidence"] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="the engine's score"):
        lv._fold(_engine_answer(probs={"0": 0.5, "1": 0.5}, score=float("nan")), Q["urgency"])
    with pytest.raises(ValueError, match="the engine's expected score"):
        lv._fold(_engine_answer(probs={"0": 0.5, "1": 0.5}, expected_score=float("nan")), Q["urgency"])
    # the engine answers in numpy float32, which is a real number and must keep working
    import numpy as np

    ok = lv._fold(_engine_answer(probs={"billing": np.float32(0.7), "technical": np.float32(0.3),
                                        "other": np.float32(0.0)}, choice="billing"), Q["dept"])
    assert ok["probabilities"] == {"billing": 0.7, "technical": 0.3, "other": 0.0}
    assert ok["confidence"] == pytest.approx(0.444, abs=1e-4)
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
    ("prediction_set", ["x", "y"], "holds 2 options"),
])
def test_route_reviews_on_each_engine_signal(field, value, word):
    a = _bare_agent()
    folded = {"abstain": False, "unbound": False, "prediction_set": ["x"], "answer_confidence": 0.8}
    folded[field] = value
    r = a._route({}, folded, None)
    assert r["route"] == "review" and word in r["route_reason"]


def test_unbound_is_reported_but_does_not_route():
    """It fired on 71% of in-domain banking77 rows with the published threshold."""
    a = _bare_agent()
    folded = {"abstain": False, "unbound": True, "prediction_set": ["x"], "answer_confidence": 0.8}
    assert a._route({}, folded, None)["route"] == "auto"


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


# ----------------------------------------------------------------- the hook lifecycle
def _hook_agent(monkeypatch, answer, calls):
    a = object.__new__(lv.VegaAgent)
    a.model_id = "vega-test"
    a.hooks = []
    a.hooks_raise = True
    a.hooks_timeout = None

    def fake_decide(self, state, questions, **kw):
        calls.append((state, questions))
        return answer

    monkeypatch.setattr(lv.VegaAgent, "_decide", fake_decide)
    return a


def test_predict_runs_the_hook_lifecycle(monkeypatch):
    answer = {"answers": {}, "usage": {"input_tokens": 3, "output_tokens": 0}, "modality": "text"}
    calls, events = [], []
    a = _hook_agent(monkeypatch, answer, calls)

    class Probe:
        def on_predict_start(self, ctx):
            events.append(("start", ctx.states, dict(ctx.questions)))

        def on_predict_end(self, ctx):
            events.append(("end", ctx.results))

        def on_error(self, ctx):
            events.append(("error", ctx.error))

    q = {"dept": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}
    out = a.predict("s", q, hooks=Probe())
    assert out is answer
    assert [e[0] for e in events] == ["start", "end"]
    assert events[0][1] == "s" and events[0][2] == q, "the hook sees the call's state and questions"
    assert events[1][1] is answer, "the end hook sees the finished answer"
    assert calls == [("s", q)]


def test_a_start_hook_can_short_circuit_and_rewrite_questions(monkeypatch):
    answer = {"answers": {}, "usage": {"input_tokens": 3, "output_tokens": 0}, "modality": "text"}
    calls = []
    a = _hook_agent(monkeypatch, answer, calls)
    other = {"dept": {"type": "noul", "instructions": "will they churn?"}}
    given = {"dept": {"type": "noul", "instructions": "x"}}

    class Skipper:
        def on_predict_start(self, ctx):
            ctx.questions = other
            ctx.skip({"answers": {}, "cached": True})

    assert a.predict("s", given, hooks=Skipper()) == {"answers": {}, "cached": True}
    assert calls == [], "skip() short-circuits the decision"

    class Rewriter:
        def on_predict_start(self, ctx):
            ctx.questions = other

    a.predict("s", given, hooks=Rewriter())
    assert calls and calls[0][1] is other, "the rewritten questions are what decide on"


def test_a_failing_hook_follows_the_raise_policy(monkeypatch):
    answer = {"answers": {}, "usage": {"input_tokens": 3, "output_tokens": 0}, "modality": "text"}
    calls, events = [], []
    a = _hook_agent(monkeypatch, answer, calls)

    class Boom:
        def on_predict_start(self, ctx):
            raise RuntimeError("boom")

        def on_error(self, ctx):
            events.append("error")

        def on_predict_end(self, ctx):
            events.append("end")

    q = {"dept": {"type": "noul", "instructions": "x"}}
    with pytest.raises(RuntimeError, match="boom"):
        a.predict("s", q, hooks=Boom())
    assert events == ["error", "end"], "on_error fired, then on_predict_end on the failure path"
    events.clear()
    with pytest.raises(RuntimeError, match="boom"):
        a.predict("s", q, hooks=Boom(), hooks_raise=True)
    assert a.predict("s", q, hooks=Boom(), hooks_raise=False) is answer, \
        "hooks_raise=False warns and continues"


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
    assert agent.thresholds()["policy"] == "conformal set + abstain"


def _synthetic_rows(n=28):
    """Labelled rows a head can learn: two departments, two kinds of complaint."""
    rows = []
    bills = ["Charged twice for %s, refund the duplicate." % m for m in
             ("March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December", "January", "February", "the first quarter", "the renewal")]
    bugs = ["The app crashes when I open %s." % s for s in
            ("settings", "the inbox", "a report", "the dashboard", "search", "billing history",
             "my profile", "notifications", "the export page", "the login screen", "the map",
             "attachments", "the calendar", "the archive")]
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
    # "band: ..." when the ten calibration rows could certify the line, "band could not hold ..."
    # with the conformal rule deciding when they could not; either way the band is what spoke
    assert out["answers"]["dept"]["route_reason"].startswith("band")
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


# ----------------------------------------------------------------------------- video
# The clip is split into what the engine can read (laya/video.py) and the path is chosen by
# content. The helpers need the ffmpeg binary and nothing else, so they are tested on synthetic
# clips ffmpeg itself generates; the model-backed tests ride the LAYA_VEGA_MEDIA_TESTS gate.

needs_ffmpeg = pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None, reason="ffmpeg is not on the PATH")


def _clip(tmp_path, name, audio_src, video_src="color=c=orange:s=160x120:d=3"):
    import subprocess
    out = str(tmp_path / ("%s.mp4" % name))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", video_src,
                    "-f", "lavfi", "-i", audio_src, "-t", "3", "-shortest", "-pix_fmt", "yuv420p", out],
                   check=True)
    return out


def test_probe_survives_a_banner_that_reports_no_duration(monkeypatch):
    # an elementary .h264 has no container duration, so ffmpeg banners "Duration: N/A" and the
    # fallback used to crash the three-way unpack. 0.0 is what an absent `format.duration`
    # already gets on the ffprobe path. Weight-free: ffprobe is hidden and the banner is fed in.
    import types

    from laya import video as v

    banner = (b"Input #0, h264, from 'clip.h264':\n"
              b"  Duration: N/A, bitrate: N/A\n"
              b"  Stream #0:0: Video: h264 (High), yuv420p, 64x64, 10 fps\n")
    monkeypatch.setattr(v.shutil, "which", lambda name: "/usr/bin/true" if name == "ffmpeg" else None)
    monkeypatch.setattr(v.subprocess, "run",
                        lambda args, **kw: types.SimpleNamespace(returncode=1, stdout=b"", stderr=banner))
    assert v.probe("clip.h264") == {"duration_s": 0.0, "has_audio": False, "has_video": True}
    timed = banner.replace(b"Duration: N/A", b"Duration: 00:01:02.50")
    monkeypatch.setattr(v.subprocess, "run",
                        lambda args, **kw: types.SimpleNamespace(returncode=1, stdout=b"", stderr=timed))
    assert v.probe("clip.h264")["duration_s"] == 62.5


@needs_ffmpeg
def test_video_helpers_on_synthetic_clips(tmp_path):
    import numpy as np
    from laya import video as v

    silent = _clip(tmp_path, "silent", "anullsrc=r=16000:cl=mono")
    tone = _clip(tmp_path, "tone", "sine=f=440:r=16000", "testsrc=s=160x120:d=3")
    for path in (silent, tone):
        info = v.probe(path)
        assert info["has_video"] and info["has_audio"] and 2.5 < info["duration_s"] < 3.5
    quiet = v.extract_audio(silent)
    loud = v.extract_audio(tone)
    assert quiet.dtype == np.float32 and loud.dtype == np.float32
    assert abs(len(loud) - 3 * v.TARGET_SR) < v.TARGET_SR // 10, "resampled to 16 kHz, 3 s"
    assert not v.has_speech(quiet), "silence never reaches the transcriber"
    assert v.has_speech(loud), "a loud track is worth transcribing"
    # the word gate: a tone comes back from whisper as rows of dots, which are not words
    assert not v.has_speech(loud, ". . . . . . . . . . . .")
    assert not v.has_speech(loud, "thank you")
    assert v.has_speech(loud, "refund the duplicate today")
    frames, times = v.sample_frames(tone, 4)
    assert len(frames) == 4 and [round(t, 2) for t in times] == [0.38, 1.12, 1.88, 2.62]
    assert frames[0].size == (160, 120)
    sheet = v.contact_sheet(frames, cell=128)
    assert sheet.size == (256, 256), "four frames tile 2x2"


@needs_ffmpeg
def test_video_bytes_go_through_a_temporary_file(tmp_path):
    from laya import video as v

    path = _clip(tmp_path, "silent", "anullsrc=r=16000:cl=mono")
    raw = open(path, "rb").read()
    got, tmp = v.as_path(raw)
    try:
        assert tmp is not None and got == tmp and os.path.getsize(got) == len(raw)
    finally:
        os.unlink(tmp)
    assert v.as_path(path) == (path, None)
    with pytest.raises(TypeError, match="path or raw video-file bytes"):
        v.as_path(3.5)
    with pytest.raises(FileNotFoundError):
        v.as_path(str(tmp_path / "missing.mp4"))


def test_video_is_one_of_the_exclusive_inputs():
    a = object.__new__(lv.VegaAgent)
    with pytest.raises(ValueError, match="exactly one of state=, image=, audio=, video="):
        a.predict("text", {"q": {"type": "noul", "instructions": "x"}}, video="clip.mp4")


def test_router_sends_video_to_vega():
    r = lv.Router(fast=_StubAgent("fast"), vega=None)
    assert r.choose(video="clip.mp4")[0] == "vega"
    with pytest.raises(ValueError, match="text only"):
        r.predict(video="clip.mp4", questions={"q": {"type": "noul", "instructions": "x"}}, model="fast")


@needs_ffmpeg
@pytest.mark.skipif(not MEDIA_TESTS, reason="set LAYA_VEGA_MEDIA_TESTS=1 (pulls the vision tower)")
def test_video_silent_clip_goes_to_frames(agent, tmp_path):
    q = {"q": {"type": "noul", "instructions": "Is the picture mostly one flat colour?"}}
    out = agent.predict(video=_clip(tmp_path, "silent", "anullsrc=r=16000:cl=mono"), questions=q)
    assert out["modality"] == "video" and out["video"]["path"] == "frames"
    assert len(out["video"]["frame_times"]) == 4 and "transcript" not in out
    assert set(out["answers"]["q"]) >= {"noul", "route", "prediction_set"}


@needs_ffmpeg
@pytest.mark.skipif(not MEDIA_TESTS or __import__("shutil").which("say") is None,
                    reason="needs LAYA_VEGA_MEDIA_TESTS=1 and macOS `say` to make a speech clip")
def test_video_with_speech_decides_on_the_transcript(agent, tmp_path):
    import subprocess
    aiff = str(tmp_path / "speech.aiff")
    subprocess.run(["say", "-o", aiff, "We were billed twice for March. Please refund the "
                    "duplicate today or we will cancel our plan."], check=True)
    clip = str(tmp_path / "speech.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=160x120",
                    "-i", aiff, "-shortest", "-pix_fmt", "yuv420p", "-c:a", "aac", clip], check=True)
    q = {"dept": {"type": "choice", "instructions": "Which department should handle this?",
                  "criteria": {"billing": "invoices, payments, refunds",
                               "technical": "bugs, outages", "other": "everything else"}}}
    out = agent.predict(video=clip, questions=q)
    assert out["modality"] == "video" and out["video"]["path"] == "speech"
    assert "billed" in out["video"]["transcript"].lower()
    assert out["answers"]["dept"]["choice"] == "billing"


# ----------------------------------------------------------------------------- rows from a file
ROUTING_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "data", "routing_labels.csv")
ROUTING_Q = {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "invoices, payments, refunds", "hr": "jobs, payroll, leave, policy",
                          "pr": "press, media, public statements", "sales": "pricing, demos, quotes, trials",
                          "technical": "bugs, outages, API and product questions"}},
    "urgency": {"type": "score", "instructions": "How soon does this need a reply?",
                "criteria": ["can wait", "this week", "today"]},
}


def test_the_example_csv_loads_as_fit_rows():
    rows = lv.VegaAgent.load_rows(ROUTING_CSV, ROUTING_Q)
    assert len(rows) == 60 and all(set(r) == {"state", "gold"} for r in rows)
    teams = {r["gold"]["team"] for r in rows}
    assert teams == set(ROUTING_Q["team"]["criteria"]), "every team in the sheet has rows"
    assert {r["gold"]["urgency"] for r in rows} == {"0", "1", "2"}
    # the rows become vegaml examples with the labels in the engine's own spelling
    a = object.__new__(lv.VegaAgent)
    examples, sheet = a._examples(ROUTING_CSV, ROUTING_Q)
    assert len(examples) == 60 and set(sheet) == {"team", "urgency"}
    state, labels, _ = examples[0]
    assert labels == {"team": "billing", "urgency": "2"} and "charged twice" in state


def test_csv_columns_must_be_question_ids(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("state,dept\nhello,billing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="match no question id"):
        lv.VegaAgent.load_rows(str(p), ROUTING_Q)
    p.write_text("text,team\nhello,billing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a `state` column"):
        lv.VegaAgent.load_rows(str(p), ROUTING_Q)


def test_jsonl_rows_load_as_given(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"state": "a", "gold": {"team": "hr"}}\n\n{"state": "b", "gold": {"team": "pr"}}\n',
                 encoding="utf-8")
    rows = lv.VegaAgent.load_rows(str(p))
    assert [r["gold"]["team"] for r in rows] == ["hr", "pr"]


def test_a_gold_keyed_to_no_question_id_is_refused_not_dropped():
    # load_rows refuses an unknown CSV column; the dict form of the same mistake used to be
    # filtered out by `if qid in qs`, so fit reported success on labels that trained nothing.
    a = object.__new__(lv.VegaAgent)
    rows = [{"state": "s", "questions": ROUTING_Q, "gold": {"team": "hr", "dept": "billing"}}]
    with pytest.raises(ValueError, match=r"gold key\(s\) \['dept'\] match no question id"):
        a._examples(rows, None)
    with pytest.raises(ValueError, match=r"gold key\(s\) \['dept', 'x'\]"):
        a._examples([{"state": "s", "gold": {"dept": "billing", "x": "y"}}], ROUTING_Q)
    examples, sheet = a._examples([{"state": "s", "gold": {"team": "hr"}}], ROUTING_Q)
    assert examples[0][1] == {"team": "hr"}, "a gold naming only some of the questions still fits"


@needs_model
def test_fit_from_the_example_csv(agent):
    report = agent.fit(ROUTING_CSV, questions=ROUTING_Q, calibrate=True)
    assert report["n_rows"] == 60 and set(report["heads"]) == {"team", "urgency"}
    assert "calibration" in report and set(report["calibration"]["questions"]) <= {"team", "urgency"}
    out = agent.predict("Our trial ends tomorrow and we want to buy before it lapses.", ROUTING_Q)
    assert out["answers"]["team"]["choice"] in ROUTING_Q["team"]["criteria"]
    assert out["answers"]["team"]["route"] in ("auto", "review")


# ----------------------------------------------------------------------------- fit's own rules
def test_min_rows_counts_the_widest_question_and_the_calibration_share():
    assert lv.VegaAgent.min_rows(ROUTING_Q) == 35            # 5 options x 5 folds, plus 10 to calibrate
    assert lv.VegaAgent.min_rows(ROUTING_Q, calibrate=False) == 25
    assert lv.VegaAgent.min_rows({"q": {"type": "noul", "instructions": "x"}}) == 20


def test_split_holds_out_at_least_ten_rows_across_every_label():
    ex = [("s%d" % i, {"team": "abcde"[i % 5]}, {}) for i in range(40)]
    fit, cal = lv.VegaAgent._split(ex, lv.VegaAgent.CALIBRATION_SHARE)
    assert len(cal) == 10 and len(fit) == 30
    assert {e[1]["team"] for e in cal} == set("abcde"), "every label is on the calibration side"
    ex2 = [("s%d" % i, {"d": "ab"[i % 2]}, {}) for i in range(24)]
    fit2, cal2 = lv.VegaAgent._split(ex2, lv.VegaAgent.CALIBRATION_SHARE)
    assert len(cal2) == 10 and len(fit2) == 14, "a quarter of 24 is 6, but ten is the floor"
    assert fit2 + cal2 and lv.VegaAgent._split(ex2, 0.25) == (fit2, cal2), "deterministic"


def test_fit_refuses_fewer_rows_than_the_minimum():
    a = object.__new__(lv.VegaAgent)
    a.folds = 5
    # spread over the sheet's own five teams: `fit` reads every row's gold before it counts rows
    teams = list(ROUTING_Q["team"]["criteria"])
    rows = [{"state": "s%d" % i, "gold": {"team": teams[i % 5], "urgency": "1"}} for i in range(30)]
    with pytest.raises(ValueError, match="at least 35 labelled rows"):
        a.fit(rows, questions=ROUTING_Q)
    with pytest.raises(ValueError, match="at least 25 labelled rows"):
        a.fit(rows[:20], questions=ROUTING_Q, calibrate=False)


def test_an_unreachable_band_hands_the_route_to_the_conformal_rule():
    class _Band:
        questions = {"team": {"threshold": 1.000001, "reachable": False, "examples": 10,
                              "target_precision": 0.95}}

    class _Cal:
        band = _Band()

    a = object.__new__(lv.VegaAgent)
    a.alpha, a.risk, a.calibration = 0.1, "0.1", _Cal()
    raw = {"route": "review", "route_reason": "confidence 1.000 is below the 1.000 review line", "_qid": "team"}
    single = {"abstain": False, "unbound": False, "prediction_set": ["billing"], "answer_confidence": 0.9}
    r = a._route(raw, single, None)
    assert r["route"] == "auto" and r["route_reason"].startswith("band could not hold precision 0.95 on 10")
    wide = dict(single, prediction_set=["billing", "hr", "pr"])
    assert a._route(raw, wide, None)["route"] == "review"
    # a band that did hold keeps deciding
    _Band.questions["team"]["reachable"] = True
    assert a._route(raw, single, None) == {"route": "review", "route_reason": "band: confidence 1.000 is below the 1.000 review line"}


# -----------------------------------------------------------------------------
# frame sampling: the times must be the ones the frames came from, weight-free

def test_sample_frames_reports_the_time_each_frame_came_from(monkeypatch):
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    from laya import video as v

    buf = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buf, "PNG")
    png = buf.getvalue()

    def fake_run(args, what):
        return png if float(args[4]) in (1.25, 6.25, 8.75) else b""   # t=3.75 yields nothing

    monkeypatch.setattr(v, "_run", fake_run)
    monkeypatch.setattr(v.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    frames, times = v.sample_frames("unused.mp4", n=4, duration_s=10.0)
    assert len(frames) == 3
    assert [round(t, 3) for t in times] == [1.25, 6.25, 8.75]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ----------------------------------------------------------------------------- 0.4.29: policies
class _Cal:
    """The two things the helpers read off a vegaml Calibration."""

    def __init__(self, labels):
        self.questions = {"team": {"labels": labels, "type": "choice"}}

    def apply(self, result):
        out = dict(result)
        out["answers"] = {q: dict(a, calibrated_by_test=True) for q, a in result["answers"].items()}
        return out


def test_an_option_the_policy_never_saw_skips_its_calibration():
    cal = _Cal(["billing", "technical"])
    raw = {"answers": {"team": {"probs": {"billing": 0.2, "technical": 0.3, "refunds": 0.5}},
                       "urgent": {"probs": {"false": 0.4, "true": 0.6}}}}
    stale = lv._stale_for_calibration(raw, cal)
    assert stale == {"team": ["refunds"]}
    done = lv._apply_calibration(raw, cal, stale)
    assert "calibrated_by_test" not in done["answers"]["team"]
    assert done["answers"]["urgent"]["calibrated_by_test"] is True
    # a strict subset of the fitted options is padded and calibrated, as before
    sub = {"answers": {"team": {"probs": {"billing": 1.0}}}}
    assert lv._stale_for_calibration(sub, cal) == {}


def test_the_guarded_head_records_its_residual():
    import numpy as np

    class _Head:
        def __init__(self, *a, **k):
            pass

        def predict(self, x, allowed=None):
            return {"a": 1.0}

    head = lv._guarded_head_class(_Head)()
    head.mu, head.sd, head.c = np.zeros(3, np.float32), np.ones(3, np.float32), np.zeros(3, np.float32)
    head.P = np.array([[1.0], [0.0], [0.0]], np.float32)
    head.predict(np.array([1.0, 0.0, 0.0]))
    assert head.last_residual == 0.0
    head.predict(np.array([0.0, 1.0, 0.0]))
    assert abs(head.last_residual - 1.0) < 1e-6


def test_a_record_past_the_limit_is_outbound():
    a = _bare_agent()

    class _V:
        heads = {"team": (type("H", (), {"last_residual": 0.9})(), 0.8)}

    a.vega, a._novelty = _V(), ("team", 0.7)
    assert a._read_novelty() == {"residual": 0.9, "limit": 0.7, "question": "team"}
    a._novelty = ("team", 0.95)
    assert a._read_novelty() is None
    a._novelty = None
    assert a._read_novelty() is None
