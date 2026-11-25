"""The vega lane on `laya.Router`: media, long states and fitted heads go to vega; everything the
encoders can read keeps going to the encoders exactly as before 0.4.25.

Weight-free: a stub stands in for `laya.Agent(vegaml=True)`, so the decisions are tested, not
the engine. The lane is off when vegaml is absent and `vega` is left at "auto", which is what
the first test pins, since that is every pre-0.4.25 deployment's situation.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya.router import VEGA_MODEL, Router, _estimate_tokens  # noqa: E402

HAS_VEGAML = importlib.util.find_spec("vegaml") is not None
Q = {"dept": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "money", "technical": "bugs"}},
     "urgent": {"type": "noul", "instructions": "Urgent?"}}
ENGLISH = "Please refund the duplicate charge on invoice 4411 today."
LONG = ("The contract between the parties states the following. " * 3000)   # ~40k tokens


class _StubEngine:
    def __init__(self, heads):
        self.heads = heads


class _StubVega:
    """What the Router needs from a vega agent: `.vega.heads`, `predict`, `predict_long`."""
    model_id = "stub/vega"

    def __init__(self, heads=()):
        self.vega = _StubEngine({h: object() for h in heads})
        self.calls = []

    def predict(self, state=None, questions=None, lang=None, image=None, audio=None, video=None, **kw):
        self.calls.append({"state": state, "image": image, "audio": audio, "video": video, "lang": lang, **kw})
        answers = {q: {"choice": "billing", "probabilities": {"billing": 0.9, "technical": 0.1},
                       "answer_confidence": 0.9, "confidence": 0.9, "route": "auto"} if d["type"] == "choice"
                   else {"noul": 0.2, "answer_confidence": 0.8, "confidence": 0.8, "route": "auto"}
                   for q, d in questions.items()}
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}, "latency_ms": 1.0,
                "modality": "image" if image is not None else "audio" if audio is not None else
                "video" if video is not None else "text"}

    system_one = predict

    def predict_long(self, state, questions, **kw):
        out = self.predict(state, questions, **kw)
        out["usage"]["windows"] = 1
        return out


# ---------------------------------------------------------------------- off by default without vegaml
@pytest.mark.skipif(HAS_VEGAML, reason="vegaml is installed here, so auto turns the lane on")
def test_without_vegaml_the_lane_is_off_and_nothing_changes():
    r = Router()
    assert not r.vega_enabled and VEGA_MODEL not in r.models
    with pytest.raises(ValueError, match="unknown model"):
        r.resolve("vega")
    assert r.route(ENGLISH).model == "english"
    assert r.route(LONG).model != VEGA_MODEL, "a long state stays on the encoders when there is no lane"
    with pytest.raises(ValueError, match="needs the vega lane"):
        r.route(None, Q, media="image")
    with pytest.raises(ImportError, match="pip install -U laya"):
        Router(vega=True)


def test_vega_false_keeps_the_old_behaviour_even_with_an_agent_available():
    r = Router(vega=False)
    assert not r.vega_enabled
    assert r.route(LONG).model != VEGA_MODEL
    with pytest.raises(ValueError, match="needs the vega lane"):
        r.predict(None, Q, image=object())


# ------------------------------------------------------------------------------- the lane itself
def test_an_attached_vega_agent_turns_the_lane_on_and_routes_what_the_encoders_cannot_read():
    stub = _StubVega(heads=("urgent",))
    r = Router(vega=stub)
    assert r.vega_enabled and r.load(VEGA_MODEL) is stub
    # text the encoders can read is untouched
    d = r.route(ENGLISH, {"dept": Q["dept"]})
    assert d.model == "english" and d.reason == "English Latin text"
    # media
    d = r.route(None, Q, media="image")
    assert d.model == VEGA_MODEL and "image input" in d.reason and d["repo"] == "stub/vega"
    # a state past the encoders' budget
    d = r.route(LONG, {"dept": Q["dept"]})
    assert d.model == VEGA_MODEL and "past the 8192" in d.reason
    # a question with a fitted head, however short the state
    d = r.route(ENGLISH, Q)
    assert d.model == VEGA_MODEL and "fitted heads for urgent" in d.reason
    # explicit model= still wins, both ways
    assert r.route(LONG, Q, model="english").model == "english"
    assert r.route(ENGLISH, {"dept": Q["dept"]}, model="vega").model == VEGA_MODEL


def test_predict_hands_media_and_long_states_to_vega_and_stamps_the_routing():
    stub = _StubVega()
    r = Router(vega=stub)
    out = r.predict(None, Q, image="invoice.png")
    assert out["routing"]["model"] == VEGA_MODEL and out["modality"] == "image"
    assert stub.calls[-1]["image"] == "invoice.png" and stub.calls[-1]["state"] is None
    out = r.predict(LONG, Q)
    assert out["routing"]["model"] == VEGA_MODEL and stub.calls[-1]["state"] == LONG
    out = r.predict_long(LONG, Q)
    assert out["routing"]["model"] == VEGA_MODEL and out["usage"]["windows"] == 1
    with pytest.raises(ValueError, match="pass one of"):
        r.predict(None, Q, image="a", audio="b")
    with pytest.raises(ValueError, match="not both"):
        r.predict(ENGLISH, Q, video="c")


def test_vega_over_tokens_is_the_line_and_the_estimate_is_characters_over_four():
    stub = _StubVega()
    assert _estimate_tokens("a" * 400) == 100 and _estimate_tokens({"k": "b" * 40}) == 10
    r = Router(vega=stub, vega_over_tokens=100)
    assert r.route("x" * 404, {"dept": Q["dept"]}).model == VEGA_MODEL
    assert r.route("x" * 396, {"dept": Q["dept"]}).model != VEGA_MODEL


def test_predict_batch_never_routes_to_vega():
    stub = _StubVega(heads=("urgent",))
    r = Router(vega=stub)
    decisions = [r._route(LONG, Q, allow_vega=False), r._route(ENGLISH, Q, allow_vega=False)]
    assert all(d.model != VEGA_MODEL for d in decisions)
    with pytest.raises(ValueError, match="needs the vega lane"):
        r._route(None, Q, media="image", allow_vega=False)


def test_a_reader_name_registers_a_lazy_lane():
    r = Router(vega="4b")
    assert r.vega_enabled and r.route(None, Q, media="audio")["repo"] == "4b"
    assert VEGA_MODEL not in r.models and VEGA_MODEL not in r.registered, "the lane is not a checkpoint"
    assert r.resolve("vega") == VEGA_MODEL
    with pytest.raises(ValueError, match="names the vega lane"):
        r.register("vega", "some/repo")


@pytest.mark.skipif(not HAS_VEGAML, reason="needs vegaml to test the auto lane")
def test_auto_turns_the_lane_on_when_vegaml_is_present_without_importing_it():
    before = "vegaml" in sys.modules
    r = Router()
    assert r.vega_enabled
    assert ("vegaml" in sys.modules) == before, "building a Router must not import vegaml"
    assert r.route(ENGLISH, {"dept": Q["dept"]}).model == "english"
    assert r.route(LONG, {"dept": Q["dept"]}).model == VEGA_MODEL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
