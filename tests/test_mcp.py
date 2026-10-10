"""MCP layer tests: schema, shape and registration. No model weights, no network.

Requires the mcp extra:  pip install "laya[mcp]"

Run: python tests/test_mcp.py
Skips cleanly (exit 0) when the mcp package is not installed, so the core
install keeps working.

Device and preload-list tests follow the laya.serve environment contract
(LAYA_DEVICE / LAYA_PRELOAD / LAYA_MODELS / LAYA_THREADS / LAYA_AUTO_TASK /
LAYA_DEFAULT_MODEL).
"""
import inspect
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import mcp  # noqa: F401
except ImportError:
    print("SKIP: mcp extra not installed (pip install 'laya[mcp]')")
    sys.exit(0)

from laya.mcp.device import agent_device, device_report, env_device, resolve_device, router_agent  # noqa: E402
from laya.mcp.server import _models_from_env, server as mcp_server  # noqa: E402
from laya.mcp.tools import (  # noqa: E402
    BATCH_ITEM_OVERRIDES,
    PRESETS,
    PRESET_ALIASES,
    ToolError,
    _overrides,
    batch_item_key_doc,
    get_available_presets,
    laya_decide,
    laya_predict,
    laya_predict_batch,
    laya_preset,
    laya_route,
    laya_route_batch,
    laya_shortlist,
    laya_status,
    validate_batch_requests,
    validate_budget,
    validate_lang,
    validate_lang_guess,
    validate_min_confidence,
    validate_model,
    validate_preset,
    validate_questions,
    validate_state,
    validate_task,
)

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append("%s%s" % (name, (" -- " + detail) if detail and not cond else ""))
    print("   %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""), flush=True)


def expect_tool_error(name, fn, want_code):
    try:
        fn()
    except ToolError as exc:
        ok(name, exc.code == want_code, "got %r want %r" % (exc.code, want_code))
    except Exception as exc:  # noqa: BLE001
        ok(name, False, "wrong exception %r" % exc)
    else:
        ok(name, False, "no ToolError raised")


# --- device ---------------------------------------------------------------

def test_device():
    ok("device/force_cpu", resolve_device("cpu") == "cpu")
    ok("device/force_cuda", resolve_device("cuda") == "cuda")
    old = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "cpu"
        ok("device/env_cpu", resolve_device() == "cpu")
        os.environ["LAYA_DEVICE"] = "CUDA"
        ok("device/env_case_insensitive", resolve_device() == "cuda")
        os.environ["LAYA_DEVICE"] = "cpu"
        ok("device/force_beats_env", resolve_device("cuda") == "cuda")
    finally:
        if old is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old
    ok("device/fallback", resolve_device(None) in ("cuda", "mps", "xpu", "cpu"))
    # laya.serve contract: LAYA_DEVICE is normalised to what torch's device parser accepts --
    # the device type is lower-cased, the optional index is left alone -- and the same string is
    # both handed to torch and reported by `resolve_device`.
    old_dev = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "cuda:1"
        ok("device/env_for_torch", env_device() == "cuda:1")
        ok("device/env_label", resolve_device() == "cuda:1")
        # torch's parser is case-sensitive, so the type is normalised rather than passed through.
        os.environ["LAYA_DEVICE"] = "CUDA"
        ok("device/env_upper_type_lowercased", env_device() == "cuda")
        ok("device/env_upper_matches_label", env_device() == resolve_device())
        os.environ["LAYA_DEVICE"] = "CUDA:0"
        ok("device/env_upper_index_kept", env_device() == "cuda:0")
        os.environ["LAYA_DEVICE"] = "  CPU  "
        ok("device/env_padded_upper_lowercased", env_device() == "cpu")
        os.environ["LAYA_DEVICE"] = "   "
        ok("device/env_blank_none", env_device() is None)
    finally:
        if old_dev is None:
            os.environ.pop("LAYA_DEVICE", None)
        else:
            os.environ["LAYA_DEVICE"] = old_dev
    rep = device_report()
    ok("device/report_keys", set(rep) >= {"device", "torch_cuda", "torch_version"})
    ok("device/report_cuda_bool", isinstance(rep["torch_cuda"], bool))

    # The auto-detect branch is a second implementation of Agent's own chain
    # (laya/agent.py:355-362), so it has to be exercised on devices the runner does not have.
    # `resolve_device` imports torch inside its body, which makes the module the seam: a fake
    # here forces every branch without CUDA, MPS or XPU hardware. Before this block the only
    # auto-detect assertion was the domain check above, and it passed on a narrower chain.
    def fake_torch(cuda=False, mps=False, xpu=False, mps_attr=True, xpu_attr=True):
        from types import SimpleNamespace

        mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))
        backends = {}
        if mps_attr:
            backends["mps"] = SimpleNamespace(is_available=lambda: mps)
        mod.backends = SimpleNamespace(**backends)
        if xpu_attr:
            mod.xpu = SimpleNamespace(is_available=lambda: xpu)
        return mod

    def labelled_with(fake):
        real = sys.modules.get("torch")
        sys.modules["torch"] = fake
        try:
            return resolve_device()
        finally:
            if real is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = real

    old_env = os.environ.pop("LAYA_DEVICE", None)
    try:
        ok("device/auto_cuda_beats_mps",
           labelled_with(fake_torch(cuda=True, mps=True)) == "cuda")
        # The regression: an Apple-silicon machine has MPS and no CUDA, and Agent builds on MPS.
        ok("device/auto_mps_without_cuda",
           labelled_with(fake_torch(mps=True, xpu=True)) == "mps")
        ok("device/auto_xpu_after_mps",
           labelled_with(fake_torch(xpu=True)) == "xpu")
        # The `hasattr` guards are load-bearing: without one, an absent `torch.backends.mps`
        # raises inside the try block and the `except Exception` swallows it into `cpu`, so an
        # XPU machine is labelled cpu. Asking for the branch that survives the missing
        # attribute is the only way to see that happen.
        ok("device/auto_xpu_when_mps_attribute_missing",
           labelled_with(fake_torch(xpu=True, mps_attr=False)) == "xpu")
        ok("device/auto_nothing_is_cpu", labelled_with(fake_torch()) == "cpu")
        # torch.backends.mps and torch.xpu appeared in different torch versions; an older one
        # must still label cpu rather than raising out of a status call.
        ok("device/auto_old_torch_no_attributes",
           labelled_with(fake_torch(mps_attr=False, xpu_attr=False)) == "cpu")
        # laya_status reads the label through device_report, so the payload has to move with it.
        real = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch(mps=True)
        try:
            ok("device/report_follows_mps", device_report()["device"] == "mps")
        finally:
            if real is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = real
        # And on the machine running this suite, the label must be the device Agent would pick.
        import torch

        available = ("cuda" if torch.cuda.is_available() else
                     "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else
                     "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu")
        ok("device/host_matches_torch", resolve_device() == available,
           "label=%r torch=%r (cuda=%s mps=%s)" % (
               resolve_device(), available, torch.cuda.is_available(),
               hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    finally:
        if old_env is not None:
            os.environ["LAYA_DEVICE"] = old_env


# --- schema -----------------------------------------------------------------

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": {"billing": "money", "other": "rest"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    "churn_risk": {"type": "noul", "instructions": "Leaving?"},
}
STATE = {"body": "refund please"}


def test_schema():
    ok("schema/state_ok", validate_state({"text": "hi"}) == {"text": "hi"})
    for bad in (None, [], "x", {}):
        expect_tool_error("schema/state_bad_%r" % (bad,), lambda b=bad: validate_state(b), "invalid_state")

    out = validate_questions({"dept": {"type": "choice", "instructions": "pick dept",
                                       "criteria": {"a": "A things", "b": "B things"}}})
    ok("schema/questions_choice_ok", out["dept"]["type"] == "choice" and set(out["dept"]["criteria"]) == {"a", "b"})
    out = validate_questions({"urg": {"type": "score", "instructions": "how urgent", "criteria": ["low", "high"]}})
    ok("schema/questions_score_ok", out["urg"]["criteria"] == ["low", "high"])
    out = validate_questions({"risk": {"type": "noul", "instructions": "is true?"}})
    ok("schema/questions_noul_ok", out["risk"]["type"] == "noul")
    out = validate_questions({"risk": {"type": "noul", "instructions": "is true?",
                                       "criteria": {"true": "affirmative", "false": "negative"}}})
    ok("schema/questions_noul_criteria_ok", "criteria" in out["risk"])
    # A noul question takes its criteria keyed only 'true'/'false'. The agent rejects anything
    # else with a ValueError, so accepting 'yes'/'no' here only moved the failure deeper, where
    # the tool wrapper reported a caller mistake as an internal_error.
    expect_tool_error("schema/questions_noul_criteria_bad_key",
                      lambda: validate_questions(
                          {"risk": {"type": "noul", "instructions": "is true?",
                                    "criteria": {"yes": "affirmative", "no": "negative"}}}),
                      "invalid_questions")

    bad_questions = [
        None,
        [],
        {},
        {"x": {"type": "nope", "instructions": "i"}},
        {"x": {"type": "choice", "instructions": ""}},
        {"x": {"type": "choice", "instructions": "i"}},
        {"x": {"type": "score", "instructions": "i"}},
        {"x": {"type": "score", "instructions": "i", "criteria": []}},
        {"x": {"type": "choice", "instructions": "i", "criteria": {}}},
    ]
    for i, bad in enumerate(bad_questions):
        expect_tool_error("schema/questions_bad_%d" % i, lambda b=bad: validate_questions(b), "invalid_questions")

    ok("schema/preset_ok", validate_preset("triage") == "triage")
    expect_tool_error("schema/preset_bad", lambda: validate_preset("nope"), "invalid_preset")
    ok("schema/model_none_auto", validate_model(None) == "auto")
    ok("schema/model_auto", validate_model("auto") == "auto")
    ok("schema/model_multilingual", validate_model("multilingual") == "multilingual")
    expect_tool_error("schema/model_bad", lambda: validate_model("gpt4"), "invalid_model")


def test_model_names():
    """`model` is core's registry, not a second list kept here.

    The names and aliases live in `laya.router`, and `router.predict(model=...)` runs whatever
    arrives through `normalise_name` a few lines after `validate_model` sees it. So any name core
    resolves has to survive this layer, and it has to come back canonical: the `routing.model` a
    caller reads should not depend on how the checkpoint was spelled in the request.
    """
    from laya.router import DEFAULT_MODELS, _ALIASES, normalise_name

    for name in sorted(DEFAULT_MODELS):
        ok("model/canonical_%s" % name, validate_model(name) == name)
    for alias, canonical in sorted(_ALIASES.items()):
        got = validate_model(alias)
        ok("model/alias_%s" % alias, got == canonical, "got %r want %r" % (got, canonical))
    for spelling in ("EN", " Multilingual ", "Typed-Decisions", "AUTO"):
        want = "auto" if spelling.strip().lower() == "auto" else normalise_name(spelling)
        ok("model/casing_and_spacing_%r" % spelling, validate_model(spelling) == want)
    for auto in (None, "auto", "Auto", " auto "):
        ok("model/auto_%r" % (auto,), validate_model(auto) == "auto")
    for bad in ("gpt4", "", "   ", "english-ish", "laya-typed", 5, [], {}):
        expect_tool_error("model/rejected_%r" % (bad,), lambda b=bad: validate_model(b),
                          "invalid_model")

    # The list a client learns from is now core's words, so it must still name every option:
    # the checkpoints, the aliases, and this layer's own `auto` sentinel.
    try:
        validate_model("gpt4")
        message = ""
    except ToolError as exc:
        message = exc.message
    ok("model/error_lists_checkpoints", all(n in message for n in sorted(DEFAULT_MODELS)), repr(message))
    ok("model/error_lists_aliases", "alias" in message and "'en'" in message, repr(message))
    ok("model/error_keeps_auto", "'auto'" in message, repr(message))


def test_presets():
    """The preset list, and the state field each preset reads, come from core.

    Two things used to be able to drift here: the table of names (which was missing `email`, so a
    caller had no way to ask for the preset the CLI has), and which field of the state a preset's
    questions read. A preset says that out loud -- "What does the customer want in `message`?" -- so
    the field is read back out of the questions instead of kept beside them, and `laya_preset` puts a
    caller's lone string under it. Nothing else about the state is touched: more than one key is the
    caller's shape, and guessing there would be a worse failure than the honest one.
    """
    import laya
    from laya.presets import state_field

    def build(attr):
        return getattr(laya, attr)()

    class StateRouter(FakeRouter):
        def predict(self, state, questions, **kwargs):
            self.state = state
            self.questions = questions
            return super().predict(state, questions, **kwargs)

    # The names are a list only in one place, and every entry has to resolve in core.
    fields = {}
    for name, attr in sorted(PRESETS.items()):
        questions = build(attr)
        field = state_field(questions)
        fields[name] = field
        ok("preset/questions_%s" % name, isinstance(questions, dict) and bool(questions))
        ok("preset/names_one_field_%s" % name, isinstance(field, str), repr(field))
        ok("preset/field_is_asked_for_%s" % name,
           any(field in q["instructions"] for q in questions.values()), repr(field))

    # `email` was the whole missing half of the table; the CLI has had it all along.
    ok("preset/email_exposed", "email" in PRESETS, repr(sorted(PRESETS)))

    for name, field in sorted(fields.items()):
        router = StateRouter()
        laya_preset(name, {"text": "the request"}, router=router, preset_builder=build)
        ok("preset/lone_string_placed_%s" % name, router.state == {field: "the request"},
           repr(router.state))
        laya_preset(name, {field: "the request"}, router=router, preset_builder=build)
        ok("preset/right_key_untouched_%s" % name, router.state == {field: "the request"},
           repr(router.state))
        laya_preset(name, {"text": "the request", "lang": "en"}, router=router, preset_builder=build)
        ok("preset/multi_key_untouched_%s" % name,
           router.state == {"text": "the request", "lang": "en"}, repr(router.state))
        laya_preset(name, {"payload": {"nested": 1}}, router=router, preset_builder=build)
        ok("preset/lone_non_string_untouched_%s" % name,
           router.state == {"payload": {"nested": 1}}, repr(router.state))
        # the questions that get answered are the preset's own, not a hand-copied set
        ok("preset/questions_forwarded_%s" % name, router.questions == build(PRESETS[name]))

    for name in sorted(PRESETS):
        ok("preset/canonical_%s" % name, validate_preset(name) == name)
    for alias, canonical in sorted(PRESET_ALIASES.items()):
        ok("preset/alias_%s" % alias, validate_preset(alias) == canonical)
        # an alias has to reach the same preset through the tool, not just the validator
        router = StateRouter()
        laya_preset(alias, {"text": "the request"}, router=router, preset_builder=build)
        ok("preset/alias_through_tool_%s" % alias,
           router.questions == build(PRESETS[canonical]), repr(router.questions))
    ok("preset/aliases_are_not_names", not (set(PRESET_ALIASES) & set(PRESETS)))

    for bad in ("nope", "", "   ", "Email", "model router", "guard_questions", 5, [], {}, None):
        expect_tool_error("preset/rejected_%r" % (bad,), lambda b=bad: validate_preset(b),
                          "invalid_preset")

    # What a client can discover without calling: the table, and the tools/list description built
    # from it. Both are derived, so neither can advertise a name the tool rejects.
    info = get_available_presets()
    ok("preset/table_covered_by_info", sorted(info) == sorted(PRESETS), repr(sorted(info)))
    for name, entry in sorted(info.items()):
        ok("preset/info_builder_%s" % name, entry["questions"] == PRESETS[name], repr(entry))
        ok("preset/info_field_%s" % name, entry.get("state_field") == fields[name], repr(entry))
    ok("preset/info_lists_aliases", info["model_router"].get("aliases") == ["router"],
       repr(info["model_router"]))
    ok("preset/info_alias_names_are_not_entries", "router" not in info, repr(sorted(info)))

    description = next(t.description for t in asyncio.run(mcp_server.list_tools())
                       if t.name == "laya_preset")
    for name in sorted(PRESETS):
        ok("preset/desc_names_%s" % name, "'%s'" % name in description, description)
    for name, field in sorted(fields.items()):
        ok("preset/desc_field_%s" % name, "'%s' reads `%s`" % (name, field) in description,
           description)
    ok("preset/desc_names_the_alias", "'router' is 'model_router'" in description, description)


def test_model_forwarding():
    """An alias reaches core canonical, and changes nothing else about the answer."""

    class EchoRouter:
        _agents = {"english": FakeAgent()}

        def __init__(self):
            self.calls = []

        def predict(self, state, questions, **kwargs):
            # No `routing` key on purpose: what the tool then reports is the name it settled on,
            # which is the piece an alias could have leaked through.
            self.calls.append(kwargs)
            return {"answers": {"department": {"choice": "billing", "confidence": 0.94}}}

    router = EchoRouter()
    by_alias = laya_predict(STATE, QUESTIONS, model="laya", router=router)
    by_name = laya_predict(STATE, QUESTIONS, model="english", router=router)
    ok("forward/model_seen_by_core", [c.get("model") for c in router.calls] == ["english", "english"],
       repr(router.calls))
    ok("forward/reports_canonical", by_alias["routing"]["model"] == "english", repr(by_alias["routing"]))
    ok("forward/answers_identical", by_alias["answers"] == by_name["answers"])



# --- shape (mocked router, no weights) ---------------------------------------

class FakeAgent:
    device = "cpu"


class FakeAgentDirect:
    """An Agent used for model='auto' with no Router present (#444).

    Agent.predict / system_one return the system_one payload but carry no
    'routing' key, because the Agent itself does no routing.
    """

    device = "cpu"

    def predict(self, state, questions, **kwargs):
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                answers[name] = {"choice": "billing", "confidence": 0.94, "probs": {"billing": 0.94}}
            elif spec["type"] == "score":
                answers[name] = {"score": 1.84, "confidence": 0.8, "distribution": [0.1, 0.3, 0.6]}
            else:
                answers[name] = {"noul": 0.892, "confidence": 0.89}
        # Note: no "routing" key -- an Agent does not route.
        return {"answers": answers}


class FakeRouter:
    _agents = {"english": FakeAgent()}

    def predict(self, state, questions, **kwargs):
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                answers[name] = {"choice": "billing", "confidence": 0.94, "probs": {"billing": 0.94}}
            elif spec["type"] == "score":
                answers[name] = {"score": 1.84, "confidence": 0.8, "distribution": [0.1, 0.3, 0.6]}
            else:
                answers[name] = {"noul": 0.892, "confidence": 0.89}
        return {"answers": answers, "routing": {"model": "english", "repo": "fake/laya", "reason": "latin script"}}

    def route(self, state, questions):
        class D:
            model = "multilingual"
            reason = "non-Latin script (devanagari)"
            repo = "fake/repo"
        return D()


class FakeRouterNoRouting(FakeRouter):
    """A router whose predict returns an Agent-shaped payload (no 'routing' key)."""

    def predict(self, state, questions, **kwargs):
        result = super().predict(state, questions, **kwargs)
        result.pop("routing", None)
        return result


def test_shape():
    out = laya_predict(STATE, QUESTIONS, model="auto", router=FakeRouter())
    ok("shape/predict_keys", set(out) >= {"answers", "routing", "latency_ms"})
    ok("shape/predict_choice", out["answers"]["department"]["choice"] == "billing")
    ok("shape/predict_noul_float", isinstance(out["answers"]["churn_risk"]["noul"], float))
    ok("shape/predict_routing", out["routing"]["model"] == "english")
    # The device is the real device of the answering checkpoint (Agent.device),
    # not a guess; it is omitted when it cannot be read.
    ok("shape/predict_device", out["device"] == "cpu", repr(out.get("device")))
    ok("shape/predict_latency_ms", isinstance(out["latency_ms"], (int, float)) and out["latency_ms"] >= 0)

    class RouterNoAgents:
        def predict(self, state, questions, **kwargs):
            return {"answers": {}, "routing": {"model": "english"}}

    out = laya_predict(STATE, QUESTIONS, model="auto", router=RouterNoAgents())
    ok("shape/predict_device_absent_when_unreadable", "device" not in out, repr(set(out)))

    out = laya_route(STATE, QUESTIONS, router=FakeRouter())
    ok("shape/route_dict", out == {"model": "multilingual", "repo": "fake/repo",
                                   "reason": "non-Latin script (devanagari)"})

    def builder(attr):
        assert attr == "triage_questions"
        return {"intent": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}

    out = laya_preset("triage", {"message": "help"}, router=FakeRouter(), preset_builder=builder)
    ok("shape/preset_answers", "intent" in out["answers"])

    out = laya_status(router=FakeRouter(), loaded=["english"], preload=True)
    ok("shape/status_ready", out["router_ready"] is True)
    ok("shape/status_loaded", out["loaded"] == ["english"])
    ok("shape/status_versions", "laya" in out["package_versions"])
    ok("shape/status_device", out["device"] == "cpu", repr(out.get("device")))
    ok("shape/status_device_is_fact", out["device_is_preference"] is False)
    ok("shape/status_checkpoint_devices", out["checkpoint_devices"] == {"english": "cpu"},
       repr(out.get("checkpoint_devices")))
    out = laya_status(router=None, loaded=None, preload=True)
    ok("shape/status_pref_before_load",
       out["device_is_preference"] is True and out["checkpoint_devices"] == {}
       and out["device"] in ("cpu", "cuda", "mps"),
       repr(out.get("device")))

    expect_tool_error("shape/predict_missing_router",
                      lambda: laya_predict(STATE, QUESTIONS, model="auto", router=None),
                      "models_not_ready")
    expect_tool_error("shape/predict_bad_questions",
                      lambda: laya_predict(STATE, {}, model="auto", router=FakeRouter()),
                      "invalid_questions")

    # #444: model='auto' with an Agent and no Router runs the Agent directly and
    # reports a null model -- 'auto' is a routing directive, not a checkpoint name.
    out = laya_predict(STATE, QUESTIONS, model="auto", agent=FakeAgentDirect())
    ok("shape/predict_auto_agent_model_null",
       out["routing"] == {"model": None, "repo": None, "reason": "auto routing without router"},
       repr(out.get("routing")))
    ok("shape/predict_auto_agent_answers", out["answers"]["department"]["choice"] == "billing")
    ok("shape/predict_auto_agent_no_device", "device" not in out, repr(set(out)))
    # auto with neither Router nor Agent still errors, as before.
    expect_tool_error("shape/predict_auto_needs_router_or_agent",
                      lambda: laya_predict(STATE, QUESTIONS, model="auto"),
                      "models_not_ready")
    # a Router payload without a 'routing' key hits the same null-model fallback.
    out = laya_predict(STATE, QUESTIONS, model="auto", router=FakeRouterNoRouting())
    ok("shape/predict_auto_fallback_model_null",
       out["routing"] == {"model": None, "repo": None, "reason": "auto routing without router"},
       repr(out.get("routing")))
    ok("shape/predict_auto_fallback_answers", out["answers"]["department"]["choice"] == "billing")
    # an explicit checkpoint still names itself in the same fallback.
    out = laya_predict(STATE, QUESTIONS, model="english", router=FakeRouterNoRouting())
    ok("shape/predict_explicit_fallback_names_checkpoint",
       out["routing"] == {"model": "english", "repo": None, "reason": "explicit model"},
       repr(out.get("routing")))


def test_question_forwarding():
    questions = {
        "intent": {"type": "choice", "instructions": "Which intent?",
                   "criteria": {"A": None, "B": {"desc": "billing"}}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", {"desc": "blocking"}]},
        "positive": {"type": "noul", "instructions": "Is this positive?",
                     "criteria": {"false": None, "true": "yes"},
                     "labels": {"false": "B", "true": "A"}},
        # `option_order` is documented for every question type (README, "Option order"); it
        # used to be dropped here, so the rotation-averaging recipe got k identical answers.
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "b", "technical": "t", "sales": "s"},
                 "option_order": [2, 0, 1]},
        "severity": {"type": "score", "instructions": "How severe?",
                     "criteria": ["minor", "major"], "option_order": [1, 0]},
        "spam": {"type": "noul", "instructions": "Is this spam?", "option_order": [1, 0]},
    }

    class CapturingRouter(FakeRouter):
        def predict(self, state, received, **kwargs):
            self.predicted_questions = received
            return super().predict(state, received, **kwargs)

        def route(self, state, received):
            self.routed_questions = received
            return super().route(state, received)

    router = CapturingRouter()
    laya_predict(STATE, questions, router=router)
    laya_route(STATE, questions, router=router)
    ok("questions/predict_preserves_supported_values", router.predicted_questions == questions)
    ok("questions/route_preserves_supported_values", router.routed_questions == questions)


def test_option_order_forwarding():
    """`option_order` reaches the answering call from every tool that takes questions."""
    order_q = {"team": {"type": "choice", "instructions": "Which team?",
                        "criteria": {"billing": "b", "technical": "t", "sales": "s"},
                        "option_order": [2, 0, 1]}}

    router = BatchRouter()
    laya_predict_batch([{"state": STATE, "questions": order_q},
                        {"state": STATE, "questions": QUESTIONS}], router=router)
    forwarded = router.predict_batch_calls[0][0]
    ok("option_order/batch_item_forwarded",
       forwarded[0]["questions"]["team"].get("option_order") == [2, 0, 1], repr(forwarded[0]))
    ok("option_order/batch_item_without_order_unchanged",
       all("option_order" not in spec for spec in forwarded[1]["questions"].values()))
    router = BatchRouter()
    bad = {"team": dict(order_q["team"], option_order=[0, 0, 0])}
    expect_tool_error("option_order/batch_item_invalid",
                      lambda: laya_predict_batch([{"state": STATE, "questions": bad}], router=router),
                      "invalid_questions")
    ok("option_order/batch_invalid_not_called", router.predict_batch_calls == [])

    # Shortlist passthrough (n <= k): the question is answered as given, so the order applies
    # and nothing was dropped.
    router = ShortlistRouter({"english": ShortlistAgent()})
    out = laya_shortlist(STATE, order_q, model="english", k=5, router=router, embed_fn=_raising_embed)
    ok("option_order/shortlist_passthrough_forwarded",
       router.seen_questions["team"].get("option_order") == [2, 0, 1], repr(router.seen_questions))
    ok("option_order/shortlist_passthrough_no_drop_signal",
       "option_order_dropped" not in out, repr(out.get("option_order_dropped")))
    # A narrowed choice is answered over its k ranked labels, so an order over all n options no
    # longer describes it. Core drops the order rather than remapping it (#1063), so the call
    # answers over the kept labels and the wrapper reports the drop per question, with the old
    # and kept option counts: core's RuntimeWarning never reaches an MCP client.
    router = ShortlistRouter({"english": ShortlistAgent()})
    narrowed = {"topic": dict(SHORTLIST_QUESTIONS["topic"], option_order=[4, 3, 2, 1, 0])}
    out = laya_shortlist(STATE, narrowed, model="english", k=2,
                         router=router, embed_fn=_tie_embed)
    ok("option_order/shortlist_narrowed_answers",
       out["answers"]["topic"]["choice"] == "billing", repr(out["answers"]))
    ok("option_order/shortlist_narrowed_order_dropped",
       "option_order" not in router.seen_questions["topic"], repr(router.seen_questions["topic"]))
    ok("option_order/shortlist_narrowed_drop_signal",
       out.get("option_order_dropped") == {"topic": {"n": 5, "kept": 2}},
       repr(out.get("option_order_dropped")))
    ok("option_order/shortlist_narrowed_caller_order_kept",
       narrowed["topic"]["option_order"] == [4, 3, 2, 1, 0], repr(narrowed["topic"]))
    # The drop is reported in the response, never printed: the MCP server speaks JSON-RPC
    # on stdout, and core's RuntimeWarning about the drop belongs on stderr.
    import contextlib
    import io

    buf = io.StringIO()
    router = ShortlistRouter({"english": ShortlistAgent()})
    with contextlib.redirect_stdout(buf):
        laya_shortlist(STATE, narrowed, model="english", k=2,
                       router=router, embed_fn=_tie_embed)
    ok("option_order/shortlist_narrowed_stdout_clean",
       buf.getvalue() == "", repr(buf.getvalue()[:100]))
    # A structurally invalid order (not a permutation) is still refused up front, before any
    # predict: only a valid order on a narrowed question reaches the drop path.
    router = ShortlistRouter({"english": ShortlistAgent()})
    bad = {"topic": dict(SHORTLIST_QUESTIONS["topic"], option_order=[0, 0, 0, 0, 0])}
    expect_tool_error("option_order/shortlist_invalid_rejected",
                      lambda: laya_shortlist(STATE, bad, model="english", k=2,
                                             router=router, embed_fn=_tie_embed),
                      "invalid_questions")
    ok("option_order/shortlist_invalid_not_called", router.seen_questions is None)
    # The signal survives the JSON boundary an MCP client reads: _wrap dumps whatever the tool
    # returns, so the drop has to come back under its real key name after json round-trip.
    import json

    from laya.mcp import server as server_mod

    router = ShortlistRouter({"english": ShortlistAgent()})
    payload = json.loads(server_mod._wrap(
        laya_shortlist, state=STATE, questions=narrowed, model="english", k=2,
        router=router, embed_fn=_tie_embed))
    ok("option_order/shortlist_narrowed_mcp_json_drop_signal",
       payload.get("option_order_dropped") == {"topic": {"n": 5, "kept": 2}},
       repr(payload.get("option_order_dropped")))
    # Non-choice questions are never narrowed, so their order is forwarded whatever k is.
    router = ShortlistRouter({"english": ShortlistAgent()})
    scored = {"urgency": dict(SHORTLIST_QUESTIONS["urgency"], option_order=[1, 0])}
    out = laya_shortlist(STATE, scored, model="english", k=1, router=router, embed_fn=_raising_embed)
    ok("option_order/shortlist_non_choice_forwarded",
       router.seen_questions["urgency"].get("option_order") == [1, 0], repr(router.seen_questions))
    ok("option_order/shortlist_non_choice_no_drop_signal",
       "option_order_dropped" not in out, repr(out.get("option_order_dropped")))


def test_real_device():
    # agent_device: the real device read from a loaded agent (no weights).
    ok("device/agent_str", agent_device(FakeAgent()) == "cpu")
    ok("device/agent_missing_attr", agent_device(object()) is None)
    ok("device/agent_none", agent_device(None) is None)

    class TorchDevice:  # torch.device-like: a .type attribute
        type = "mps"

    ok("device/agent_torch_like",
       agent_device(type("Agent", (), {"device": TorchDevice()})()) == "mps")

    # router_agent: read-only on the _agents mapping, never load() (which
    # reorders the LRU and would rebuild an evicted checkpoint).
    ok("device/router_agents", router_agent(FakeRouter(), "english") is not None)
    ok("device/router_agents_missing", router_agent(FakeRouter(), "typed-decisions") is None)
    # Aliased name, resolved with the core normaliser (English -> english).
    ok("device/router_normalised_alias", router_agent(FakeRouter(), "English") is not None)
    ok("device/router_bare", router_agent(type("Bare", (), {})(), "english") is None)
    ok("device/router_none", router_agent(None, "english") is None)

    # A router whose load() raises must still yield the right device: proof
    # that the device read never calls load().
    class RouterLoadIsASideEffect:
        _agents = {"english": FakeAgent()}

        def load(self, name):
            raise AssertionError("router_agent must not call load()")

        def predict(self, state, questions, **kwargs):
            return {"answers": {}, "routing": {"model": "english"}}

    out = laya_predict(STATE, QUESTIONS, model="auto", router=RouterLoadIsASideEffect())
    ok("device/load_never_called", out.get("device") == "cpu", repr(out.get("device")))


# --- contract on the private Router._agents name (no weights, no network) ----

def test_private_contract():
    # Why this test exists: laya.mcp.device.router_agent reads the private
    # Router._agents mapping, because it is the only side-effect-free way to
    # read a loaded agent's real device. If the core ever renames _agents,
    # the device would silently disappear from the laya_status/laya_predict
    # answers: every other test in this file uses fakes that carry their own
    # _agents attribute, so only a test built on a real Router would notice.
    # A rename must break CI loudly instead of degrading the answers silently.
    import laya

    # A real Router with preload=False (the default) loads nothing on
    # construction: no checkpoint build, no download (downloads only happen
    # inside load()/preload()). If that ever changed, this line would fail
    # here rather than on the network.
    r = laya.Router()
    ok("contract/no_download_on_construct", list(r.loaded) == [], repr(list(r.loaded)))

    class MpsDevice:  # torch.device-like: a .type attribute
        type = "mps"

    fake = type("Agent", (), {"device": MpsDevice()})()
    r.attach("english", fake)  # public API: registers under the normalised name
    ok("contract/attach_resident", list(r.loaded) == ["english"], repr(list(r.loaded)))
    ok("contract/router_agents_readable", router_agent(r, "english") is fake)
    ok("contract/agent_device_mps", agent_device(router_agent(r, "english")) == "mps")
    out = laya_status(router=r, preload=False)
    ok("contract/status_mps",
       out.get("checkpoint_devices") == {"english": "mps"}
       and out.get("device") == "mps" and out.get("device_is_preference") is False,
       repr(out.get("checkpoint_devices")))
    # attach() normalises the name, so the alias must read it back too.
    ok("contract/alias_english", router_agent(r, "English") is fake)


# --- shortlist tool (mocked router/agent, no weights) -------------------------

SHORTLIST_QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "Which topic?",
        "criteria": {
            "billing": "invoices and money",
            "shipping": "delivery status",
            "returns": "send items back",
            "account": "login and profile",
            "other": "anything else",
        },
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
}


class ShortlistAgent:
    device = "cpu"


class ShortlistDirectAgent(ShortlistAgent):
    def __init__(self):
        self.seen = None

    def predict(self, state, questions, **kwargs):
        self.seen = questions
        return {"answers": {name: {"choice": list(spec["criteria"])[0], "confidence": 0.9}
                            for name, spec in questions.items()}}


class ShortlistRouter:
    """Fake router recording what predict received (no weights, no torch)."""

    def __init__(self, agents, routed="english"):
        self._agents = dict(agents)
        self._routed = routed
        self.seen_questions = None
        self.seen_kwargs = None

    def route(self, state, questions):
        return {"model": self._routed, "repo": "fake/repo", "reason": "unit-test route"}

    def predict(self, state, questions, **kwargs):
        self.seen_questions = questions
        self.seen_kwargs = kwargs
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                labels = list(spec["criteria"])
                answers[name] = {"choice": labels[0], "confidence": 0.9,
                                 "probs": {label: round(1.0 / len(labels), 3) for label in labels}}
            elif spec["type"] == "score":
                answers[name] = {"score": 1.5, "confidence": 0.8, "distribution": [0.4, 0.6]}
            else:
                answers[name] = {"noul": 0.7, "confidence": 0.9}
        return {"answers": answers,
                "routing": {"model": kwargs.get("model", "english"), "repo": "fake/repo",
                            "reason": "explicit model"}}


class LoadingRouter(ShortlistRouter):
    """ShortlistRouter plus the on-demand load() Router.predict relies on."""

    def __init__(self, agents, routed="english"):
        super().__init__(agents, routed)
        self.load_calls = []

    def load(self, name):
        self.load_calls.append(name)
        agent = ShortlistAgent()
        self._agents[name] = agent
        return agent


def _tie_embed(texts):
    # Every text gets the zero vector, so all cosine scores tie at 0 and the
    # documented tie rule ("ties keep the earlier label") keeps the first k.
    return [[0.0, 0.0] for _ in texts]


def _raising_embed(texts):
    raise AssertionError("embed_fn must not be called when every choice passes through")


def test_shortlist():
    expect_tool_error("shortlist/state_bad",
                      lambda: laya_shortlist([], SHORTLIST_QUESTIONS), "invalid_state")
    expect_tool_error("shortlist/questions_bad",
                      lambda: laya_shortlist(STATE, {}), "invalid_questions")
    expect_tool_error("shortlist/model_bad",
                      lambda: laya_shortlist(STATE, SHORTLIST_QUESTIONS, model="gpt4"), "invalid_model")
    for bad_k in (0, -3, True, 2.5, "3"):
        expect_tool_error("shortlist/k_bad_%r" % (bad_k,),
                          lambda b=bad_k: laya_shortlist(
                              STATE, SHORTLIST_QUESTIONS, k=b,
                              router=ShortlistRouter({"english": ShortlistAgent()})),
                          "invalid_k")
    expect_tool_error("shortlist/auto_needs_router",
                      lambda: laya_shortlist(STATE, SHORTLIST_QUESTIONS), "models_not_ready")
    # Explicit model whose checkpoint is neither resident nor loadable.
    expect_tool_error("shortlist/explicit_not_loaded",
                      lambda: laya_shortlist(STATE, SHORTLIST_QUESTIONS, model="english",
                                             router=ShortlistRouter({})),
                      "models_not_ready")

    # Passthrough: every choice has <= k labels, so embed_fn is never called.
    small = {"dept": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}
    router = ShortlistRouter({"english": ShortlistAgent()})
    out = laya_shortlist(STATE, small, model="english", k=5, router=router, embed_fn=_raising_embed)
    meta = out["shortlist"]["dept"]
    ok("shortlist/passthrough_meta", meta["passthrough"] is True and meta["scores"] is None
       and meta["k"] == 5 and meta["n"] == 2, repr(meta))
    ok("shortlist/passthrough_labels", meta["labels"] == ["a", "b"], repr(meta["labels"]))
    ok("shortlist/passthrough_answers", out["answers"]["dept"]["choice"] == "a", repr(out["answers"]))
    ok("shortlist/passthrough_forwarded", list(router.seen_questions["dept"]["criteria"]) == ["a", "b"])
    ok("shortlist/passthrough_device", out.get("device") == "cpu", repr(out.get("device")))
    ok("shortlist/passthrough_routing",
       out["routing"] == {"model": "english", "repo": None, "reason": "explicit model"},
       repr(out["routing"]))
    ok("shortlist/passthrough_latency", isinstance(out["latency_ms"], float))

    # Default k comes from laya.shortlist: a choice smaller than it passes through.
    # Read from the library rather than copied, so raising DEFAULT_SHORTLIST_K cannot
    # make this assert a stale value against a tool layer that moved correctly.
    from laya.shortlist import DEFAULT_SHORTLIST_K

    router = ShortlistRouter({"english": ShortlistAgent()})
    three = {"dept": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C"}}}
    out = laya_shortlist(STATE, three, model="english", router=router, embed_fn=_raising_embed)
    ok("shortlist/default_k_passthrough", out["shortlist"]["dept"]["k"] == DEFAULT_SHORTLIST_K
       and out["shortlist"]["dept"]["passthrough"] is True, repr(out["shortlist"]))

    # Shortlist path: 5 options with k=2 -> predict sees exactly the kept labels.
    calls = []

    def recording_embed(texts):
        calls.append(list(texts))
        return _tie_embed(texts)

    router = ShortlistRouter({"english": ShortlistAgent()})
    out = laya_shortlist(STATE, SHORTLIST_QUESTIONS, model="english", k=2,
                         router=router, embed_fn=recording_embed)
    meta = out["shortlist"]["topic"]
    ok("shortlist/meta_shape", meta["k"] == 2 and meta["n"] == 5 and meta["passthrough"] is False,
       repr(meta))
    ok("shortlist/meta_labels_tie_order", meta["labels"] == ["billing", "shipping"], repr(meta["labels"]))
    ok("shortlist/meta_scores", meta["scores"] == [0.0, 0.0], repr(meta["scores"]))
    ok("shortlist/predict_saw_reduced",
       list(router.seen_questions["topic"]["criteria"]) == ["billing", "shipping"],
       repr(router.seen_questions["topic"]))
    ok("shortlist/embed_called_once", len(calls) == 1 and len(calls[0]) == 6,
       repr([len(c) for c in calls]))
    # Non-choice questions are forwarded unchanged and get no shortlist entry.
    ok("shortlist/non_choice_forwarded", router.seen_questions["urgency"] == SHORTLIST_QUESTIONS["urgency"])
    ok("shortlist/non_choice_no_meta", "urgency" not in out["shortlist"], repr(sorted(out["shortlist"])))
    ok("shortlist/input_not_mutated", len(SHORTLIST_QUESTIONS["topic"]["criteria"]) == 5)
    # No order was supplied, so there is nothing to drop and no signal in the response.
    ok("shortlist/no_order_no_drop_signal",
       "option_order_dropped" not in out, repr(out.get("option_order_dropped")))

    # Auto mode: route once, then answer with an explicit model= so the forward
    # pass does not re-route; the reported routing is the real route decision.
    router = ShortlistRouter({"multilingual": ShortlistAgent()}, routed="multilingual")
    out = laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, router=router, embed_fn=_tie_embed)
    ok("shortlist/auto_routing_reported",
       out["routing"] == {"model": "multilingual", "repo": "fake/repo", "reason": "unit-test route"},
       repr(out["routing"]))
    ok("shortlist/auto_explicit_model", router.seen_kwargs == {"model": "multilingual"},
       repr(router.seen_kwargs))

    # Lazy server (LAYA_PRELOAD=0): a routed checkpoint that is not resident is
    # loaded on demand, the same on-demand build Router.predict performs.
    router = LoadingRouter({}, routed="multilingual")
    out = laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, router=router, embed_fn=_tie_embed)
    ok("shortlist/lazy_load_once", router.load_calls == ["multilingual"], repr(router.load_calls))
    ok("shortlist/lazy_device", out.get("device") == "cpu", repr(out.get("device")))

    # Direct agent injection (no router), mirroring laya_predict's agent= path.
    agent = ShortlistDirectAgent()
    out = laya_shortlist(STATE, small, model="english", k=5, agent=agent, embed_fn=_raising_embed)
    ok("shortlist/direct_agent", out["answers"]["dept"]["choice"] == "a"
       and out["routing"]["reason"] == "explicit model", repr(out["routing"]))


# --- batch tools (mocked router, no weights) ----------------------------------

class BatchRouter(FakeRouter):
    """Records each batch call; returns one result per request, input order."""

    def __init__(self):
        self.predict_batch_calls = []
        self.route_batch_calls = []

    def _answer_for(self, request):
        answers = {}
        for name, spec in request["questions"].items():
            if spec["type"] == "choice":
                answers[name] = {"choice": "billing", "confidence": 0.9}
            elif spec["type"] == "score":
                answers[name] = {"score": 1.5, "confidence": 0.8}
            else:
                answers[name] = {"noul": 0.7, "confidence": 0.9}
        return {"answers": answers,
                "routing": {"model": request.get("model") or "english",
                            "repo": "fake/laya", "reason": "batch route"}}

    def predict_batch(self, requests, batch_size=None, **kwargs):
        self.predict_batch_calls.append((list(requests), batch_size, kwargs))
        return [self._answer_for(request) for request in requests]

    def route_batch(self, requests, hooks_timeout=None):
        self.route_batch_calls.append((list(requests), hooks_timeout))
        return [{"model": request.get("model") or "english", "repo": "fake/laya",
                 "reason": "batch route"} for request in requests]


class ShortRouter:
    """Router whose batches misreport their own size."""

    def predict_batch(self, requests, batch_size=None):
        return []

    def route_batch(self, requests, hooks_timeout=None):
        return []


class StrictRouteRouter:
    """A Router stub that predates the hooks_timeout kwarg on route_batch.

    laya_route_batch must send `route_batch(items)` when the caller did not
    ask for a hook deadline -- sending `route_batch(items, hooks_timeout=None)`
    would break every caller that attached a pre-#577 Router (or a stub like
    this one). This is the witness for M2 in the PR's mutation harness.
    """

    def route_batch(self, requests):
        return [{"model": "english", "repo": "fake/laya",
                 "reason": "strict route"} for _ in requests]


BATCH_REQUESTS = [
    {"state": {"body": "refund please"}, "questions": QUESTIONS},
    {"state": {"body": "please refund invoice 2"}, "questions": QUESTIONS,
     "model": "english", "task": "massive", "lang": "en"},
    {"state": {"body": "mera account double charge hua"},
     "questions": {"triage": {"type": "noul", "instructions": "Leaving?"}},
     "lang": "hi"},
]


def test_batch_validation():
    for bad in (None, {}, "x", {"state": STATE, "questions": QUESTIONS}):
        expect_tool_error("batch/requests_bad_%r" % (type(bad).__name__,),
                          lambda b=bad: validate_batch_requests(b), "invalid_request")
    expect_tool_error("batch/requests_empty",
                      lambda: validate_batch_requests([]), "invalid_request")
    expect_tool_error("batch/item_not_object",
                      lambda: validate_batch_requests(["x"]), "invalid_request")
    expect_tool_error("batch/item_missing_state",
                      lambda: validate_batch_requests([{"questions": QUESTIONS}]), "invalid_state")
    expect_tool_error("batch/item_missing_questions",
                      lambda: validate_batch_requests([{"state": STATE}]), "invalid_questions")
    expect_tool_error("batch/item_bad_questions",
                      lambda: validate_batch_requests([{"state": STATE, "questions": {}}]),
                      "invalid_questions")
    expect_tool_error("batch/item_bad_model",
                      lambda: validate_batch_requests(
                          [{"state": STATE, "questions": QUESTIONS, "model": "gpt4"}]),
                      "invalid_model")
    for key, value in (("task", 3), ("lang", ""), ("lang", [])):
        expect_tool_error("batch/item_bad_%s" % key,
                          lambda k=key, v=value: validate_batch_requests(
                              [{"state": STATE, "questions": QUESTIONS, k: v}]),
                          "invalid_request")

    out = validate_batch_requests(BATCH_REQUESTS)
    ok("batch/validation_preserves_order", [item["state"] for item in out]
       == [request["state"] for request in BATCH_REQUESTS])
    ok("batch/validation_keeps_overrides",
       out[1].get("model") == "english" and out[1].get("task") == "massive"
       and out[1].get("lang") == "en" and out[2].get("lang") == "hi", repr(out[1]))
    # "auto" is the same thing as absent: Router.route resolves the checkpoint.
    out = validate_batch_requests([{"state": STATE, "questions": QUESTIONS, "model": "auto"}])
    ok("batch/validation_auto_dropped", "model" not in out[0], repr(out[0]))
    # Keys the Router would never expect are dropped, not forwarded.
    out = validate_batch_requests([{"state": STATE, "questions": QUESTIONS, "temperature": 0}])
    ok("batch/validation_unknown_key_dropped", set(out[0]) == {"state", "questions"}, repr(out[0]))
    # The two token budgets are per-request overrides Router.predict_batch reads off the item,
    # so they survive validation rather than being dropped with the unknown keys above.
    out = validate_batch_requests([{"state": STATE, "questions": QUESTIONS,
                                   "max_len": 1024, "head_max_len": 512}])
    ok("batch/validation_keeps_budgets", out[0].get("max_len") == 1024
       and out[0].get("head_max_len") == 512, repr(out[0]))
    # An explicit null means "keep the checkpoint's default", the same as leaving it out.
    out = validate_batch_requests([{"state": STATE, "questions": QUESTIONS, "max_len": None}])
    ok("batch/validation_null_budget_dropped", "max_len" not in out[0], repr(out[0]))
    for key, bad in (("max_len", 0), ("max_len", "1024"), ("max_len", 2.5), ("max_len", True),
                     ("head_max_len", 0), ("head_max_len", -1), ("head_max_len", "512")):
        expect_tool_error("batch/item_bad_%s_%r" % (key, bad),
                          lambda k=key, v=bad: validate_batch_requests(
                              [{"state": STATE, "questions": QUESTIONS, k: v}]),
                          "invalid_%s" % key)
    # The message that lists the accepted keys must list the ones the validator keeps --
    # a client reads that list as the schema. Both directions, derived from the validator.
    probe = {"state": STATE, "questions": QUESTIONS, "model": "english", "task": "massive",
             "lang": "en", "lang_guess": "de", "max_len": 5, "head_max_len": 5}
    kept = set(validate_batch_requests([probe])[0])
    ok("batch/validation_covers_every_override",
       kept == {"state", "questions"} | set(BATCH_ITEM_OVERRIDES), repr(sorted(kept)))
    message = ""
    try:
        validate_batch_requests([])
    except ToolError as exc:
        message = str(exc)
    listed = {part.strip().rstrip("?") for part in
              message[message.index("{") + 1:message.index("}")].split(",")}
    ok("batch/error_message_matches_validator", listed == kept,
       "listed=%r kept=%r" % (sorted(listed), sorted(kept)))


def test_batch_predict():
    expect_tool_error("batch/predict_no_router",
                      lambda: laya_predict_batch(BATCH_REQUESTS, router=None), "models_not_ready")
    expect_tool_error("batch/predict_no_method",
                      lambda: laya_predict_batch(BATCH_REQUESTS, router=FakeRouter()),
                      "internal_error")
    for bad_size in (True, 0, -2, 2.5, "3"):
        expect_tool_error("batch/predict_bad_size_%r" % (bad_size,),
                          lambda s=bad_size: laya_predict_batch(
                              BATCH_REQUESTS, batch_size=s, router=BatchRouter()),
                          "invalid_batch_size")
    # Validation runs before the router is touched: one bad item, no partial batch.
    router = BatchRouter()
    expect_tool_error("batch/predict_validates_first",
                      lambda: laya_predict_batch([{"state": STATE}, {"state": STATE, "questions": QUESTIONS}],
                                                 router=router),
                      "invalid_questions")
    ok("batch/predict_not_called_on_bad_input", router.predict_batch_calls == [])

    # A batch item's lang_guess is type-checked like the single-request path, not passed through
    # raw -- a non-string otherwise reached core and silently misrouted the item.
    router = BatchRouter()
    expect_tool_error("batch/predict_item_lang_guess_type",
                      lambda: laya_predict_batch([{"state": STATE, "questions": QUESTIONS,
                                                   "lang_guess": {"not": "a string"}}], router=router),
                      "invalid_lang_guess")
    ok("batch/predict_bad_lang_guess_not_called", router.predict_batch_calls == [])
    router = BatchRouter()
    laya_predict_batch([{"state": STATE, "questions": QUESTIONS, "lang_guess": "de"}], router=router)
    ok("batch/predict_item_lang_guess_forwarded",
       router.predict_batch_calls[0][0][0].get("lang_guess") == "de",
       repr(router.predict_batch_calls[0][0][0]))

    router = BatchRouter()
    out = laya_predict_batch(BATCH_REQUESTS, batch_size=8, router=router)
    ok("batch/predict_one_call", len(router.predict_batch_calls) == 1,
       repr(len(router.predict_batch_calls)))
    forwarded, size, kwargs = router.predict_batch_calls[0]
    ok("batch/predict_size_forwarded", size == 8, repr(size))
    ok("batch/predict_items_forwarded", [item["state"] for item in forwarded]
       == [request["state"] for request in BATCH_REQUESTS])
    # A request's own token budget must reach the forward pass: `Router.predict_batch` reads
    # max_len/head_max_len off each item and splits items that ask for different budgets into
    # separate calls, so an item that loses them is answered at the checkpoint's default window.
    router = BatchRouter()
    laya_predict_batch([{"state": STATE, "questions": QUESTIONS,
                         "max_len": 1024, "head_max_len": 512},
                        {"state": {"body": "no budget asked for"}, "questions": QUESTIONS}],
                       router=router)
    forwarded = router.predict_batch_calls[0][0]
    ok("batch/predict_budget_forwarded",
       forwarded[0].get("max_len") == 1024 and forwarded[0].get("head_max_len") == 512,
       repr({k: v for k, v in forwarded[0].items() if k not in ("state", "questions")}))
    ok("batch/predict_budget_absent_when_unset",
       "max_len" not in forwarded[1] and "head_max_len" not in forwarded[1],
       repr(sorted(forwarded[1])))

    ok("batch/predict_keys", set(out) == {"requests", "model_counts",
                                          "total_latency_ms", "per_request_latency_ms"}, repr(sorted(out)))
    ok("batch/predict_input_order", [entry["answers"].get("triage", {}).get("noul")
                                     for entry in out["requests"]] == [None, None, 0.7],
       repr(out["requests"]))
    ok("batch/predict_department", out["requests"][0]["answers"]["department"]["choice"] == "billing")
    ok("batch/predict_routing", out["requests"][1]["routing"]["model"] == "english")
    ok("batch/predict_device_resident", out["requests"][0].get("device") == "cpu",
       repr(out["requests"][0].get("device")))
    ok("batch/predict_counts", out["model_counts"] == {"english": 3}, repr(out["model_counts"]))
    ok("batch/predict_latency", out["total_latency_ms"] >= 0
       and abs(out["per_request_latency_ms"] - out["total_latency_ms"] / 3) < 0.01,
       repr(out["per_request_latency_ms"]))

    # batch_size unset must not be forwarded as None (strict old stubs included).
    router = BatchRouter()
    laya_predict_batch(BATCH_REQUESTS, router=router)
    _, size, _ = router.predict_batch_calls[0]
    ok("batch/predict_size_default", size is None)

    # Test that hooks_timeout parameter is forwarded correctly
    router = BatchRouter()
    out = laya_predict_batch(BATCH_REQUESTS, hooks_timeout=5.5, router=router)
    _, kwargs = router.predict_batch_calls[0][1], router.predict_batch_calls[0][2]
    ok("batch/hooks_timeout_forwarded", kwargs.get("hooks_timeout") == 5.5, repr(kwargs))
    
    # Test that min_confidence parameter is forwarded correctly  
    router = BatchRouter()
    out = laya_predict_batch(BATCH_REQUESTS, min_confidence=0.75, router=router)
    _, kwargs = router.predict_batch_calls[0][1], router.predict_batch_calls[0][2]
    ok("batch/min_confidence_forwarded", kwargs.get("min_confidence") == 0.75, repr(kwargs))
    
    # Test that sort_by_length parameter is forwarded correctly
    router = BatchRouter()
    out = laya_predict_batch(BATCH_REQUESTS, sort_by_length=True, router=router)
    _, kwargs = router.predict_batch_calls[0][1], router.predict_batch_calls[0][2]
    ok("batch/sort_by_length_forwarded", kwargs.get("sort_by_length") is True, repr(kwargs))
    
    # Test that defaults don't pollute kwargs
    router = BatchRouter()
    out = laya_predict_batch(BATCH_REQUESTS, router=router)
    _, kwargs = router.predict_batch_calls[0][1], router.predict_batch_calls[0][2]
    ok("batch/defaults_not_in_kwargs", "hooks_timeout" not in kwargs and 
       "min_confidence" not in kwargs and "sort_by_length" not in kwargs, repr(kwargs))
    
    # Call-level controls are validated up front like every sibling tool: a bad value is a clean
    # ToolError, not an internal_error that escaped the TypeError-only handler, and not a silent
    # 1-second hook deadline from core's float(True) == 1.0.
    expect_tool_error("batch/min_confidence_out_of_range",
                      lambda: laya_predict_batch(BATCH_REQUESTS, min_confidence=1.5, router=BatchRouter()),
                      "invalid_min_confidence")
    expect_tool_error("batch/hooks_timeout_bool_rejected",
                      lambda: laya_predict_batch(BATCH_REQUESTS, hooks_timeout=True, router=BatchRouter()),
                      "invalid_hooks_timeout")
    # min_confidence=0.0 is a real gate (abstain over nothing) and must still be forwarded.
    router = BatchRouter()
    laya_predict_batch(BATCH_REQUESTS, min_confidence=0.0, router=router)
    _, kwargs = router.predict_batch_calls[0][1], router.predict_batch_calls[0][2]
    ok("batch/min_confidence_zero_forwarded", kwargs.get("min_confidence") == 0.0, repr(kwargs))

    expect_tool_error("batch/predict_count_mismatch",
                      lambda: laya_predict_batch(BATCH_REQUESTS, router=ShortRouter()),
                      "internal_error")


def test_batch_route():
    expect_tool_error("batch/route_no_router",
                      lambda: laya_route_batch(BATCH_REQUESTS, router=None), "models_not_ready")
    expect_tool_error("batch/route_no_method",
                      lambda: laya_route_batch(BATCH_REQUESTS, router=FakeRouter()),
                      "internal_error")
    router = BatchRouter()
    out = laya_route_batch(BATCH_REQUESTS, router=router)
    ok("batch/route_one_call", len(router.route_batch_calls) == 1)
    ok("batch/route_decisions", len(out["decisions"]) == 3
       and set(out["decisions"][0]) == {"model", "repo", "reason"}, repr(out["decisions"][0]))
    ok("batch/route_counts", out["model_counts"] == {"english": 3}, repr(out["model_counts"]))
    # Route-only never predicts.
    ok("batch/route_no_forward", router.predict_batch_calls == [])
    # Unset must not shadow the Router's own timeout: the tool layer only
    # forwards when the caller actually asked for one.
    ok("batch/route_hooks_timeout_default_none",
       router.route_batch_calls[0][1] is None, repr(router.route_batch_calls[0]))
    # The unset case must ALSO work against a Router that has never heard of
    # hooks_timeout: sending `route_batch(items, hooks_timeout=None)` would
    # break every caller with an older or hand-rolled router. Wrapped so the
    # TypeError shows as a NAMED FAIL rather than crashing the whole suite.
    try:
        strict_out = laya_route_batch(BATCH_REQUESTS, router=StrictRouteRouter())
        strict_ok = (len(strict_out["decisions"]) == 3
                     and strict_out["model_counts"] == {"english": 3})
        detail = repr(strict_out)
    except TypeError as exc:
        strict_ok = False
        detail = "TypeError: %s" % exc
    ok("batch/route_strict_router_no_kwarg", strict_ok, detail)
    # hooks_timeout is forwarded verbatim (post-validate_timeout) to
    # Router.route_batch's per-call override of the operator-installed hook
    # deadline. Same shape as laya_predict_batch (see #766).
    router = BatchRouter()
    laya_route_batch(BATCH_REQUESTS, hooks_timeout=2.5, router=router)
    _, timeout = router.route_batch_calls[0]
    ok("batch/route_hooks_timeout_forwarded", timeout == 2.5, repr(timeout))
    # An int is a valid timeout -- validate_timeout coerces to float.
    router = BatchRouter()
    laya_route_batch(BATCH_REQUESTS, hooks_timeout=1, router=router)
    _, timeout = router.route_batch_calls[0]
    ok("batch/route_hooks_timeout_int_coerced", timeout == 1.0 and isinstance(timeout, float),
       repr(timeout))
    # Bad values become invalid_hooks_timeout, not internal_error: the
    # boundary check exists so a caller gets a real code back over MCP.
    for bad in (0, -1, -0.5, "soon", True, [1.0]):
        expect_tool_error("batch/route_bad_hooks_timeout_%r" % (bad,),
                          lambda b=bad: laya_route_batch(
                              BATCH_REQUESTS, hooks_timeout=b, router=BatchRouter()),
                          "invalid_hooks_timeout")
    # Validation happens before the router is touched: one bad timeout, no
    # partial dispatch.
    router = BatchRouter()
    expect_tool_error("batch/route_bad_hooks_timeout_zero_router_untouched",
                      lambda: laya_route_batch(BATCH_REQUESTS, hooks_timeout=0, router=router),
                      "invalid_hooks_timeout")
    ok("batch/route_no_dispatch_on_bad_timeout", router.route_batch_calls == [])
    expect_tool_error("batch/route_count_mismatch",
                      lambda: laya_route_batch(BATCH_REQUESTS, router=ShortRouter()),
                      "internal_error")


# --- decide tool (mocked router, no weights) ----------------------------------

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "department": {"enum": ["billing", "support", "sales"],
                       "description": "Which team should handle this ticket?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
}


class TypeEchoRouter(FakeRouter):
    """FakeRouter that also answers with the real payload shape: the `type`
    key and a probabilities dict keyed by option/level, like system_one does,
    so laya.structured's per-field probabilities are exercised."""

    def predict(self, state, questions, **kwargs):
        out = super().predict(state, questions, **kwargs)
        for name, spec in questions.items():
            answer = out["answers"][name]
            answer["type"] = spec["type"]
            if spec["type"] == "score":
                answer["probabilities"] = {"0": 0.1, "1": 0.2, "2": 0.7}
        return out


def test_decide():
    import laya

    router = TypeEchoRouter()
    out = laya_decide(STATE, DECIDE_SCHEMA, router=router)
    ok("decide/keys", set(out) >= {"values", "confidence", "probabilities", "routing", "latency_ms"},
       repr(sorted(out)))
    ok("decide/values_choice", out["values"]["department"] == "billing", repr(out["values"]))
    ok("decide/values_score_level", out["values"]["urgency"] == 2, repr(out["values"]))
    ok("decide/values_bool", out["values"]["needs_human"] is True, repr(out["values"]))
    ok("decide/confidence", set(out["confidence"]) == {"department", "urgency", "needs_human"}
       and out["confidence"]["department"] == 0.94, repr(out["confidence"]))
    ok("decide/probs_noul_pair", set(out["probabilities"]["needs_human"]) == {"false", "true"}
       and out["probabilities"]["needs_human"]["true"] == 0.892,
       repr(out["probabilities"]["needs_human"]))
    ok("decide/probs_score_offset", set(out["probabilities"]["urgency"]) >= {"0", "1", "2"},
       repr(out["probabilities"]["urgency"]))
    ok("decide/score_level_from_probs", out["values"]["urgency"] == 2, repr(out["values"]))
    ok("decide/device", out.get("device") == "cpu", repr(out.get("device")))
    ok("decide/routing", out["routing"]["model"] == "english")

    # Same values the core decide() returns for the same runner: the tool is a
    # thin MCP surface over the documented schema projection, not a second one.
    core = laya.decide(router, STATE, schema=DECIDE_SCHEMA)
    ok("decide/parity_with_core", out["values"] == core, "tool=%r core=%r" % (out["values"], core))

    # Score projection follows the argmax of the distribution, offset by minimum.
    class LowUrgencyRouter(TypeEchoRouter):
        def predict(self, state, questions, **kwargs):
            out = super().predict(state, questions, **kwargs)
            out["answers"]["urgency"] = {"type": "score", "score": 2.0, "confidence": 0.6,
                                         "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1}}
            return out

    out = laya_decide(STATE, DECIDE_SCHEMA, router=LowUrgencyRouter())
    ok("decide/score_argmax", out["values"]["urgency"] == 0, repr(out["values"]["urgency"]))

    expect_tool_error("decide/bad_state", lambda: laya_decide({}, DECIDE_SCHEMA, router=router),
                      "invalid_state")
    for bad, why in (({"type": "object"}, "no properties"),
                     ({"type": "object", "properties": {"x": {"type": "string"}}}, "free string"),
                     ({"type": "object", "properties": {"x": {"type": "array"}}}, "array"),
                     ({}, "empty"), ([], "not an object")):
        expect_tool_error("decide/bad_schema_%s" % why,
                          lambda b=bad: laya_decide(STATE, b, router=router), "invalid_schema")
    expect_tool_error("decide/bad_model",
                      lambda: laya_decide(STATE, DECIDE_SCHEMA, model="gpt4", router=router),
                      "invalid_model")
    expect_tool_error("decide/no_router",
                      lambda: laya_decide(STATE, DECIDE_SCHEMA), "models_not_ready")
    # Explicit model with a direct agent: answered by the agent, no Router needed.
    # A bare Agent payload carries no routing, which is when the tool synthesises
    # the "explicit model" routing entry (laya_predict does the same).
    class BareAgent:
        device = "cpu"

        def predict(self, state, questions, **kwargs):
            answers = {}
            for name, spec in questions.items():
                if spec["type"] == "choice":
                    answers[name] = {"type": "choice", "choice": "billing", "confidence": 0.9,
                                     "probabilities": {"billing": 0.9, "support": 0.1}}
                elif spec["type"] == "score":
                    answers[name] = {"type": "score", "score": 1.0, "confidence": 0.6,
                                     "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}}
                else:
                    answers[name] = {"type": "noul", "noul": 0.2, "confidence": 0.8}
            return {"answers": answers}

    out = laya_decide(STATE, DECIDE_SCHEMA, model="english", agent=BareAgent())
    ok("decide/direct_agent", out["values"]["department"] == "billing"
       and out["routing"] == {"model": "english", "repo": None, "reason": "explicit model"},
       repr(out["routing"]))
    ok("decide/direct_agent_device", out.get("device") == "cpu", repr(out.get("device")))


def test_shortlist_lang_parity():
    """Auto mode must forward the routed language, so per-language temperatures
    behave as they do in laya_predict. Passing an explicit model= makes
    Router.predict re-route with that model, and an explicit-model decision
    carries no detection -- so the language is lost unless shortlist passes it on.
    """
    class DetectingRouter(ShortlistRouter):
        def route(self, state, questions):
            return {
                "model": "multilingual",
                "repo": "fake/repo",
                "reason": "Latin script but language looks like 'de'",
                "detection": {"language": "de", "script": "Latin"},
            }

    router = DetectingRouter({"multilingual": ShortlistAgent()}, routed="multilingual")
    out = laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, router=router, embed_fn=_tie_embed)
    ok("shortlist_lang/auto_forwards_lang", (router.seen_kwargs or {}).get("lang") == "de",
       repr(router.seen_kwargs))
    ok("shortlist_lang/auto_still_explicit_model",
       (router.seen_kwargs or {}).get("model") == "multilingual", repr(router.seen_kwargs))
    ok("shortlist_lang/auto_routes_once",
       out["routing"]["model"] == "multilingual", repr(out["routing"]))

    # A decision without a usable language must not invent one.
    plain = ShortlistRouter({"english": ShortlistAgent()}, routed="english")
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, router=plain, embed_fn=_tie_embed)
    ok("shortlist_lang/no_detection_adds_nothing", plain.seen_kwargs == {"model": "english"},
       repr(plain.seen_kwargs))

    # Explicit-model mode is unchanged: no routing, so no language to forward.
    agent = ShortlistDirectAgent()
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="multilingual", agent=agent,
                   embed_fn=_tie_embed)
    ok("shortlist_lang/explicit_model_untouched", agent.seen is not None)


def test_shortlist_embed_cache():
    """A fixed option list must not be re-embedded on every call.

    Uses the answering checkpoint's own encoder (no injected embed_fn) through a
    weight-free fake tokenizer/encoder, so the number of tokenizer batches is a
    direct count of embedding work.
    """
    import torch

    class _Out:
        def __init__(self, hidden):
            self.last_hidden_state = hidden

    class CountingEncoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids=None, attention_mask=None):
            self.calls += 1
            ids = input_ids.float()
            return _Out(torch.stack([ids, torch.ones_like(ids)], dim=-1))

    class CacheTok:
        def __init__(self):
            self.batches = []

        def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            self.batches.append(list(texts))
            rows = [[(ord(ch) % 5) + 1 for ch in t][:max_length] or [1] for t in texts]
            width = max(len(r) for r in rows)
            ids, mask = [], []
            for row in rows:
                pad = width - len(row)
                ids.append(row + [0] * pad)
                mask.append([1] * len(row) + [0] * pad)
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long)}

    def make_agent():
        enc = CountingEncoder()
        tok = CacheTok()
        agent = ShortlistAgent()
        agent.tok = tok
        agent.encoder = enc
        agent.model = type("M", (), {"encoder": enc})()
        agent.device = torch.device("cpu")
        return agent

    # Same checkpoint, same option list: the second call re-embeds only the query.
    # Count embedded texts, not tokenizer batches: the whole set fits in one batch
    # either way, so batch count cannot tell the two cases apart.
    agent = make_agent()
    router = ShortlistRouter({"english": agent})

    def texts_since(mark):
        return sum(len(b) for b in agent.tok.batches[mark:])

    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="english", router=router)
    first_texts = texts_since(0)
    ok("shortlist_cache/first_call_embeds_query_and_options", first_texts == 6,
       "texts=%d batches=%r" % (first_texts, agent.tok.batches))

    # An identical repeat call embeds nothing at all: the query text is the same
    # too, and cached_embed_fn matches on exact strings.
    mark = len(agent.tok.batches)
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="english", router=router)
    ok("shortlist_cache/identical_repeat_embeds_nothing", texts_since(mark) == 0,
       "texts=%d batches=%r" % (texts_since(mark), agent.tok.batches[mark:]))

    # A different state is a new query, so a repeat with that state embeds the
    # query alone and reuses the five cached option rows.
    mark = len(agent.tok.batches)
    laya_shortlist({"text": "a completely different customer message"},
                   SHORTLIST_QUESTIONS, k=2, model="english", router=router)
    ok("shortlist_cache/new_state_still_embedded", texts_since(mark) == 1,
       "texts=%d batches=%r" % (texts_since(mark), agent.tok.batches[mark:]))

    # A different answering checkpoint must not share another one's embeddings.
    other = make_agent()
    other_router = ShortlistRouter({"multilingual": other})
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="multilingual", router=other_router)
    ok("shortlist_cache/other_checkpoint_embeds_its_own",
       sum(len(b) for b in other.tok.batches) == 6,
       "texts=%d" % sum(len(b) for b in other.tok.batches))
    ok("shortlist_cache/other_checkpoint_encoder_used", other.encoder.calls == 1,
       "calls=%d" % other.encoder.calls)

    # Replacing the agent (a reload) starts a fresh cache rather than reusing rows.
    reloaded = make_agent()
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="english",
                   router=ShortlistRouter({"english": reloaded}))
    ok("shortlist_cache/reloaded_agent_re_embeds",
       sum(len(b) for b in reloaded.tok.batches) == 6,
       "texts=%d" % sum(len(b) for b in reloaded.tok.batches))
    ok("shortlist_cache/other_checkpoint_still_cached",
       other.encoder.calls == 1, "calls=%d" % other.encoder.calls)

    # An injected embed_fn stays the caller's own, uncached.
    calls = []

    def counting_embed(texts):
        calls.append(list(texts))
        return [[0.0, 0.0] for _ in texts]

    inj_router = ShortlistRouter({"english": make_agent()})
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="english", router=inj_router,
                   embed_fn=counting_embed)
    laya_shortlist(STATE, SHORTLIST_QUESTIONS, k=2, model="english", router=inj_router,
                   embed_fn=counting_embed)
    ok("shortlist_cache/injected_embed_not_wrapped", len(calls) == 2,
       "calls=%d" % len(calls))


# --- per-call controls: task / lang / max_len / head_max_len -------------------

class ControlRouter:
    """A router that records which keywords the tools handed to core.

    `route` and `predict` both take **kwargs on purpose: what these checks read is which keywords
    arrive, and an absent one has to be distinguishable from a `None` one -- which is exactly what
    `_overrides` promises.
    """

    def __init__(self, routed="english"):
        self._agents = {routed: ShortlistAgent()}
        self.routed = routed
        self.predict_calls = []
        self.predict_questions = []
        self.route_calls = []

    def route(self, state, questions, **kwargs):
        self.route_calls.append(kwargs)
        return {"model": self.routed, "repo": "fake/repo",
                "reason": "unit-test route %s" % sorted(kwargs)}

    def predict(self, state, questions, **kwargs):
        self.predict_calls.append(kwargs)
        self.predict_questions.append(questions)
        return {"answers": {"department": {"choice": "billing", "confidence": 0.94}},
                "routing": {"model": kwargs.get("model", self.routed), "repo": "fake/repo",
                            "reason": "unit-test predict %s" % sorted(kwargs)}}

    def load(self, name):
        return ShortlistAgent()


class ControlAgent:
    """An ``agent=`` injection: it answers, and it remembers being asked."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append(kwargs)
        return {"answers": {"department": {"choice": "billing", "confidence": 0.94}}}


def test_controls_validation():
    """Each control is checked the way core checks it, and refused in this layer's words."""
    from laya.router import DEFAULT_MODELS, _ALIASES

    ok("task/none_is_unset", validate_task(None) is None)
    for name in sorted(DEFAULT_MODELS):
        ok("task/canonical_%s" % name, validate_task(name) == name)
    for alias in sorted(_ALIASES):
        # Accepted because core accepts it, and forwarded as the caller wrote it: `Router._route`
        # normalises the task itself, and its `reason` quotes the spelling that was asked for, so
        # canonicalising here would rewrite the explanation the caller reads back.
        ok("task/alias_%s" % alias, validate_task(alias) == alias, repr(validate_task(alias)))
    # What the sweep really protects: every name core's registry holds is accepted here, so this
    # layer can never be narrower than `router.predict(task=...)` the way a copied list was.
    from laya.router import normalise_name

    for alias in sorted(_ALIASES):
        ok("task/resolves_%s" % alias, normalise_name(validate_task(alias)) == _ALIASES[alias],
           "%s -> %s" % (alias, normalise_name(validate_task(alias))))
    # The underscore form is what the CLI and the question ids use, and `Router._route` remaps it
    # before it looks anything up. The tool remaps the same way but forwards the caller's own
    # spelling, so `routing.reason` still reads as the request that was made.
    for spelling in ("typed_decisions", "TYPED_DECISIONS", " typed_decisions ", "typed-decisions"):
        got = validate_task(spelling)
        ok("task/workflow_%r" % spelling, got == spelling, repr(got))
    for bad in ("nope", "", "   ", "english-ish", 5, [], {}, True):
        expect_tool_error("task/rejected_%r" % (bad,), lambda b=bad: validate_task(b), "invalid_task")
    try:
        validate_task("nope")
        message = ""
    except ToolError as exc:
        message = exc.message
    ok("task/error_lists_checkpoints", all(n in message for n in sorted(DEFAULT_MODELS)),
       repr(message))

    ok("lang/none_is_unset", validate_lang(None) is None)
    # Verbatim, including the codes that mean nothing to core: a blank lang falls through to
    # detection rather than failing, and whether a code names a language is core's question.
    for code in ("de", "en-US", "pt", "en_US.UTF-8", "", "  ", "DE"):
        ok("lang/verbatim_%r" % code, validate_lang(code) == code)
    for bad in (5, ["de"], {"lang": "de"}, True):
        expect_tool_error("lang/rejected_%r" % (bad,), lambda b=bad: validate_lang(b), "invalid_lang")

    ok("budget/none_is_unset", validate_budget(None, "max_len") is None)
    for value in (1, 8, 512, 8192):
        ok("budget/ok_%d" % value, validate_budget(value, "max_len") == value)
    # The code carries the argument's own name, so a client that set one wrongly learns which.
    for bad in (0, -1, 1.5, 384.0, "384", True, [], {}, "  "):
        for arg in ("max_len", "head_max_len"):
            expect_tool_error("budget/rejected_%s_%r" % (arg, bad),
                              lambda b=bad, a=arg: validate_budget(b, a), "invalid_%s" % arg)
    # 384.0 is a float and core slices with it (`ids[:max_len]`), which is a TypeError in the middle
    # of a forward pass. JSON writes both 384 and 384.0, so this is a real input, not a corner.
    expect_tool_error("budget/float_is_rejected", lambda: validate_budget(384.0, "max_len"),
                      "invalid_max_len")

    ok("overrides/all_unset", _overrides(None, None, None, None) == {})
    ok("overrides/drops_unset",
       _overrides(None, "de", None, 384) == {"lang": "de", "head_max_len": 384},
       repr(_overrides(None, "de", None, 384)))
    ok("overrides/keeps_falsy_lang", _overrides(None, "", None, None) == {"lang": ""},
       repr(_overrides(None, "", None, None)))
    ok("overrides/names_in_order",
       list(_overrides("t", "l", 1, 2)) == ["task", "lang", "max_len", "head_max_len"])


def test_controls_predict():
    """What `laya_predict` forwards, and what it refuses to forward."""
    # The baseline these checks protect: a call that sets no control reaches core exactly as it did
    # before the controls existed -- no keywords at all.
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, router=router)
    ok("predict/no_controls_call", router.predict_calls == [{}], repr(router.predict_calls))
    laya_predict(STATE, QUESTIONS, model="english", router=router)
    ok("predict/pinned_model_only_kwarg", router.predict_calls[-1] == {"model": "english"},
       repr(router.predict_calls[-1]))

    router = ControlRouter()
    out = laya_predict(STATE, QUESTIONS, task="typed_decisions", lang="de",
                       max_len=1024, head_max_len=512, router=router)
    ok("predict/all_controls_forwarded",
       router.predict_calls == [{"task": "typed_decisions", "lang": "de",
                                 "max_len": 1024, "head_max_len": 512}],
       repr(router.predict_calls))
    ok("predict/answers_untouched", out["answers"]["department"]["choice"] == "billing")

    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, lang="de", router=router)
    ok("predict/lang_only", router.predict_calls == [{"lang": "de"}], repr(router.predict_calls))
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, head_max_len=384, router=router)
    ok("predict/budget_only", router.predict_calls == [{"head_max_len": 384}],
       repr(router.predict_calls))

    # A pinned checkpoint plus a task is two answers to one question: `Router._route` checks model
    # first and never reads the task, and `Agent.system_one` does not accept it at all.
    for model in ("english", "laya", "ML"):
        router = ControlRouter()
        expect_tool_error("predict/model_plus_task_%s" % model,
                          lambda m=model: laya_predict(STATE, QUESTIONS, model=m,
                                                       task="typed_decisions", router=router),
                          "invalid_task")
        ok("predict/model_plus_task_no_call_%s" % model, router.predict_calls == [],
           repr(router.predict_calls))

    # Everything is checked before core is called, so a bad value costs no forward pass.
    router = ControlRouter()
    expect_tool_error("predict/bad_budget_before_core",
                      lambda: laya_predict(STATE, QUESTIONS, max_len=0, router=router),
                      "invalid_max_len")
    ok("predict/bad_budget_no_call", router.predict_calls == [], repr(router.predict_calls))
    router = ControlRouter()
    expect_tool_error("predict/bad_task_before_core",
                      lambda: laya_predict(STATE, QUESTIONS, task="nope", router=router),
                      "invalid_task")
    ok("predict/bad_task_no_call", router.predict_calls == [], repr(router.predict_calls))

    # The agent branch: budget and lang reach a checkpoint directly, task cannot mean anything.
    agent = ControlAgent()
    laya_predict(STATE, QUESTIONS, model="english", agent=agent)
    ok("predict/agent_no_controls_call", agent.calls == [{}], repr(agent.calls))
    agent = ControlAgent()
    out = laya_predict(STATE, QUESTIONS, model="english", lang="de", max_len=1024,
                       head_max_len=512, agent=agent)
    ok("predict/agent_budget_forwarded",
       agent.calls == [{"lang": "de", "max_len": 1024, "head_max_len": 512}], repr(agent.calls))
    ok("predict/agent_routing_recorded", out["routing"]["model"] == "english", repr(out["routing"]))
    agent = ControlAgent()
    expect_tool_error("predict/agent_task_refused",
                      lambda a=agent: laya_predict(STATE, QUESTIONS, model="english",
                                                   task="typed_decisions", agent=a),
                      "invalid_task")
    ok("predict/agent_task_no_call", agent.calls == [], repr(agent.calls))

    # A router whose predict() accepts only state and questions still answers a plain call.
    class LegacyRouter:
        _agents = {"english": FakeAgent()}

        def predict(self, state, questions):
            return {"answers": {"department": {"choice": "other", "confidence": 0.6}}}

    out = laya_predict(STATE, QUESTIONS, router=LegacyRouter())
    ok("predict/legacy_router_untouched", out["answers"]["department"]["choice"] == "other")


def test_controls_route():
    """`laya_route` takes the routing overrides, so a route can explain a pinned predict."""
    router = ControlRouter(routed="multilingual")
    laya_route(STATE, QUESTIONS, router=router)
    ok("route/no_overrides_call", router.route_calls == [{}], repr(router.route_calls))

    router = ControlRouter(routed="multilingual")
    out = laya_route(STATE, QUESTIONS, task="typed_decisions", lang="de", router=router)
    ok("route/overrides_forwarded",
       router.route_calls == [{"task": "typed_decisions", "lang": "de"}], repr(router.route_calls))
    ok("route/decision_shape", set(out) == {"model", "repo", "reason"}, repr(out))

    # `auto` is this layer's sentinel and core has no such name: it means "do not pin", so it is
    # dropped rather than forwarded as a checkpoint that does not exist.
    for unset in (None, "auto", "AUTO", " auto "):
        router = ControlRouter()
        laya_route(STATE, QUESTIONS, model=unset, router=router)
        ok("route/auto_is_absent_%r" % (unset,), router.route_calls == [{}], repr(router.route_calls))
    router = ControlRouter()
    laya_route(STATE, QUESTIONS, model="en", router=router)
    ok("route/model_canonical", router.route_calls == [{"model": "english"}],
       repr(router.route_calls))
    router = ControlRouter()
    expect_tool_error("route/model_plus_task",
                      lambda: laya_route(STATE, QUESTIONS, model="english",
                                         task="typed_decisions", router=router),
                      "invalid_task")
    ok("route/refused_before_core", router.route_calls == [], repr(router.route_calls))
    for bad in ("nope", 5, []):
        router = ControlRouter()
        expect_tool_error("route/bad_task_%r" % (bad,),
                          lambda b=bad: laya_route(STATE, QUESTIONS, task=b, router=router),
                          "invalid_task")
        ok("route/bad_task_no_call_%r" % (bad,), router.route_calls == [], repr(router.route_calls))
    router = ControlRouter()
    expect_tool_error("route/bad_lang",
                      lambda: laya_route(STATE, QUESTIONS, lang=["de"], router=router),
                      "invalid_lang")
    ok("route/bad_lang_no_call", router.route_calls == [], repr(router.route_calls))
    # No budget on a route: there is no forward pass to size, so the parameter does not exist.
    try:
        laya_route(STATE, QUESTIONS, max_len=512, router=ControlRouter())
        ok("route/no_budget_param", False, "accepted max_len")
    except TypeError:
        ok("route/no_budget_param", True)


def test_controls_shortlist():
    """The budget reaches the answering pass; the task reaches only the route that chose it."""
    small = {"dept": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}}}

    router = ControlRouter(routed="multilingual")
    laya_shortlist(STATE, small, k=2, router=router, embed_fn=_tie_embed)
    ok("shortlist/no_controls_route", router.route_calls == [{}], repr(router.route_calls))
    ok("shortlist/no_controls_predict", router.predict_calls == [{"model": "multilingual"}],
       repr(router.predict_calls))

    router = ControlRouter(routed="multilingual")
    out = laya_shortlist(STATE, small, k=2, task="typed_decisions", lang="de",
                         max_len=1024, head_max_len=384, router=router, embed_fn=_tie_embed)
    ok("shortlist/route_sees_task",
       router.route_calls == [{"task": "typed_decisions", "lang": "de"}], repr(router.route_calls))
    ok("shortlist/predict_sees_budget",
       router.predict_calls == [{"model": "multilingual", "lang": "de",
                                 "max_len": 1024, "head_max_len": 384}],
       repr(router.predict_calls))
    ok("shortlist/shortlist_still_ran", out["shortlist"]["dept"]["k"] == 2
       and len(out["shortlist"]["dept"]["labels"]) == 2, repr(out["shortlist"]["dept"]))

    router = ControlRouter(routed="multilingual")
    laya_shortlist(STATE, small, k=2, model="english", lang="de", head_max_len=384,
                   router=router, embed_fn=_tie_embed)
    ok("shortlist/pinned_skips_route", router.route_calls == [], repr(router.route_calls))
    ok("shortlist/pinned_budget",
       router.predict_calls == [{"model": "english", "lang": "de", "head_max_len": 384}],
       repr(router.predict_calls))

    router = ControlRouter()
    expect_tool_error("shortlist/pinned_plus_task",
                      lambda: laya_shortlist(STATE, small, k=2, model="english",
                                             task="typed_decisions", router=router,
                                             embed_fn=_tie_embed),
                      "invalid_task")
    ok("shortlist/pinned_plus_task_no_call", router.predict_calls == [], repr(router.predict_calls))

    agent = ControlAgent()
    laya_shortlist(STATE, small, k=2, model="english", agent=agent, lang="de",
                   head_max_len=384, embed_fn=_tie_embed)
    ok("shortlist/agent_budget", agent.calls == [{"lang": "de", "head_max_len": 384}],
       repr(agent.calls))
    agent = ControlAgent()
    expect_tool_error("shortlist/agent_task_refused",
                      lambda a=agent: laya_shortlist(STATE, small, k=2, model="english",
                                                     task="typed", agent=a, embed_fn=_tie_embed),
                      "invalid_task")
    ok("shortlist/agent_task_no_call", agent.calls == [], repr(agent.calls))

    # A bad budget is refused before the embedding pass, which is the expensive half.
    calls = []

    def counting_embed(texts):
        calls.append(len(texts))
        return _tie_embed(texts)

    router = ControlRouter()
    expect_tool_error("shortlist/bad_budget_before_embedding",
                      lambda: laya_shortlist(STATE, small, k=2, head_max_len="384", router=router,
                                             embed_fn=counting_embed),
                      "invalid_head_max_len")
    ok("shortlist/bad_budget_no_embedding", calls == [], repr(calls))


def test_controls_preset():
    """The preset fixes the questions, not the route or the budget."""
    def builder(attr):
        return {"probe": {"type": "noul", "instructions": "Does the `body` need a human?"}}

    router = ControlRouter()
    laya_preset("guard", STATE, router=router, preset_builder=builder)
    ok("preset/no_controls_call", router.predict_calls == [{}], repr(router.predict_calls))

    router = ControlRouter()
    out = laya_preset("guard", STATE, task="typed_decisions", lang="de", max_len=1024,
                      head_max_len=384, router=router, preset_builder=builder)
    ok("preset/controls_forwarded",
       router.predict_calls == [{"task": "typed_decisions", "lang": "de",
                                 "max_len": 1024, "head_max_len": 384}],
       repr(router.predict_calls))
    # The questions the preset builder produced are what core was asked, alongside the controls --
    # the preset supplies the questions, the caller supplies how they get answered.
    ok("preset/questions_reached_core",
       list(router.predict_questions[-1]) == ["probe"], repr(router.predict_questions[-1]))
    ok("preset/result_shape", set(out) >= {"answers", "routing", "latency_ms"}, repr(sorted(out)))

    for args, code in (({"max_len": -1}, "invalid_max_len"),
                       ({"task": "nope"}, "invalid_task"),
                       ({"lang": 7}, "invalid_lang"),
                       ({"head_max_len": "512"}, "invalid_head_max_len")):
        router = ControlRouter()
        expect_tool_error("preset/rejected_%r" % (args,),
                          lambda a=args: laya_preset("guard", STATE, router=router,
                                                     preset_builder=builder, **a),
                          code)
        ok("preset/rejected_no_call_%r" % (args,), router.predict_calls == [],
           repr(router.predict_calls))


def test_controls_signature_and_schema():
    """The keywords a client can see, and that none of them became required.

    A new required argument would break every prompt already in the wild; a Python keyword with no
    schema entry would be a control nobody can send. Both directions are checked.
    """
    import inspect

    controls = ["task", "lang", "lang_guess", "max_len", "head_max_len"]
    for fn, want in ((laya_predict, controls), (laya_shortlist, controls),
                     (laya_preset, controls), (laya_route, ["model", "task", "lang", "lang_guess"])):
        params = inspect.signature(fn).parameters
        for name in want:
            ok("signature/%s_has_%s" % (fn.__name__, name), name in params, repr(sorted(params)))
            ok("signature/%s_%s_default_none" % (fn.__name__, name),
               params[name].default is None, repr(params[name].default))
            ok("signature/%s_%s_keyword_only" % (fn.__name__, name),
               params[name].kind is inspect.Parameter.KEYWORD_ONLY)
    ok("signature/predict_positionals_kept",
       list(inspect.signature(laya_predict).parameters)[:3] == ["state", "questions", "model"],
       repr(list(inspect.signature(laya_predict).parameters)))

    by_name = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
    for name, want in (("laya_predict", controls), ("laya_shortlist", controls),
                       ("laya_preset", controls), ("laya_route", ["model", "task", "lang"])):
        tool = by_name[name]
        schema = tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for arg in want:
            ok("schema/%s_exposes_%s" % (name, arg), arg in props, repr(sorted(props)))
            spec = props.get(arg, {})
            ok("schema/%s_%s_nullable" % (name, arg),
               {"type": "null"} in (spec.get("anyOf") or []), repr(spec))
            ok("schema/%s_%s_default_none" % (name, arg),
               spec.get("default", "missing") is None, repr(spec))
            ok("schema/%s_%s_not_required" % (name, arg), arg not in required, repr(required))
            ok("schema/%s_desc_documents_%s" % (name, arg), arg in tool.description.lower())
        ok("schema/%s_required_unchanged" % name,
           required == (["preset", "state"] if name == "laya_preset" else ["state", "questions"]),
           repr(required))


def test_lang_guess_control():
    """`lang_guess` -- core's soft routing hint (#489 family) -- reaches the single-request tools.

    The batch surface already forwards it (``BATCH_ITEM_OVERRIDES`` advertises the key, and
    ``laya_predict_batch``/``laya_route_batch`` hand it to ``Router.*_batch``), and ``laya-serve`` and
    the CLI forward it, so a client scoring one state over MCP could ask for a probable language to
    nudge the checkpoint only by sending a batch of one. `lang_guess` is routing-only, so it is
    treated like ``task``: forwarded to a router call, refused on a pinned model (a pinned call has
    nothing to route), and stripped on a direct agent (``Agent.predict`` does not accept it).
    """
    import inspect

    # The validator mirrors validate_lang: a code passes through verbatim, a non-string is refused in
    # this layer's words. A callable is core's other accepted form but an MCP client carries JSON.
    ok("langguess/none_is_unset", validate_lang_guess(None) is None)
    for code in ("de", "en-US", "pt", "", "  ", "DE"):
        ok("langguess/verbatim_%r" % code, validate_lang_guess(code) == code)
    for bad in (5, ["de"], {"lang": "de"}, True):
        expect_tool_error("langguess/rejected_%r" % (bad,),
                          lambda b=bad: validate_lang_guess(b), "invalid_lang_guess")
    # A callable is legal in core but not expressible over the wire; the MCP boundary refuses it
    # rather than silently dropping the hint.
    expect_tool_error("langguess/callable_rejected",
                      lambda: validate_lang_guess(lambda state: "de"), "invalid_lang_guess")

    # laya_predict forwards lang_guess to core's predict, and only when set: a call with no hint
    # reaches core exactly as before the keyword was exposed here.
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, router=router)
    ok("predict/langguess_unset_absent", router.predict_calls == [{}], repr(router.predict_calls))
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, lang_guess="de", router=router)
    ok("predict/langguess_forwarded", router.predict_calls == [{"lang_guess": "de"}],
       repr(router.predict_calls))
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, lang_guess="de", task="typed_decisions", router=router)
    ok("predict/langguess_alongside_task",
       router.predict_calls == [{"task": "typed_decisions", "lang_guess": "de"}],
       repr(router.predict_calls))
    # An explicit lang and a soft hint are distinct keywords: both reach core, and core orders them.
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, lang="fr", lang_guess="de", router=router)
    ok("predict/lang_and_langguess_distinct",
       router.predict_calls == [{"lang": "fr", "lang_guess": "de"}], repr(router.predict_calls))

    # Pinned: refused before core, exactly like task -- there is nothing to route.
    for model in ("english", "laya", "ML"):
        router = ControlRouter()
        expect_tool_error("predict/model_plus_langguess_%s" % model,
                          lambda m=model: laya_predict(STATE, QUESTIONS, model=m,
                                                       lang_guess="de", router=router),
                          "invalid_lang_guess")
        ok("predict/model_plus_langguess_no_call_%s" % model, router.predict_calls == [],
           repr(router.predict_calls))
    router = ControlRouter()
    expect_tool_error("predict/bad_langguess_before_core",
                      lambda: laya_predict(STATE, QUESTIONS, lang_guess=["de"], router=router),
                      "invalid_lang_guess")
    ok("predict/bad_langguess_no_call", router.predict_calls == [], repr(router.predict_calls))

    # The direct-agent branch: Agent.predict does not accept lang_guess, so it is stripped -- a
    # no-router auto call answers through the agent without ever naming the hint to core.
    agent = ControlAgent()
    laya_predict(STATE, QUESTIONS, lang_guess="de", agent=agent)
    ok("predict/agent_langguess_stripped", agent.calls == [{}], repr(agent.calls))
    agent = ControlAgent()
    laya_predict(STATE, QUESTIONS, lang="de", lang_guess="en", agent=agent)
    ok("predict/agent_keeps_lang_drops_langguess", agent.calls == [{"lang": "de"}],
       repr(agent.calls))

    # laya_route: a routing tool, so lang_guess reaches route and is refused on a pin.
    router = ControlRouter(routed="multilingual")
    out = laya_route(STATE, QUESTIONS, lang_guess="de", router=router)
    ok("route/langguess_forwarded", router.route_calls == [{"lang_guess": "de"}],
       repr(router.route_calls))
    ok("route/langguess_decision_shape", set(out) == {"model", "repo", "reason"}, repr(out))
    router = ControlRouter()
    expect_tool_error("route/model_plus_langguess",
                      lambda: laya_route(STATE, QUESTIONS, model="english",
                                         lang_guess="de", router=router),
                      "invalid_lang_guess")
    ok("route/refused_langguess_before_core", router.route_calls == [], repr(router.route_calls))

    # laya_shortlist: lang_guess reaches the route that chose the checkpoint, and is stripped from
    # the answering pass (which pins the routed model, so the hint has nothing left to decide).
    small = {"dept": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}}}
    router = ControlRouter(routed="multilingual")
    laya_shortlist(STATE, small, k=2, lang_guess="de", router=router, embed_fn=_tie_embed)
    ok("shortlist/route_sees_langguess", router.route_calls == [{"lang_guess": "de"}],
       repr(router.route_calls))
    ok("shortlist/predict_drops_langguess",
       router.predict_calls == [{"model": "multilingual"}], repr(router.predict_calls))
    router = ControlRouter()
    expect_tool_error("shortlist/pinned_plus_langguess",
                      lambda: laya_shortlist(STATE, small, k=2, model="english",
                                             lang_guess="de", router=router,
                                             embed_fn=_tie_embed),
                      "invalid_lang_guess")
    ok("shortlist/pinned_plus_langguess_no_call", router.predict_calls == [],
       repr(router.predict_calls))

    # laya_preset threads it through laya_predict (always auto, so never refused).
    def builder(attr):
        return {"probe": {"type": "noul", "instructions": "Does the `body` need a human?"}}
    router = ControlRouter()
    laya_preset("guard", STATE, lang_guess="de", router=router, preset_builder=builder)
    ok("preset/langguess_forwarded", router.predict_calls == [{"lang_guess": "de"}],
       repr(router.predict_calls))

    # Cross-surface drift guard: lang_guess is a real core routing control (serve forwards it as a
    # body control; the batch item overrides already name it) and Router.route/predict accept it, so
    # this layer can never advertise a keyword core would reject.
    from laya import serve as serve_mod

    ok("drift/langguess_in_body_controls", "lang_guess" in serve_mod.BODY_CONTROLS,
       repr(serve_mod.BODY_CONTROLS))
    ok("drift/langguess_in_batch_overrides", "lang_guess" in BATCH_ITEM_OVERRIDES,
       repr(BATCH_ITEM_OVERRIDES))
    for method in ("route", "predict"):
        params = inspect.signature(getattr(__import__("laya.router", fromlist=["Router"]).Router,
                                           method)).parameters
        ok("drift/router_%s_accepts_langguess" % method, "lang_guess" in params,
           repr(sorted(params)))


def test_min_confidence_control():
    """`min_confidence` -- core's per-call abstention gate (#361) -- is reachable over MCP.

    #567 forwarded every other per-call control (task/lang/max_len/head_max_len) to these tools,
    but not this one: a client could pin a checkpoint and size its budget over the wire, yet could
    not ask Laya to abstain on an unsure answer -- even though `min_confidence` is a validated
    keyword of `Agent.predict`, `Router.predict`/`predict_batch` and `laya.decide` on `main`.
    """
    import inspect

    # The range/bool/non-finite rule is core's; this layer only re-labels the refusal, so it can
    # never be wider or narrower than `check_min_confidence`.
    ok("minconf/none_is_unset", validate_min_confidence(None) is None)
    for value in (0, 0.0, 0.5, 1, 1.0, 0.94):
        ok("minconf/ok_%r" % (value,), validate_min_confidence(value) == float(value),
           repr(validate_min_confidence(value)))
    for bad in (-0.1, 1.1, 2, True, False, [], {}, "0.5", float("nan"), float("inf")):
        expect_tool_error("minconf/rejected_%r" % (bad,),
                          lambda b=bad: validate_min_confidence(b), "invalid_min_confidence")

    # laya_predict forwards it to core's predict, and only when set: a no-abstention call reaches
    # core exactly as it did before the keyword was exposed here.
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, router=router)
    ok("predict/minconf_unset_absent", router.predict_calls == [{}], repr(router.predict_calls))
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, min_confidence=0.85, router=router)
    ok("predict/minconf_forwarded", router.predict_calls == [{"min_confidence": 0.85}],
       repr(router.predict_calls))
    router = ControlRouter()
    laya_predict(STATE, QUESTIONS, lang="de", max_len=1024, min_confidence=0.9, router=router)
    ok("predict/minconf_alongside_controls",
       router.predict_calls == [{"lang": "de", "max_len": 1024, "min_confidence": 0.9}],
       repr(router.predict_calls))
    router = ControlRouter()
    expect_tool_error("predict/bad_minconf",
                      lambda: laya_predict(STATE, QUESTIONS, min_confidence=2, router=router),
                      "invalid_min_confidence")
    ok("predict/bad_minconf_no_call", router.predict_calls == [], repr(router.predict_calls))
    agent = ControlAgent()
    laya_predict(STATE, QUESTIONS, model="english", min_confidence=0.5, agent=agent)
    ok("predict/agent_minconf", agent.calls == [{"min_confidence": 0.5}], repr(agent.calls))

    # laya_shortlist carries it to the answering pass beside the budget (route stays budget-free).
    small = {"dept": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}}}
    router = ControlRouter(routed="multilingual")
    laya_shortlist(STATE, small, k=2, head_max_len=384, min_confidence=0.8,
                   router=router, embed_fn=_tie_embed)
    ok("shortlist/minconf_reaches_predict",
       router.predict_calls == [{"model": "multilingual", "head_max_len": 384,
                                 "min_confidence": 0.8}], repr(router.predict_calls))
    router = ControlRouter(routed="multilingual")
    laya_shortlist(STATE, small, k=2, router=router, embed_fn=_tie_embed)
    ok("shortlist/minconf_unset_absent",
       "min_confidence" not in router.predict_calls[0], repr(router.predict_calls))
    router = ControlRouter()
    expect_tool_error("shortlist/bad_minconf",
                      lambda: laya_shortlist(STATE, small, k=2, min_confidence=-1, router=router,
                                             embed_fn=_tie_embed),
                      "invalid_min_confidence")

    # laya_preset inherits the control through laya_predict.
    def builder(attr):
        return {"probe": {"type": "noul", "instructions": "Does the `body` need a human?"}}
    router = ControlRouter()
    laya_preset("guard", STATE, min_confidence=0.7, router=router, preset_builder=builder)
    ok("preset/minconf_forwarded", router.predict_calls == [{"min_confidence": 0.7}],
       repr(router.predict_calls))

    # The flagship: laya_decide uses core's abstention to null an unsure field while still
    # reporting its confidence -- the one thing a values-consuming client needs and cannot get
    # any other way over MCP. FakeRouter answers department at 0.94, urgency at 0.8, needs_human
    # at 0.89, so a 0.9 gate keeps the first and nulls the other two.
    router = TypeEchoRouter()
    plain = laya_decide(STATE, DECIDE_SCHEMA, router=router)
    ok("decide/minconf_unset_no_nul", plain["values"]["urgency"] == 2
       and plain["values"]["needs_human"] is True, repr(plain["values"]))
    abstained = laya_decide(STATE, DECIDE_SCHEMA, min_confidence=0.9, router=router)
    ok("decide/keeps_confident_field", abstained["values"]["department"] == "billing",
       repr(abstained["values"]))
    ok("decide/nulls_low_score", abstained["values"]["urgency"] is None,
       repr(abstained["values"]))
    ok("decide/nulls_low_noul", abstained["values"]["needs_human"] is None,
       repr(abstained["values"]))
    ok("decide/confidence_survives_null",
       set(abstained["confidence"]) == {"department", "urgency", "needs_human"}
       and abstained["confidence"]["urgency"] == 0.8, repr(abstained["confidence"]))
    # A threshold of 0 abstains over nothing -- the raw answers are byte-identical to no call.
    zero = laya_decide(STATE, DECIDE_SCHEMA, min_confidence=0.0, router=router)
    ok("decide/zero_is_no_abstention", zero["values"] == plain["values"], repr(zero["values"]))

    # The tool is a thin surface over the documented core abstention, not a second projection.
    import laya

    core = laya.decide(router, STATE, schema=DECIDE_SCHEMA, min_confidence=0.9)
    ok("decide/abstention_parity_with_core", abstained["values"] == core,
       "tool=%r core=%r" % (abstained["values"], core))

    # Refused before core, so a bad threshold costs no forward pass.
    router = TypeEchoRouter()
    expect_tool_error("decide/bad_minconf",
                      lambda: laya_decide(STATE, DECIDE_SCHEMA, min_confidence=1.5, router=router),
                      "invalid_min_confidence")

    # Every tool that answers exposes the keyword, defaults it to None, and documents it.
    tools = [laya_predict, laya_shortlist, laya_preset, laya_decide]
    for fn in tools:
        params = inspect.signature(fn).parameters
        ok("sig/%s_has_min_confidence" % fn.__name__, "min_confidence" in params,
           repr(sorted(params)))
        ok("sig/%s_min_confidence_default_none" % fn.__name__,
           params["min_confidence"].default is None, repr(params["min_confidence"].default))
        ok("sig/%s_min_confidence_keyword_only" % fn.__name__,
           params["min_confidence"].kind is inspect.Parameter.KEYWORD_ONLY)
    by_name = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
    for name in ("laya_predict", "laya_shortlist", "laya_preset", "laya_decide"):
        tool = by_name[name]
        schema = tool.input_schema if hasattr(tool, "input_schema") else tool.inputSchema
        props = schema.get("properties", {})
        spec = props.get("min_confidence", {})
        ok("schema/%s_exposes_min_confidence" % name, "min_confidence" in props,
           repr(sorted(props)))
        ok("schema/%s_min_confidence_nullable" % name,
           {"type": "null"} in (spec.get("anyOf") or []), repr(spec))
        ok("schema/%s_min_confidence_typed" % name,
           any(opt.get("type") in ("number", "integer") for opt in (spec.get("anyOf") or [spec])),
           repr(spec))
        ok("schema/%s_min_confidence_not_required" % name,
           "min_confidence" not in schema.get("required", []), repr(schema.get("required")))
        ok("schema/%s_desc_documents_min_confidence" % name,
           "min_confidence" in tool.description.lower(), repr(tool.description[-80:]))


def test_question_validation_matches_the_agent():
    """MCP must reject a bad question the same way the agent does, and say so.

    `Agent._check_question` is the contract: it runs before encoding and raises `ValueError`
    for a caller mistake. The MCP tools used to check only part of it, so the rest reached the
    agent, and `_wrap` mapped the `ValueError` to `internal_error` -- the code this layer
    reserves for a tool that broke ("predict returned non-object", "router has no route()").
    An MCP client was told the server failed when it had sent a malformed question.

    This is the same shape as the serve-side test that calls the real guard from a stub
    router: no weights, and the core guard is the oracle rather than a second hand-written
    copy of the rules.
    """
    from laya.agent import Agent

    def core_rejects(qid, qdef):
        try:
            Agent._check_question(qid, dict(qdef))
        except ValueError:
            return True
        except Exception:  # noqa: BLE001 -- any refusal is still a refusal
            return True
        return False

    # (label, question) for every rule the agent enforces and the tools did not.
    parity = [
        ("noul_criteria_bad_key",
         {"type": "noul", "instructions": "is true?",
          "criteria": {"yes": "affirmative", "no": "negative"}}),
        ("noul_criteria_extra_key",
         {"type": "noul", "instructions": "is true?",
          "criteria": {"true": "yes", "false": "no", "maybe": "perhaps"}}),
        ("noul_labels_wrong_names",
         {"type": "noul", "instructions": "is true?", "labels": {"A": "yes", "B": "no"}}),
        ("noul_labels_missing_true",
         {"type": "noul", "instructions": "is true?", "labels": {"false": "no"}}),
        ("noul_labels_identical",
         {"type": "noul", "instructions": "is true?",
          "labels": {"false": "same", "true": "same"}}),
        ("noul_labels_empty",
         {"type": "noul", "instructions": "is true?", "labels": {"false": "", "true": "yes"}}),
        ("noul_labels_not_object",
         {"type": "noul", "instructions": "is true?", "labels": ["yes", "no"]}),
        # A label is text, not a value. The agent checks the type before using it, so these
        # have to be refused here too -- stringifying them would accept exactly what the agent
        # rejects, and the failure would surface as an internal_error instead.
        ("noul_labels_numeric",
         {"type": "noul", "instructions": "is true?", "labels": {"false": 0, "true": 1}}),
        ("noul_labels_mixed_types",
         {"type": "noul", "instructions": "is true?", "labels": {"false": "no", "true": 1}}),
        ("noul_labels_bool",
         {"type": "noul", "instructions": "is true?", "labels": {"false": False, "true": True}}),
        ("noul_labels_null",
         {"type": "noul", "instructions": "is true?", "labels": {"false": None, "true": "yes"}}),
        ("noul_labels_list",
         {"type": "noul", "instructions": "is true?", "labels": {"false": ["no"], "true": "yes"}}),
        ("noul_labels_whitespace_equal",
         {"type": "noul", "instructions": "is true?", "labels": {"false": " same ", "true": "same"}}),
        ("score_level_null",
         {"type": "score", "instructions": "how bad", "criteria": ["fine", None]}),
        ("score_level_null_first",
         {"type": "score", "instructions": "how bad", "criteria": [None, "fine"]}),
        ("labels_on_choice",
         {"type": "choice", "instructions": "which?", "criteria": {"a": "first"},
          "labels": {"false": "no", "true": "yes"}}),
        # `option_order` must be a permutation of the option indices: anything else drops an
        # option or shows one twice.
        ("option_order_repeat",
         {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": "", "c": ""},
          "option_order": [0, 0, 0]}),
        ("option_order_short",
         {"type": "choice", "instructions": "which?", "criteria": {"a": "", "b": "", "c": ""},
          "option_order": [1, 0]}),
        ("option_order_long",
         {"type": "score", "instructions": "how bad", "criteria": ["fine", "bad"],
          "option_order": [1, 0, 2]}),
        ("option_order_out_of_range",
         {"type": "score", "instructions": "how bad", "criteria": ["fine", "bad"],
          "option_order": [1, 2]}),
        ("option_order_negative",
         {"type": "noul", "instructions": "is true?", "option_order": [-1, 0]}),
        ("option_order_bools",
         {"type": "noul", "instructions": "is true?", "option_order": [True, False]}),
        ("option_order_strings",
         {"type": "noul", "instructions": "is true?", "option_order": ["1", "0"]}),
        ("option_order_floats",
         {"type": "noul", "instructions": "is true?", "option_order": [1.0, 0.0]}),
        ("option_order_not_list",
         {"type": "noul", "instructions": "is true?", "option_order": "10"}),
        ("option_order_null",
         {"type": "noul", "instructions": "is true?", "option_order": None}),
    ]
    for label, qdef in parity:
        ok("question_parity/core_rejects_%s" % label, core_rejects("q", qdef))
        expect_tool_error("question_parity/mcp_rejects_%s" % label,
                          lambda d=qdef: validate_questions({"q": d}), "invalid_questions")

    # The other direction is deliberately not asserted: the tools reject a few shapes the
    # agent tolerates (a choice `criteria` list, a non-string `instructions`). Widening what
    # MCP accepts is a separate contract decision and is not what this change is about.
    accepted_core = [
        ("choice_criteria_list", {"type": "choice", "instructions": "which?",
                                  "criteria": ["a", "b"]}),
    ]
    for label, qdef in accepted_core:
        ok("question_parity/core_accepts_%s" % label, not core_rejects("q", qdef))

    # A valid order is accepted by both, and kept: dropping it silently ignored a documented key.
    valid_orders = [
        ("option_order_choice", {"type": "choice", "instructions": "which?",
                                 "criteria": {"a": "", "b": "", "c": ""}, "option_order": [2, 0, 1]}),
        ("option_order_score", {"type": "score", "instructions": "how bad",
                                "criteria": ["fine", "bad"], "option_order": [1, 0]}),
        ("option_order_noul", {"type": "noul", "instructions": "is true?", "option_order": [1, 0]}),
        ("option_order_identity", {"type": "noul", "instructions": "is true?", "option_order": [0, 1]}),
    ]
    for label, qdef in valid_orders:
        ok("question_parity/core_accepts_%s" % label, not core_rejects("q", qdef))
        kept = validate_questions({"q": qdef})["q"].get("option_order")
        ok("question_parity/mcp_keeps_%s" % label, kept == qdef["option_order"], repr(kept))


def test_a_bad_question_is_a_caller_error_not_a_server_fault():
    """End to end through the wrapper: the client must see `invalid_questions`.

    The wrapper turns a `ToolError` into that code and preserves its message, while any other
    exception becomes `internal_error`. So the only thing standing between a malformed question
    and a "the server broke" answer is whether the tools reject it themselves.
    """
    import json as _json

    from laya.mcp import server as server_mod

    class ValidatingAgent:
        """The real guard, no checkpoint: what `Agent.system_one` runs before encoding."""

        def predict(self, state, questions, **kwargs):
            from laya.agent import Agent
            for qid, qdef in questions.items():
                Agent._check_question(qid, qdef)
            return {"answers": {qid: {"type": "noul", "noul": 0.5, "noul_label": None}
                                for qid in questions}}

    class Router:
        loaded = ["english"]

        def predict(self, state, questions, model=None, **kwargs):
            return ValidatingAgent().predict(state, questions)

        def route(self, state, questions, **kwargs):
            return {"model": "english", "repo": "r", "reason": "stub"}

    for label, qdef in (
        ("noul_criteria_bad_key", {"type": "noul", "instructions": "is true?",
                                   "criteria": {"yes": "affirmative", "no": "negative"}}),
        ("noul_labels_wrong_names", {"type": "noul", "instructions": "is true?",
                                     "labels": {"A": "yes", "B": "no"}}),
        ("noul_labels_numeric", {"type": "noul", "instructions": "is true?",
                                 "labels": {"false": 0, "true": 1}}),
        ("noul_labels_mixed_types", {"type": "noul", "instructions": "is true?",
                                     "labels": {"false": "no", "true": 1}}),
        ("score_level_null", {"type": "score", "instructions": "how bad",
                              "criteria": ["fine", None]}),
        ("option_order_repeat", {"type": "noul", "instructions": "is true?",
                                 "option_order": [0, 0]}),
    ):
        try:
            server_mod._wrap(server_mod.laya_predict, state=STATE, questions={"q": qdef},
                             model="english", router=Router())
            ok("caller_error/rejected_%s" % label, False, "no error raised")
        except Exception as exc:  # noqa: BLE001 -- the wrapper raises McpToolError
            payload = getattr(exc, "message", None) or str(exc)
            try:
                code = _json.loads(payload).get("error")
            except Exception:  # noqa: BLE001
                code = None
            ok("caller_error/%s" % label, code == "invalid_questions", "got %r" % (code,))


def test_timeout_removed():
    # The per-call timeout was removed: a ThreadPoolExecutor shutdown waits for
    # the work anyway, and MCP clients apply their own request timeout. The tool
    # functions no longer accept a timeout argument.
    import inspect

    import laya.mcp.tools as tools_mod

    for fn in (laya_predict, laya_route, laya_preset, laya_shortlist,
               laya_predict_batch, laya_route_batch, laya_decide):
        ok("timeout/param_absent_%s" % fn.__name__, "timeout" not in inspect.signature(fn).parameters)
    ok("timeout/executor_absent", "ThreadPoolExecutor" not in inspect.getsource(tools_mod))


# --- server registration (schema only, no model load) ------------------------

def test_batch_item_shape_as_documented():
    """Every place the batch item shape is written out says what the validator keeps.

    Three strings enumerate the keys a batch item may carry: `validate_batch_requests`' error
    message (checked against the validator in test_batch_validation), each batch tool's registered
    description, and the README's MCP section. A client reads them as the schema, so all three are
    compared to `BATCH_ITEM_OVERRIDES` -- the tuple `_validate_batch_item` branches on.
    `laya_route_batch` leaves the two token budgets out, because routing runs no forward pass and
    a budget there would be ignored.
    """
    by_name = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
    for tool, omit in (("laya_predict_batch", ()),
                       ("laya_route_batch", ("max_len", "head_max_len"))):
        want = batch_item_key_doc(omit=omit)
        ok("schema/%s_item_shape" % tool, want in (by_name[tool].description or ""),
           "documents %r" % want)

    # `encoding="utf-8"` explicitly: the README is UTF-8 and contains non-ASCII (the language
    # examples), while `Path.read_text()` defaults to the locale encoding -- cp1252 on the Windows
    # runner, where the decode raised `UnicodeDecodeError` and this suite failed. It went unseen
    # because the Windows lane ran this file without the `mcp` extra, so it skipped at the top.
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    marker = "one tool call takes an array of `{"
    ok("docs/readme_names_the_batch_shape", marker in readme, "sentence not found")
    if marker in readme:
        start = readme.index(marker) + len(marker)
        inner = readme[start:readme.index("}", start)]
        named = {part.strip().rstrip("?") for part in inner.split(",")}
        ok("docs/readme_batch_item_keys",
           named == {"state", "questions"} | set(BATCH_ITEM_OVERRIDES), repr(sorted(named)))


def test_cli_mcp_page_batch_rows():
    """Each batch row on the MCP docs page names its own key set, not the sibling's.

    `docs/cli-mcp.md` is the page a client reads before opening the tools/list response, so the
    `requests` cell is a fourth copy of the shape. Until now the three-copy gate above covered
    the tool descriptions and the README but not this table, and the `laya_route_batch` row read
    "same shape as `laya_predict_batch`" -- which is wrong, because `server.py:310` registers the
    route tool with `batch_item_key_doc(omit=("max_len", "head_max_len"))`. Routing runs no
    forward pass, so the token budgets are dropped; the row has to say the six keys the tool
    actually accepts.

    Both rows are read from the rendered markdown (the third `|`-delimited cell of the table),
    the brace-list is parsed out, and the resulting key set is compared to `batch_item_key_doc`
    with the same `omit` the server registers. The pre-fix wording is banned by direct substring
    so the check fails if the row reverts to pointing at the sibling instead of naming itself.
    """
    import re

    page = (Path(__file__).resolve().parents[1] / "docs" / "cli-mcp.md").read_text(encoding="utf-8")

    def row_cell(tool):
        for line in page.splitlines():
            if line.startswith("| `%s` |" % tool):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    return cells[2]
        return None

    for tool, omit in (("laya_predict_batch", ()),
                       ("laya_route_batch", ("max_len", "head_max_len"))):
        cell = row_cell(tool)
        ok("docs-mcp/%s_row_present" % tool, cell is not None, "row not found")
        if cell is None:
            continue
        match = re.search(r"`\{([^`]*)\}`", cell)
        ok("docs-mcp/%s_row_has_brace_list" % tool, match is not None,
           "cell=%r" % cell)
        if match is None:
            continue
        named = {part.strip().rstrip("?") for part in match.group(1).split(",")}
        want = {"state", "questions"} | (set(BATCH_ITEM_OVERRIDES) - set(omit))
        ok("docs-mcp/%s_row_keys" % tool, named == want,
           "row=%r want=%r" % (sorted(named), sorted(want)))

    ok("docs-mcp/route_batch_row_drops_sibling_reference",
       "same shape as `laya_predict_batch`" not in page)


def test_models_from_env():
    old = os.environ.get("LAYA_MODELS")
    try:
        os.environ.pop("LAYA_MODELS", None)
        ok("models/mcp_default", _models_from_env() == ["english", "multilingual"])
        os.environ["LAYA_MODELS"] = ""
        ok("models/empty_default", _models_from_env() == ["english", "multilingual"])
        os.environ["LAYA_MODELS"] = " english , multilingual "
        ok("models/whitespace", _models_from_env() == ["english", "multilingual"])
        os.environ["LAYA_MODELS"] = "typed-decisions"
        ok("models/explicit_single", _models_from_env() == ["typed-decisions"])
        os.environ["LAYA_MODELS"] = "english, multilingual, typed-decisions,"
        ok("models/trailing_comma", _models_from_env() == ["english", "multilingual", "typed-decisions"])
    finally:
        if old is None:
            os.environ.pop("LAYA_MODELS", None)
        else:
            os.environ["LAYA_MODELS"] = old


def test_auto_task_env():
    """LAYA_AUTO_TASK must reach the Router the MCP server builds, as it does the serve one.

    README's MCP section says these variables follow the contract at the top of laya.serve, so the
    check is built through both surfaces' real builders -- `laya.mcp.server._ensure_router()` and
    `laya.serve.build_router()` -- rather than by reading a flag off a hand-made Router.
    `Router.route()` delegates to the private `_route`, documented as deciding "without loading or
    running anything", so no checkpoint is downloaded here.
    """
    import laya.mcp.server as mcp_mod  # the module, not the MCPServer instance
    from laya.router import _TYPED_DECISION_WORKFLOWS
    from laya.serve import build_router

    state = {"body": "I was charged twice, please refund the duplicate"}
    schemas = {name: {qid: {"type": "choice", "options": ["yes", "no"]}
                      for qid in sorted(ids)}
               for name, ids in sorted(_TYPED_DECISION_WORKFLOWS.items())}
    saved = {k: os.environ.get(k) for k in ("LAYA_AUTO_TASK", "LAYA_PRELOAD")}
    saved_router = mcp_mod._ROUTER
    try:
        os.environ["LAYA_PRELOAD"] = "0"  # neither surface may build a checkpoint here

        def build(value):
            if value is None:
                os.environ.pop("LAYA_AUTO_TASK", None)
            else:
                os.environ["LAYA_AUTO_TASK"] = value
            mcp_mod._ROUTER = None  # the server caches the Router it built
            return mcp_mod._ensure_router(), build_router()

        for value, label in ((None, "unset"), ("0", "off"), ("1", "on"), ("true", "true")):
            mcp_router, serve_router = build(value)
            want = value in ("1", "true")
            ok("auto_task/%s_mcp" % label, mcp_router.auto_task_detection is want,
               repr(mcp_router.auto_task_detection))
            # The parity the README claims: one variable, one meaning on both surfaces.
            ok("auto_task/%s_matches_serve" % label,
               mcp_router.auto_task_detection == serve_router.auto_task_detection,
               "mcp=%r serve=%r" % (mcp_router.auto_task_detection,
                                    serve_router.auto_task_detection))

        # And the decision each workflow actually gets, on every workflow rather than one.
        on_mcp, on_serve = build("1")
        off_mcp, off_serve = build(None)
        for name, questions in sorted(schemas.items()):
            on_got = on_mcp.route(state, questions)["model"]
            off_got = off_mcp.route(state, questions)["model"]
            on_serve_got = on_serve.route(state, questions)["model"]
            off_serve_got = off_serve.route(state, questions)["model"]
            ok("auto_task/on_routes_typed_decisions_%s" % name, on_got == "typed-decisions",
               "got %r" % on_got)
            ok("auto_task/on_parity_%s" % name, on_got == on_serve_got,
               "mcp=%r serve=%r" % (on_got, on_serve_got))
            ok("auto_task/off_parity_%s" % name, off_got == off_serve_got,
               "mcp=%r serve=%r" % (off_got, off_serve_got))
            ok("auto_task/off_not_typed_%s" % name, off_got != "typed-decisions",
               "got %r" % off_got)
    finally:
        mcp_mod._ROUTER = saved_router
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_default_model_env():
    """LAYA_DEFAULT_MODEL has to reach both servers' Routers, with one meaning.

    README's MCP section says these variables follow the contract at the top of laya.serve, so
    both halves are built through the real builders -- `laya.mcp.server._ensure_router()` and
    `laya.serve.build_router()` -- and read off the Router each one actually returns. The
    variable itself is the routing fallback README prescribes for a mostly-non-English deployment
    (`Router(default="multilingual")`), which until now only examples/server.py could be told.
    The one place the two surfaces part is what an unresolvable name costs: serve refuses to
    start, a stdio server has no startup to refuse, so MCP has to carry the same words in a
    tool error.
    """
    import laya.mcp.server as mcp_mod  # the module, not the MCPServer instance
    from laya.serve import build_router

    saved = {k: os.environ.get(k)
             for k in ("LAYA_DEFAULT_MODEL", "LAYA_PRELOAD", "LAYA_AUTO_TASK")}
    saved_router = mcp_mod._ROUTER
    try:
        os.environ["LAYA_PRELOAD"] = "0"    # neither surface may build a checkpoint here
        os.environ["LAYA_AUTO_TASK"] = "0"  # leaves the fallback the only thing moving

        def build(value):
            if value is None:
                os.environ.pop("LAYA_DEFAULT_MODEL", None)
            else:
                os.environ["LAYA_DEFAULT_MODEL"] = value
            mcp_mod._ROUTER = None  # the server caches the Router it built
            return mcp_mod._ensure_router(), build_router()

        # Unset / empty / blank mean "whatever the Router's own default is", read off the
        # signature rather than restated, so this cannot drift the next time it moves.
        from laya.router import Router as _Router
        _stock = inspect.signature(_Router.__init__).parameters["default"].default
        for label, value, want in (
                ("unset", None, _stock),
                ("empty", "", _stock),
                ("blank", "   ", _stock),
                ("canonical", "multilingual", "multilingual"),
                ("alias", "ml", "multilingual"),
                ("padded_upper", " MULTI ", "multilingual"),
                ("typed", "typed-decisions", "typed-decisions")):
            try:
                mcp_router, serve_router = build(value)
            except (Exception, SystemExit) as exc:  # noqa: BLE001 -- every value here is valid
                # Named rather than fatal: a surface that raises for a name core accepts is the
                # bug, and a suite that dies on the way to reporting it leaves the cause unsaid.
                ok("default_model/%s_builds" % label, False,
                   "raised %r: %s" % (type(exc).__name__, exc))
                continue
            ok("default_model/%s_mcp" % label, mcp_router.default == want,
               repr(mcp_router.default))
            # The parity the README claims: one variable, one meaning on both surfaces.
            ok("default_model/%s_matches_serve" % label,
               mcp_router.default == serve_router.default,
               "mcp=%r serve=%r" % (mcp_router.default, serve_router.default))

        # What the setting decides, at `route()` rather than at the attribute. `route` is
        # documented as deciding "without loading or running anything", so nothing is downloaded.
        stock_mcp, stock_serve = build(None)
        nonenglish_mcp, nonenglish_serve = build("multilingual")
        for index, state in enumerate(("12345 !!!", "Quero cancelar")):
            ok("default_model/stock_%d" % index,
               stock_mcp.route(state).model == _stock == stock_serve.route(state).model,
               "mcp=%r serve=%r" % (stock_mcp.route(state).model, stock_serve.route(state).model))
            ok("default_model/fallback_multilingual_%d" % index,
               nonenglish_mcp.route(state).model == "multilingual"
               == nonenglish_serve.route(state).model,
               "mcp=%r serve=%r" % (nonenglish_mcp.route(state).model,
                                    nonenglish_serve.route(state).model))
        # A fallback, not a pin, on both surfaces.
        ok("default_model/placed_text_unaffected",
           nonenglish_mcp.route({"body": "Please refund the duplicate charge"}).model == "english"
           == nonenglish_serve.route({"body": "Please refund the duplicate charge"}).model)

        serve_message = mcp_message = ""
        os.environ["LAYA_DEFAULT_MODEL"] = "mutli-lingual"
        try:
            build_router()
            ok("default_model/serve_refuses_to_start", False, "build_router() returned a Router")
        except SystemExit as exc:
            serve_message = str(exc)
            ok("default_model/serve_refuses_to_start", True)
        except Exception as exc:  # noqa: BLE001 -- a traceback is not "exits with a message"
            ok("default_model/serve_refuses_to_start", False,
               "raised %r instead of exiting: %s" % (type(exc).__name__, exc))
        mcp_mod._ROUTER = None
        try:
            mcp_mod._ensure_router()
            ok("default_model/mcp_refuses", False, "_ensure_router() returned a Router")
        except ToolError as exc:
            mcp_message = exc.message
            ok("default_model/mcp_refuses_with_a_tool_error",
               exc.code == "internal_error", repr(exc.code))
        except Exception as exc:  # noqa: BLE001 -- anything else escapes the tool wrapper
            ok("default_model/mcp_refuses_with_a_tool_error", False,
               "raised %r instead of a ToolError, so the client gets a server crash" % type(exc).__name__)
        # One message, both surfaces: the operator fixes the typo from what they were told,
        # whether or not the server came up.
        ok("default_model/both_surfaces_name_the_variable_and_the_value",
           all("invalid LAYA_DEFAULT_MODEL 'mutli-lingual'" in m for m in (serve_message, mcp_message)),
           "serve=%r mcp=%r" % (serve_message, mcp_message))
        ok("default_model/both_surfaces_carry_core_s_words",
           all("unknown model" in m for m in (serve_message, mcp_message)),
           "serve=%r mcp=%r" % (serve_message, mcp_message))
        # The docstring promises a failed build is retriable rather than cached.
        ok("default_model/failed_build_is_not_cached", mcp_mod._ROUTER is None,
           repr(mcp_mod._ROUTER))
    finally:
        mcp_mod._ROUTER = saved_router
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_server_registration():
    tools = asyncio.run(mcp_server.list_tools())
    names = sorted(t.name for t in tools)
    ok("server/tool_names", names == ["laya_decide", "laya_predict", "laya_predict_batch",
                                      "laya_preset", "laya_route", "laya_route_batch",
                                      "laya_shortlist", "laya_status"], repr(names))
    decision = {"laya_predict", "laya_route", "laya_preset", "laya_shortlist",
                "laya_predict_batch", "laya_route_batch", "laya_decide"}
    for t in tools:
        desc = (t.description or "").lower()
        ok("server/desc_%s_nonempty" % t.name, bool(desc.strip()), repr(desc))
        # The guardrails constant is on the decision tools only;
        # laya_status reports instead of deciding.
        if t.name in decision:
            ok("server/desc_%s_guardrail" % t.name, "do not use" in desc)
        if t.name == "laya_predict":
            ok("server/desc_noul_labels", "optional labels" in desc)
        if t.name.endswith("_batch"):
            ok("server/desc_%s_requests" % t.name, "non-empty array" in desc)


def test_server_shortlist_k_default():
    """The shortlist tool's advertised `k` default must be `laya.shortlist`'s.

    `laya/mcp/server.py` registers the tool with `k: int = 20` written out as a literal,
    because the module stays importable without numpy -- `laya.shortlist` imports numpy at
    module level, so deriving the default from it would put numpy on the MCP server's import
    path. The literal is therefore necessary, and necessary hand-copies get a drift gate.

    Nothing guarded the copy: with `DEFAULT_SHORTLIST_K = 32` the tool layer, which resolves it,
    moved, and the MCP surface stayed on 20 with no test reporting the divergence. So both
    surfaces a client can observe -- the Python signature FastMCP registers and the `tools/list`
    schema it advertises -- are read against the library constant rather than against 20.
    """
    import inspect

    from laya.mcp import server as server_mod
    from laya.shortlist import DEFAULT_SHORTLIST_K

    params = inspect.signature(server_mod.laya_shortlist_tool).parameters
    ok("server/shortlist_k_default", params["k"].default == DEFAULT_SHORTLIST_K,
       "signature=%r library=%r" % (params["k"].default, DEFAULT_SHORTLIST_K))

    by_name = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
    schema = by_name["laya_shortlist"].input_schema \
        if hasattr(by_name["laya_shortlist"], "input_schema") \
        else by_name["laya_shortlist"].inputSchema
    advertised = schema.get("properties", {}).get("k", {}).get("default", "<absent>")
    ok("schema/shortlist_k_default", advertised == DEFAULT_SHORTLIST_K,
       "schema=%r library=%r" % (advertised, DEFAULT_SHORTLIST_K))


def test_shortlist_metadata_keys_are_documented():
    """Every key `predict_shortlist` publishes must be named by all three surfaces that list it.

    `laya/shortlist.py` writes a five-key block per question: ``labels``, ``scores``, ``k``,
    ``n`` and ``passthrough``. The tool description, the README and `docs/cli-mcp.md` each
    paraphrase that block, and all three stopped at four -- they dropped ``passthrough``, the
    one key that tells a caller whether the shortlist actually narrowed anything. A question
    whose option count is at or below ``k`` is answered whole with ``scores`` set to ``None``,
    so a client that assumes a real ranking was produced reads ``None`` as "no scores" rather
    than "nothing to rank".

    The key set is read out of `laya/shortlist.py`'s AST rather than typed here, so a sixth key
    added to the metadata fails this gate on all three pages at once.
    """
    import ast
    import re

    root = Path(__file__).resolve().parents[1]
    # `encoding="utf-8"` on every read below, for the reason recorded at the README read in the
    # batch-shape gate: `Path.read_text()` and `open()` default to the locale encoding, which is
    # cp1252 on the Windows runner, and these pages and `laya/shortlist.py` carry UTF-8.
    with open(root / "laya" / "shortlist.py", encoding="utf-8") as f:
        shortlist_src = f.read()
    # Scoped to predict_shortlist's own body: predict_tournament also builds a `meta[qid]` dict,
    # and a module-wide walk would depend on which one ast.walk happens to reach first.
    body = [node for node in ast.walk(ast.parse(shortlist_src))
            if isinstance(node, ast.FunctionDef) and node.name == "predict_shortlist"]
    meta_keys = None
    for node in ast.walk(body[0] if body else ast.parse("")):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "meta"):
            meta_keys = [ast.literal_eval(k) for k in node.value.keys]
            break
    ok("shortlist/meta_keys read from the AST",
       meta_keys == ["labels", "scores", "k", "n", "passthrough"],
       "AST read %r from laya/shortlist.py" % (meta_keys,))

    # Each key needs a concept word on the page; the paraphrase never uses the raw key names.
    REQUIRED = {
        "labels": r"label",
        "scores": r"score",
        "k": r"\bk\b",
        "n": r"option count",
        "passthrough": r"passed through|unshortlisted",
    }

    by_name = {t.name: t for t in asyncio.run(mcp_server.list_tools())}
    surfaces = {
        "tool description": by_name["laya_shortlist"].description,
        "README": (root / "README.md").read_text(encoding="utf-8"),
        "docs/cli-mcp.md": (root / "docs" / "cli-mcp.md").read_text(encoding="utf-8"),
    }
    for where, text in surfaces.items():
        flat = " ".join(text.split())
        for key, pattern in REQUIRED.items():
            ok("shortlist/%s names %s" % (where, key),
               re.search(pattern, flat) is not None,
               "the %s never spells the %r metadata key (looked for /%s/)" % (where, key, pattern))

    # The pre-fix wording stopped the list at the option count; a surface that closes the list
    # there again has dropped a key, whatever else it says.
    for where, text in surfaces.items():
        flat = " ".join(text.split())
        ok("shortlist/%s does not end the list at option count" % where,
           re.search(r"option count[),. ]*(?:The|the|for each| `k`|$)", flat) is None
           or re.search(r"option count, and whether", flat) is not None,
           "the list still closes on the option count: %s" % where)


# Called here rather than from the tail list, because that block is where three open PRs each
# append a line; one insertion point keeps this gate's diff to the lines it actually owns.
test_shortlist_metadata_keys_are_documented()


def test_cli_mcp_page_tool_argument_rows():
    """Every single-shot tool row on the MCP docs page names every parameter the handler takes.

    The "Main inputs" cell is a fourth copy of the shape, after the validator error message, each
    tool's registered description, and the README. The existing gate at
    `test_batch_item_shape_as_documented` covers the batch rows' per-request keys but does not
    read the six single-shot rows, so a whole control can fall off the table while the tool
    accepts it. The `laya_decide` row was in that state: the server registers
    `laya_decide_tool(state, schema, model='auto', min_confidence=None)` and the tool's own
    description spells out `min_confidence`, but the docs row read
    "`state`, `schema`, optional `model`" -- so a caller who trusts the table hand-rolls the
    abstention check on `values`, never learning the tool already nulls an unsure field.

    Each row's third cell is read straight from `docs/cli-mcp.md`, and every public parameter
    `inspect.signature()` sees on the registered handler must appear backticked in the cell.
    Batch tools are skipped because their cell uses a `{...}` item shape that the sibling gate
    above already holds against `batch_item_key_doc`. The pre-fix wording is banned by direct
    substring so the check fails if the `laya_decide` row reverts.
    """
    import inspect

    from laya.mcp import server as server_mod

    page = (Path(__file__).resolve().parents[1] / "docs" / "cli-mcp.md").read_text(encoding="utf-8")

    def row_cell(tool):
        for line in page.splitlines():
            if line.startswith("| `%s` |" % tool):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    return cells[2]
        return None

    for tool in ("laya_status", "laya_route", "laya_predict",
                 "laya_shortlist", "laya_preset", "laya_decide"):
        cell = row_cell(tool)
        ok("docs-mcp/args_row_%s_present" % tool, cell is not None, "row not found")
        if cell is None:
            continue
        handler = getattr(server_mod, "%s_tool" % tool)
        params = [
            p.name for p in inspect.signature(handler).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and not p.name.startswith("_")
            and p.name not in ("router", "preset_builder")  # server-injected deps
        ]
        missing = [p for p in params if "`%s`" % p not in cell]
        ok("docs-mcp/args_row_%s_names_every_param" % tool, not missing,
           "missing=%r cell=%r" % (missing, cell))

    ok("docs-mcp/args_row_decide_drops_the_three_param_version",
       "| `state`, `schema`, optional `model` |" not in page)


test_device()
test_real_device()
test_private_contract()
test_schema()
test_model_names()
test_presets()
test_model_forwarding()
test_shape()
test_question_forwarding()
test_option_order_forwarding()
test_shortlist()
test_batch_validation()
test_batch_predict()
test_batch_route()
test_decide()
test_shortlist_lang_parity()
test_shortlist_embed_cache()
test_controls_validation()
test_controls_predict()
test_controls_route()
test_controls_shortlist()
test_controls_preset()
test_controls_signature_and_schema()
test_lang_guess_control()
test_min_confidence_control()
test_question_validation_matches_the_agent()
test_a_bad_question_is_a_caller_error_not_a_server_fault()
test_timeout_removed()
test_models_from_env()
test_batch_item_shape_as_documented()
test_cli_mcp_page_batch_rows()
test_auto_task_env()
test_default_model_env()
test_server_registration()
test_server_shortlist_k_default()
test_cli_mcp_page_tool_argument_rows()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all mcp tests passed")
sys.exit(1 if FAIL else 0)
