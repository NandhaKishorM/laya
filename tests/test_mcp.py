"""MCP layer tests: schema, shape and registration. No model weights, no network.

Requires the mcp extra:  pip install "laya[mcp]"

Run: python tests/test_mcp.py
Skips cleanly (exit 0) when the mcp package is not installed, so the core
install keeps working.

Device and preload-list tests follow the laya.serve environment contract
(LAYA_DEVICE / LAYA_PRELOAD / LAYA_MODELS / LAYA_THREADS).
"""
import asyncio
import os
import sys
from pathlib import Path

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
    PRESETS,
    PRESET_ALIASES,
    ToolError,
    _overrides,
    get_available_presets,
    laya_predict,
    laya_preset,
    laya_route,
    laya_shortlist,
    laya_status,
    validate_budget,
    validate_lang,
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
    ok("device/fallback", resolve_device(None) in ("cuda", "cpu"))
    # laya.serve contract: LAYA_DEVICE goes verbatim to torch; the label is lowercased.
    old_dev = os.environ.get("LAYA_DEVICE")
    try:
        os.environ["LAYA_DEVICE"] = "cuda:1"
        ok("device/env_raw_for_torch", env_device() == "cuda:1")
        ok("device/env_label", resolve_device() == "cuda:1")
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
    out = validate_questions({"risk": {"type": "noul", "instructions": "is true?", "criteria": {"yes": "affirmative"}}})
    ok("schema/questions_noul_criteria_ok", "criteria" in out["risk"])

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


def test_question_forwarding():
    questions = {
        "intent": {"type": "choice", "instructions": "Which intent?",
                   "criteria": {"A": None, "B": {"desc": "billing"}}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", {"desc": "blocking"}]},
        "positive": {"type": "noul", "instructions": "Is this positive?",
                     "criteria": {"false": None, "true": "yes"},
                     "labels": {"false": "B", "true": "A"}},
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

    # Default k comes from laya.shortlist (20): a 3-option choice passes through.
    router = ShortlistRouter({"english": ShortlistAgent()})
    three = {"dept": {"type": "choice", "instructions": "pick",
                      "criteria": {"a": "A", "b": "B", "c": "C"}}}
    out = laya_shortlist(STATE, three, model="english", router=router, embed_fn=_raising_embed)
    ok("shortlist/default_k_passthrough", out["shortlist"]["dept"]["k"] == 20
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

    controls = ["task", "lang", "max_len", "head_max_len"]
    for fn, want in ((laya_predict, controls), (laya_shortlist, controls),
                     (laya_preset, controls), (laya_route, ["model", "task", "lang"])):
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
def test_timeout_removed():
    # The per-call timeout was removed: a ThreadPoolExecutor shutdown waits for
    # the work anyway, and MCP clients apply their own request timeout. The tool
    # functions no longer accept a timeout argument.
    import inspect

    import laya.mcp.tools as tools_mod

    for fn in (laya_predict, laya_route, laya_preset, laya_shortlist):
        ok("timeout/param_absent_%s" % fn.__name__, "timeout" not in inspect.signature(fn).parameters)
    ok("timeout/executor_absent", "ThreadPoolExecutor" not in inspect.getsource(tools_mod))


# --- server registration (schema only, no model load) ------------------------

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


def test_server_registration():
    tools = asyncio.run(mcp_server.list_tools())
    names = sorted(t.name for t in tools)
    ok("server/tool_names", names == ["laya_predict", "laya_preset", "laya_route", "laya_shortlist", "laya_status"], repr(names))
    decision = {"laya_predict", "laya_route", "laya_preset", "laya_shortlist"}
    for t in tools:
        desc = (t.description or "").lower()
        ok("server/desc_%s_nonempty" % t.name, bool(desc.strip()), repr(desc))
        # The guardrails constant is on the three decision tools only;
        # laya_status reports instead of deciding.
        if t.name in decision:
            ok("server/desc_%s_guardrail" % t.name, "do not use" in desc)
        if t.name == "laya_predict":
            ok("server/desc_noul_labels", "optional labels" in desc)


test_device()
test_real_device()
test_private_contract()
test_schema()
test_model_names()
test_presets()
test_model_forwarding()
test_shape()
test_question_forwarding()
test_shortlist()
test_controls_validation()
test_controls_predict()
test_controls_route()
test_controls_shortlist()
test_controls_preset()
test_controls_signature_and_schema()
test_timeout_removed()
test_models_from_env()
test_server_registration()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all mcp tests passed")
sys.exit(1 if FAIL else 0)
