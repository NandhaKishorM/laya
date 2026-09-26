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
    ToolError,
    laya_predict,
    laya_preset,
    laya_route,
    laya_shortlist,
    laya_status,
    validate_model,
    validate_preset,
    validate_questions,
    validate_state,
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
test_shape()
test_question_forwarding()
test_shortlist()
test_timeout_removed()
test_models_from_env()
test_server_registration()

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all mcp tests passed")
sys.exit(1 if FAIL else 0)
