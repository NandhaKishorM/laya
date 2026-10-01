"""Unit tests for Laya LlamaIndex integration.

Tests verify single/multi-selector logic, confidence threshold fallback gating,
query router dispatch, and LlamaIndex schema conventions without requiring
model downloads, GPU, or external services.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.llamaindex import (
    LayaLowConfidenceError,
    LayaMultiSelector,
    LayaQueryRouter,
    LayaSingleSelector,
    QueryBundle,
    ToolMetadata,
    _extract_query_str,
    _format_choices_criteria,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append(f"{name}:\n     got  {got!r}\n     want {want!r}")


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append(f"{name} {detail}")


# --------------------------------------------------------------- Mock Agent
class MockLayaAgent:
    """Mock agent returning deterministic responses for testing."""

    def __init__(self, response_fn):
        self.response_fn = response_fn

    def predict(self, state, questions, **kwargs):
        return self.response_fn(state, questions)


# --------------------------------------------------------------- 1. Query & Criteria Formatting
check("extract/str", _extract_query_str("What is the revenue?"), "What is the revenue?")
check(
    "extract/query_bundle",
    _extract_query_str(QueryBundle(query_str="How do I reset password?")),
    "How do I reset password?",
)


class CustomQueryObj:
    def __init__(self, text):
        self.query_str = text


check(
    "extract/custom_query_obj",
    _extract_query_str(CustomQueryObj("custom text")),
    "custom text",
)

# Test criteria formatting from ToolMetadata
tools = [
    ToolMetadata(name="vector_store", description="Search financial documents"),
    ToolMetadata(name="sql_db", description="Query relational tables"),
    {"name": "summary_tool", "description": "High level summarization"},
    "Simple string description tool",
]
criteria = _format_choices_criteria(tools)
check(
    "criteria/vector_store",
    criteria["choice_0"],
    "vector_store: Search financial documents",
)
check(
    "criteria/sql_db",
    criteria["choice_1"],
    "sql_db: Query relational tables",
)
check(
    "criteria/summary_tool",
    criteria["choice_2"],
    "summary_tool: High level summarization",
)
check(
    "criteria/string_tool",
    criteria["choice_3"],
    "Simple string description tool",
)


# --------------------------------------------------------------- 2. LayaSingleSelector
def mock_single_selector_response(state, questions):
    text = str(state).lower()
    if "sql" in text or "table" in text or "database" in text:
        chosen = "choice_1"
        conf = 0.94
    elif "lowconf" in text:
        chosen = "choice_0"
        conf = 0.42
    else:
        chosen = "choice_0"
        conf = 0.91

    return {
        "model": "mock-single",
        "answers": {
            "selector": {
                "choice": chosen,
                "answer_confidence": conf,
                "confidence": conf,
            }
        },
    }


mock_agent = MockLayaAgent(mock_single_selector_response)
single_selector = LayaSingleSelector(agent=mock_agent)

# Standard selection for choice_0
res0 = single_selector.select(tools, "Search for quarterly 10-K report")
check("single/index_0", res0.selections[0].index, 0)
check_true(
    "single/reason_0",
    "Selected 'vector_store' via Laya System 1 decision" in res0.selections[0].reason,
)

# Standard selection for choice_1 (sql)
res1 = single_selector.select(tools, QueryBundle(query_str="Query the SQL database for customers"))
check("single/index_1", res1.selections[0].index, 1)
check_true(
    "single/reason_1",
    "Selected 'sql_db' via Laya System 1 decision" in res1.selections[0].reason,
)

# Confidence threshold with fallback
gated_selector = LayaSingleSelector(
    agent=mock_agent,
    confidence_threshold=0.80,
    fallback_index=2,
)
res_fallback = gated_selector.select(tools, "lowconf query here")
check("single/fallback_index", res_fallback.selections[0].index, 2)
check_true(
    "single/fallback_reason",
    "Selected fallback index 2" in res_fallback.selections[0].reason,
)

# Confidence threshold with raise_on_low_confidence
raising_selector = LayaSingleSelector(
    agent=mock_agent,
    confidence_threshold=0.80,
    raise_on_low_confidence=True,
)
try:
    raising_selector.select(tools, "lowconf query here")
    check("single/raise_error", False, True)
except LayaLowConfidenceError as e:
    check("single/raise_error", True, True)
    check("single/error_conf", e.confidence, 0.42)
    check("single/error_thresh", e.threshold, 0.80)

# Empty choices validation
try:
    single_selector.select([], "any query")
    check("single/empty_choices", False, True)
except ValueError:
    check("single/empty_choices", True, True)


# Async single selector
async def run_async_single():
    res_async = await single_selector.aselect(tools, "Search the SQL database")
    check("single/async_index", res_async.selections[0].index, 1)


asyncio.run(run_async_single())


# --------------------------------------------------------------- 3. LayaMultiSelector
def mock_multi_selector_response(state, questions):
    text = str(state).lower()
    if "compare" in text or "both" in text:
        # High probability for both vector (0) and sql (1)
        probs = {"choice_0": 0.55, "choice_1": 0.40, "choice_2": 0.05}
    elif "none_qualify" in text:
        # All below threshold
        probs = {"choice_0": 0.15, "choice_1": 0.10, "choice_2": 0.05}
    else:
        probs = {"choice_0": 0.85, "choice_1": 0.10, "choice_2": 0.05}

    return {
        "model": "mock-multi",
        "answers": {
            "selector": {
                "choice": max(probs, key=probs.get),
                "probabilities": probs,
                "answer_confidence": 0.85,
            }
        },
    }


mock_multi_agent = MockLayaAgent(mock_multi_selector_response)
multi_selector = LayaMultiSelector(
    agent=mock_multi_agent,
    probability_threshold=0.30,
)

# Multi-select returns multiple tools
res_multi = multi_selector.select(tools[:3], "Compare both vector docs and SQL records")
check("multi/count", len(res_multi.selections), 2)
check("multi/idx_0", res_multi.selections[0].index, 0)
check("multi/idx_1", res_multi.selections[1].index, 1)

# Multi-select with max_outputs limit
limited_selector = LayaMultiSelector(
    agent=mock_multi_agent,
    probability_threshold=0.30,
    max_outputs=1,
)
res_limited = limited_selector.select(tools[:3], "Compare both vector docs and SQL records")
check("multi/limited_count", len(res_limited.selections), 1)
check("multi/limited_idx", res_limited.selections[0].index, 0)

# Multi-select fallback to top option when none meet threshold
res_none = multi_selector.select(tools[:3], "none_qualify search")
check("multi/fallback_top_count", len(res_none.selections), 1)
check("multi/fallback_top_idx", res_none.selections[0].index, 0)


# Async multi-selector
async def run_async_multi():
    res_async = await multi_selector.aselect(tools[:3], "Compare both sources")
    check("multi/async_count", len(res_async.selections), 2)


asyncio.run(run_async_multi())


# --------------------------------------------------------------- 4. LayaQueryRouter
class MockQueryEngine:
    def __init__(self, name):
        self.name = name

    def query(self, q):
        return f"Result from {self.name} for: {q}"

    async def aquery(self, q):
        return f"Async result from {self.name} for: {q}"


engines = {
    "vector": MockQueryEngine("VectorEngine"),
    "sql": MockQueryEngine("SqlEngine"),
    "summary": MockQueryEngine("SummaryEngine"),
}


def mock_router_dispatcher(state, questions):
    text = str(state).lower()
    if "sql" in text or "table" in text:
        chosen = "sql"
        conf = 0.96
    elif "summary" in text:
        chosen = "summary"
        conf = 0.91
    elif "lowconf" in text:
        chosen = "summary"
        conf = 0.35
    else:
        chosen = "vector"
        conf = 0.89

    return {
        "model": "mock-router",
        "answers": {
            "route": {
                "choice": chosen,
                "answer_confidence": conf,
                "confidence": conf,
            }
        },
    }


router_agent = MockLayaAgent(mock_router_dispatcher)
laya_qr = LayaQueryRouter(
    query_engines=engines,
    descriptions={
        "vector": "Semantic search over documentation",
        "sql": "Relational SQL queries",
        "summary": "Document summarization",
    },
    agent=router_agent,
    confidence_threshold=0.70,
    fallback_key="vector",
)

# Route direct
check("router/route_sql", laya_qr.route("Run a query on the SQL table"), "sql")
check("router/route_summary", laya_qr.route("Provide an executive summary"), "summary")
check("router/route_vector", laya_qr.route("How does authentication work?"), "vector")

# Confidence fallback
check("router/route_fallback", laya_qr.route("lowconf request"), "vector")

# Execute query()
res_q = laya_qr.query("Run a query on the SQL table")
check("router/query_exec", res_q, "Result from SqlEngine for: Run a query on the SQL table")


# Async query
async def run_async_router():
    res_aq = await laya_qr.aquery("Provide an executive summary")
    check("router/aquery_exec", res_aq, "Async result from SummaryEngine for: Provide an executive summary")


asyncio.run(run_async_router())


# Callable engine support
callable_engines = {
    "calc": lambda q: f"Calculated: {q}",
    "search": lambda q: f"Searched: {q}",
}
callable_qr = LayaQueryRouter(
    query_engines=callable_engines,
    agent=MockLayaAgent(
        lambda s, q: {"answers": {"route": {"choice": "calc", "answer_confidence": 0.99}}}
    ),
)
check("router/callable", callable_qr.query("2 + 2"), "Calculated: 2 + 2")


# --------------------------------------------------------------- 5. Remote HTTP Execution (Mocked)
from unittest.mock import MagicMock, patch


class DummyHTTPResponse:
    def __init__(self, data_dict):
        self.data = json.dumps(data_dict).encode("utf-8")

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


import json

remote_response = {
    "model": "laya-multilingual",
    "answers": {
        "selector": {
            "choice": "choice_1",
            "answer_confidence": 0.98,
            "confidence": 0.98,
        }
    },
}

with patch("urllib.request.build_opener") as mock_build_opener:
    mock_opener = MagicMock()
    mock_opener.open.return_value = DummyHTTPResponse(remote_response)
    mock_build_opener.return_value = mock_opener

    remote_selector = LayaSingleSelector(
        base_url="http://localhost:8000",
        api_key="sk-test-token",
    )
    res_remote = remote_selector.select(tools, "Query via remote server")
    check("remote/index", res_remote.selections[0].index, 1)

    # Check request headers and URL
    call_args = mock_opener.open.call_args
    req = call_args[0][0]
    check("remote/url", req.full_url, "http://localhost:8000/v1/systemone")
    check("remote/auth", req.headers.get("Authorization"), "Bearer sk-test-token")


# --------------------------------------------------------------- 6. Per-call decision controls
#
# `laya.integrations.langchain` has forwarded `max_len` / `head_max_len` and the five hook
# arguments since #530 / #532; this surface forwarded only `model`, so a RAG pipeline choosing
# between more tools than the default head budget fits could not widen its own window. The rule
# now lives in one module the three integrations import, and the lists below are read out of it
# rather than written out again here -- a control added to `._controls` without reaching this file
# fails here. Three classes, three `_execute_decision` call sites, so every surface is driven.
import inspect
from laya.agent import Agent
from laya.integrations import _controls
from laya.integrations import crewai as crewai_module
from laya.integrations import langchain as langchain_module
from laya.integrations import llamaindex as llamaindex_module
from laya.router import Router

CONTROLS = (tuple(_controls.PREDICT_CONTROLS) + tuple(_controls.DECISION_CONTROLS)
            + tuple(_controls.HOOK_CONTROLS))


def _params(fn):
    return set(inspect.signature(fn).parameters)


check("controls/budget tuple names budget_kwargs",
      set(_params(_controls.budget_kwargs)), set(_controls.PREDICT_CONTROLS))
check("controls/decision tuple names decision_kwargs",
      set(_params(_controls.decision_kwargs)), set(_controls.DECISION_CONTROLS))
check("controls/hook tuple names hook_kwargs",
      set(_params(_controls.hook_kwargs)), set(_controls.HOOK_CONTROLS))

# Everything called a control here is an argument a real runner accepts, or a selector would fail
# with a TypeError deep inside core instead of at the call site.
_agent_params = _params(Agent.system_one)
_router_params = _params(Router.predict)
for _c in CONTROLS:
    check_true("controls/%s accepted by Agent" % _c, _c in _agent_params)
    check_true("controls/%s accepted by Router.predict" % _c, _c in _router_params)

for cls in (LayaSingleSelector, LayaMultiSelector, LayaQueryRouter):
    for _c in CONTROLS:
        check_true("controls/%s takes %s" % (cls.__name__, _c), _c in _params(cls.__init__))
check("controls/_execute_decision takes every control",
      set(_params(llamaindex_module._execute_decision)) - {"state", "questions", "agent",
                                                           "base_url", "api_key", "model"},
      set(CONTROLS))

# The three wrappers end at the same runner call, so they must accept the same controls.
for _mod in (langchain_module, crewai_module):
    check("controls/%s agrees with llamaindex" % _mod.__name__.rsplit(".", 1)[-1],
          set(_params(_mod._execute_decision)),
          set(_params(llamaindex_module._execute_decision)))


class RecordingAgent:
    """A runner that keeps the kwargs of every call, the way a pipeline would have to be debugged."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append(kwargs)
        return {
            "answers": {
                "selector": {"choice": "choice_0", "confidence": 0.9, "answer_confidence": 0.9,
                             "probabilities": {"choice_0": 0.9, "choice_1": 0.4}},
                "route": {"choice": "sql", "confidence": 0.9, "answer_confidence": 0.9},
            },
            "routing": {"model": "english", "repo": None, "reason": "explicit model"},
        }


ALL_CONTROLS = {"max_len": 1024, "head_max_len": 512, "lang": "fr", "min_confidence": 0.4,
                "hooks": ["H"], "on_predict_start": "S",
                "on_predict_end": "E", "hooks_raise": True, "hooks_timeout": 0.5}

# One entry per class that can take a decision here: (label, class, constructor extras, one call).
SURFACES = [
    ("single", LayaSingleSelector, {},
     lambda s: s.select(tools, "Run a query on the SQL table")),
    ("multi", LayaMultiSelector, {},
     lambda s: s.select(tools, "Run a query on the SQL table")),
    ("query_router", LayaQueryRouter, {"query_engines": engines},
     lambda s: s.route("Run a query on the SQL table")),
]

HOOK_SAMPLES = (("hooks", [object()]), ("on_predict_start", object()),
                ("on_predict_end", object()), ("hooks_raise", False), ("hooks_timeout", 0.5))


def local_kwargs(cls, extra, run, **controls):
    """Run one decision on a fresh surface, or report the surface's own refusal.

    The broad except is deliberate: a wrapper that cannot even be built or that drops an attribute
    has to show up as a named mismatch, not as a traceback that hides every later check.
    """
    agent = RecordingAgent()
    try:
        run(cls(agent=agent, **dict(extra, **controls)))
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return agent.calls[0]


def remote_body(cls, extra, run, **controls):
    """POST through the real urllib path with the opener mocked, and return the JSON body sent."""
    with patch("urllib.request.build_opener") as mock_build_opener:
        mock_opener = MagicMock()
        mock_opener.open.return_value = DummyHTTPResponse(remote_response)
        mock_build_opener.return_value = mock_opener
        run(cls(base_url="http://localhost:8000", **dict(extra, **controls)))
        return json.loads(mock_opener.open.call_args[0][0].data)


for _label, _cls, _extra, _run in SURFACES:
    check("controls/%s with nothing set sends nothing" % _label,
          local_kwargs(_cls, _extra, _run), {})
    check("controls/%s forwards every control" % _label,
          local_kwargs(_cls, _extra, _run, **ALL_CONTROLS), ALL_CONTROLS)
    check("controls/%s forwards one budget alone" % _label,
          local_kwargs(_cls, _extra, _run, head_max_len=256), {"head_max_len": 256})
    check("controls/%s forwards lang alone" % _label,
          local_kwargs(_cls, _extra, _run, lang="fr"), {"lang": "fr"})
    check("controls/%s forwards min_confidence alone" % _label,
          local_kwargs(_cls, _extra, _run, min_confidence=0.4), {"min_confidence": 0.4})
    # 0 and [] are decisions, not absences: truthiness tests here would drop them.
    check("controls/%s keeps falsy values" % _label,
          local_kwargs(_cls, _extra, _run, head_max_len=0, hooks=[], hooks_raise=False,
                       min_confidence=0.0),
          {"head_max_len": 0, "hooks": [], "hooks_raise": False, "min_confidence": 0.0})
    check("controls/%s alongside model" % _label,
          local_kwargs(_cls, _extra, _run, model="laya-multilingual", max_len=1024),
          {"model": "laya-multilingual", "max_len": 1024})
    check("controls/%s remote body carries both budgets" % _label,
          {k: v for k, v in remote_body(_cls, _extra, _run, max_len=1024,
                                        head_max_len=384).items()
           if k in _controls.PREDICT_CONTROLS},
          {"max_len": 1024, "head_max_len": 384})
    check("controls/%s remote body omits unset budgets" % _label,
          [k for k in remote_body(_cls, _extra, _run) if k in _controls.PREDICT_CONTROLS], [])
    check("controls/%s remote body keeps a zero" % _label,
          remote_body(_cls, _extra, _run, head_max_len=0).get("head_max_len"), 0)
    # `lang` / `min_confidence` are laya-serve `BODY_CONTROLS`, so the same override reaches the
    # remote node -- and an unset one stays out of the body rather than shadowing the deployment.
    check("controls/%s remote body carries the decision controls" % _label,
          {k: v for k, v in remote_body(_cls, _extra, _run, lang="es", min_confidence=0.3).items()
           if k in _controls.DECISION_CONTROLS},
          {"lang": "es", "min_confidence": 0.3})
    check("controls/%s remote body omits unset decision controls" % _label,
          [k for k in remote_body(_cls, _extra, _run) if k in _controls.DECISION_CONTROLS], [])
    check("controls/%s remote body keeps min_confidence=0.0" % _label,
          remote_body(_cls, _extra, _run, min_confidence=0.0).get("min_confidence"), 0.0)
    # A hook is a Python callable that runs inside `predict`; a serve node cannot receive one.
    for _c, _sample in HOOK_SAMPLES:
        try:
            remote_body(_cls, _extra, _run, **{_c: _sample})
            check_true("controls/%s remote refuses %s" % (_label, _c), False, "no error raised")
        except ValueError as exc:
            check_true("controls/%s remote refuses %s" % (_label, _c),
                       _c in str(exc) and "laya-serve" in str(exc))
        except Exception as exc:
            check_true("controls/%s remote refuses %s" % (_label, _c), False, type(exc).__name__)

# The caller's own hook objects have to arrive, not a copy or a re-wrapped stand-in. Read with
# `.get()` so a surface that drops them reports a named failure instead of a KeyError that hides
# the rest of the run.
_sentinel_hooks = [RecordingAgent()]
_ident_agent = RecordingAgent()
LayaSingleSelector(agent=_ident_agent, hooks=_sentinel_hooks).select(
    tools, "Run a query on the SQL table")
_seen_hooks = _ident_agent.calls[0].get("hooks")
check_true("controls/forwards the caller's objects",
           _seen_hooks is _sentinel_hooks and _seen_hooks[0] is _sentinel_hooks[0], repr(_seen_hooks))


# --------------------------------------------------------------- Confidence source
# Every gate in this module reads the same number: the calibrated `answer_confidence`
# core gates on, never the entropy `confidence` (see `laya.confidence`). The two are on
# different scales, so they must not be interchangeable here.


class DisagreeingSelectorAgent(RecordingAgent):
    """Answers whose two confidence fields disagree, on purpose."""

    def __init__(self, response_fn=None):
        super().__init__()
        self.response_fn = response_fn

    def predict(self, state, questions, **kwargs):
        self.calls.append(kwargs)
        if self.response_fn is not None:
            return self.response_fn(state, questions)
        return super().predict(state, questions, **kwargs)


def _disagree(choice_answer):
    def response(state, questions):
        return {"model": "m",
                "answers": {"selector": dict({"choice": "choice_0"}, **choice_answer),
                            "route": {"choice": "sql", "confidence": 0.9, "answer_confidence": 0.9}}}
    return DisagreeingSelectorAgent(response)


# Below the calibrated threshold but above the entropy one: gates on the calibrated number.
low_agent = _disagree({"confidence": 0.95, "answer_confidence": 0.4})
low_sel = LayaSingleSelector(agent=low_agent, confidence_threshold=0.80, fallback_index=1).select(
    tools, "anything")
check("gate/reads calibrated not entropy",
      low_sel.selections[0].index if hasattr(low_sel, "selections") else low_sel, 1)
try:
    LayaSingleSelector(agent=_disagree({"confidence": 0.95, "answer_confidence": 0.4}),
                       confidence_threshold=0.80,
                       raise_on_low_confidence=True).select(tools, "anything")
    check_true("gate/raises on the calibrated number", False, "no error raised")
except LayaLowConfidenceError as err:
    check("gate/error carries the calibrated number", err.confidence, 0.4)

# An answer with no usable confidence keeps the old behaviour: treated as fully confident.
def _silent_selector(state, questions):
    return {"model": "m",
            "answers": {"selector": {"choice": "choice_2"},
                        "route": {"choice": "sql", "confidence": 0.9, "answer_confidence": 0.9}}}


kept = LayaSingleSelector(agent=DisagreeingSelectorAgent(_silent_selector)).select(tools, "anything")
check("gate/missing confidence still passes",
      kept.selections[0].index if hasattr(kept, "selections") else kept, 2)


# --------------------------------------------------------------- Results Summary
print(f"PASS: {len(PASS)}")
print(f"FAIL: {len(FAIL)}")
for f in FAIL:
    print(f"  - {f}")

if FAIL:
    sys.exit(1)
