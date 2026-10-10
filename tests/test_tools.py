"""Unit tests for Laya Tool and Function Selection integration.

Tests verify tool metadata extraction, criteria formatting, confidence threshold
fallback gating, direct execution, remote HTTP payload construction, and async
selection without requiring model weights, GPU, or external network access.
"""
import asyncio
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya
from laya.integrations._controls import (
    DECISION_CONTROLS,
    HOOK_CONTROLS,
    PREDICT_CONTROLS,
)
from laya.integrations.tools import (
    LayaLowConfidenceError,
    LayaToolSelector,
    _extract_tool_metadata,
    _format_tools_criteria,
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
        self.calls = []

    def predict(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, "kwargs": kwargs})
        return self.response_fn(state, questions)


# --------------------------------------------------------------- 1. Tool Metadata Extraction
# String tools
check("extract/str_colon", _extract_tool_metadata("search: Search web", 0), ("search", "Search web"))
check("extract/str_plain", _extract_tool_metadata("calculator", 1), ("calculator", "calculator"))

# Dict tools
check("extract/dict", _extract_tool_metadata({"name": "sql", "description": "Run SQL query"}, 0),
      ("sql", "Run SQL query"))
check("extract/dict_function",
      _extract_tool_metadata({"type": "function", "function": {"name": "get_stock", "description": "Fetch stock price"}}, 0),
      ("get_stock", "Fetch stock price"))

# Class / Object tools (LangChain / CrewAI style)
class DummyLangChainTool:
    name = "web_search"
    description = "Searches Google for query"

check("extract/obj_attrs", _extract_tool_metadata(DummyLangChainTool(), 0),
      ("web_search", "Searches Google for query"))

# LlamaIndex style with metadata
class DummyLlamaIndexMeta:
    name = "rag_search"
    description = "Search vector index"

class DummyLlamaIndexTool:
    metadata = DummyLlamaIndexMeta()

check("extract/llama_meta", _extract_tool_metadata(DummyLlamaIndexTool(), 0),
      ("rag_search", "Search vector index"))

# Python callable function
def calculate_tax(amount: float) -> float:
    """Calculate standard state tax."""
    return amount * 0.08

check("extract/function", _extract_tool_metadata(calculate_tax, 0),
      ("calculate_tax", "Calculate standard state tax."))


# --------------------------------------------------------------- 2. Criteria Formatting
tools_list = [
    DummyLangChainTool(),
    calculate_tax,
    {"name": "db_lookup", "description": "Query Postgres database"},
]

criteria_no_direct, lookup_no_direct = _format_tools_criteria(tools_list, allow_direct_answer=False)
check("criteria/count", len(criteria_no_direct), 3)
check("criteria/tool_0", criteria_no_direct["tool_0"], "web_search: Searches Google for query")
check("criteria/tool_1", criteria_no_direct["tool_1"], "calculate_tax: Calculate standard state tax.")
check("criteria/tool_2", criteria_no_direct["tool_2"], "db_lookup: Query Postgres database")
check("lookup/tool_0", lookup_no_direct["tool_0"][2], "web_search")

criteria_direct, lookup_direct = _format_tools_criteria(tools_list, allow_direct_answer=True)
check("criteria/direct_count", len(criteria_direct), 4)
check_true("criteria/has_direct", "direct_answer" in criteria_direct)
check("lookup/direct_index", lookup_direct["direct_answer"][0], -1)


# --------------------------------------------------------------- 3. Tool Selection Logic
def make_mock_response(chosen_key, confidence=0.92, probabilities=None):
    if probabilities is None:
        probabilities = {chosen_key: confidence, "tool_1": 0.05, "tool_2": 0.03}
    return {
        "model": "laya-multilingual",
        "answers": {
            "tool_select": {
                "type": "choice",
                "choice": chosen_key,
                "confidence": confidence,
                "answer_confidence": confidence,
                "probabilities": probabilities,
            }
        },
    }

mock_agent = MockLayaAgent(lambda s, q: make_mock_response("tool_0", confidence=0.95))
selector = LayaToolSelector(tools=tools_list, agent=mock_agent)

decision = selector.select("Find latest news on quantum computing")
check("select/tool_name", decision.tool_name, "web_search")
check("select/tool_index", decision.tool_index, 0)
check("select/confidence", decision.confidence, 0.95)
check("select/is_direct_answer", decision.is_direct_answer, False)
check("select/is_fallback", decision.is_fallback, False)
check_true("select/reason", "Selected tool 'web_search'" in decision.reason)
check_true("select/probabilities_mapped", "web_search" in decision.probabilities)

# Direct answer selection
mock_agent_direct = MockLayaAgent(
    lambda s, q: make_mock_response("direct_answer", confidence=0.99,
                                    probabilities={"direct_answer": 0.99, "tool_0": 0.01})
)
selector_direct = LayaToolSelector(
    tools=tools_list,
    allow_direct_answer=True,
    agent=mock_agent_direct,
)
decision_direct = selector_direct.select("Hello, how are you today?")
check("select/direct_name", decision_direct.tool_name, "__direct_answer__")
check("select/direct_index", decision_direct.tool_index, -1)
check("select/direct_tool_is_none", decision_direct.tool, None)
check("select/direct_flag", decision_direct.is_direct_answer, True)


# --------------------------------------------------------------- 4. Confidence Threshold & Fallbacks
# Below threshold without fallback raises LayaLowConfidenceError
mock_agent_low = MockLayaAgent(lambda s, q: make_mock_response("tool_0", confidence=0.45))
selector_low = LayaToolSelector(tools=tools_list, confidence_threshold=0.80, agent=mock_agent_low)

try:
    selector_low.select("Obscure question")
    FAIL.append("threshold/should_raise: expected LayaLowConfidenceError")
except LayaLowConfidenceError as e:
    PASS.append("threshold/raises_error")
    check("threshold/error_conf", e.confidence, 0.45)
    check("threshold/error_thresh", e.threshold, 0.80)

# Below threshold with fallback tool
fallback_tool = {"name": "general_llm_fallback", "description": "General conversational fallback"}
selector_fb = LayaToolSelector(
    tools=tools_list,
    confidence_threshold=0.80,
    fallback_tool=fallback_tool,
    agent=mock_agent_low,
)
decision_fb = selector_fb.select("Obscure question")
check("threshold/fallback_used", decision_fb.is_fallback, True)
check("threshold/fallback_tool_name", decision_fb.tool_name, "general_llm_fallback")
check("threshold/fallback_tool_obj", decision_fb.tool, fallback_tool)
check("threshold/fallback_index", decision_fb.tool_index, -1)


# --------------------------------------------------------------- 4b. Threshold Validation & Fail-Closed Behavior
# Validation of confidence_threshold to [0.0, 1.0]
for bad_thresh in (-0.1, 1.05, 2.0, True, False, float("nan"), "0.5"):
    try:
        LayaToolSelector(tools=tools_list, confidence_threshold=bad_thresh)
        FAIL.append(f"threshold_val/bad_{bad_thresh!r}: expected ValueError")
    except ValueError:
        PASS.append(f"threshold_val/rejects_{bad_thresh!r}")

check("threshold_val/accepts_zero", LayaToolSelector(tools=tools_list, confidence_threshold=0.0).confidence_threshold, 0.0)
check("threshold_val/accepts_one", LayaToolSelector(tools=tools_list, confidence_threshold=1.0).confidence_threshold, 1.0)
check("threshold_val/accepts_float", LayaToolSelector(tools=tools_list, confidence_threshold=0.75).confidence_threshold, 0.75)

# Fail-closed: empty answers object must raise rather than selecting tool_0
empty_agent = MockLayaAgent(lambda s, q: {"model": "laya-multilingual", "answers": {}})
selector_empty = LayaToolSelector(tools=tools_list, agent=empty_agent)
try:
    selector_empty.select("Query")
    FAIL.append("fail_closed/empty_answers: expected RuntimeError")
except RuntimeError as e:
    PASS.append("fail_closed/empty_answers_raises")
    check_true("fail_closed/empty_answers_msg", "Malformed or missing decision response" in str(e))

# Fail-closed with fallback: empty answers routes to fallback tool
selector_empty_fb = LayaToolSelector(tools=tools_list, fallback_tool=fallback_tool, agent=empty_agent)
dec_empty_fb = selector_empty_fb.select("Query")
check("fail_closed/empty_fallback_used", dec_empty_fb.is_fallback, True)
check("fail_closed/empty_fallback_name", dec_empty_fb.tool_name, "general_llm_fallback")

# Fail-closed: missing / unknown choice key
unknown_choice_agent = MockLayaAgent(lambda s, q: {"model": "laya-multilingual", "answers": {"tool_select": {"choice": "unknown_tool", "confidence": 0.99}}})
selector_unknown = LayaToolSelector(tools=tools_list, agent=unknown_choice_agent)
try:
    selector_unknown.select("Query")
    FAIL.append("fail_closed/unknown_choice: expected RuntimeError")
except RuntimeError as e:
    PASS.append("fail_closed/unknown_choice_raises")
    check_true("fail_closed/unknown_choice_msg", "choice 'unknown_tool' not in candidate tools" in str(e))

selector_unknown_fb = LayaToolSelector(tools=tools_list, fallback_tool=fallback_tool, agent=unknown_choice_agent)
dec_unknown_fb = selector_unknown_fb.select("Query")
check("fail_closed/unknown_choice_fallback_used", dec_unknown_fb.is_fallback, True)

# Fail-closed: unusable / None / NaN confidence must fail closed
nan_conf_agent = MockLayaAgent(lambda s, q: {
    "model": "laya-multilingual",
    "answers": {
        "tool_select": {
            "choice": "tool_0",
            "confidence": float("nan"),
            "answer_confidence": None,
        }
    }
})
selector_nan = LayaToolSelector(tools=tools_list, agent=nan_conf_agent)
try:
    selector_nan.select("Query")
    FAIL.append("fail_closed/nan_confidence: expected LayaLowConfidenceError")
except LayaLowConfidenceError:
    PASS.append("fail_closed/nan_confidence_raises")

selector_nan_fb = LayaToolSelector(tools=tools_list, fallback_tool=fallback_tool, agent=nan_conf_agent)
dec_nan_fb = selector_nan_fb.select("Query")
check("fail_closed/nan_confidence_fallback_used", dec_nan_fb.is_fallback, True)

# Fail-closed protects executable call(): never runs tool_0 on missing/malformed/unknown decisions
tool_exec_counter = {"calls": 0}
def dangerous_tool():
    tool_exec_counter["calls"] += 1
    return "executed"

selector_safe_call = LayaToolSelector(tools=[dangerous_tool], agent=empty_agent)
try:
    selector_safe_call.call("Query")
    FAIL.append("fail_closed/call_should_not_execute: expected RuntimeError")
except RuntimeError:
    PASS.append("fail_closed/call_prevented_execution")
check("fail_closed/tool_not_called", tool_exec_counter["calls"], 0)


# --------------------------------------------------------------- 5. Direct Execution (call)
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

mock_agent_calc = MockLayaAgent(lambda s, q: make_mock_response("tool_0", confidence=0.99))
selector_calc = LayaToolSelector(tools=[add_numbers], agent=mock_agent_calc)
calc_result = selector_calc.call("Calculate 10 + 25", None, 10, 25)
check("call/callable_execution", calc_result, 35)

# Direct answer returns None on call()
selector_call_direct = LayaToolSelector(tools=[add_numbers], allow_direct_answer=True, agent=mock_agent_direct)
direct_call_res = selector_call_direct.call("Hello", None, 1, 2)
check("call/direct_answer_returns_none", direct_call_res, None)


# --------------------------------------------------------------- 6. Async Selection (aselect)
async def test_async_select():
    sel = LayaToolSelector(tools=tools_list, agent=mock_agent)
    res = await sel.aselect("Search quantum computing")
    check("async/tool_name", res.tool_name, "web_search")

asyncio.run(test_async_select())


# --------------------------------------------------------------- 7. LCEL / Runnable Interface
runnable = selector.as_runnable()
runnable_res = runnable.invoke("Search quantum computing")
check("runnable/tool_name", runnable_res.tool_name, "web_search")

# Chain with pipe operator
chained = runnable | (lambda d: f"Calling: {d.tool_name}")
check("runnable/chain", chained.invoke("Search quantum computing"), "Calling: web_search")


# --------------------------------------------------------------- 8. Remote HTTP Mode & Hook Rejection
mock_http_response = json.dumps(make_mock_response("tool_0", confidence=0.93)).encode("utf-8")

class MockHTTPResponse:
    def read(self):
        return mock_http_response
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class MockHTTPHandler:
    def open(self, req, timeout=10.0):
        body = json.loads(req.data.decode("utf-8"))
        if "questions" not in body or "state" not in body:
            raise ValueError("Malformed request body")
        return MockHTTPResponse()

original_build_opener = laya.integrations.tools.urllib.request.build_opener
laya.integrations.tools.urllib.request.build_opener = lambda *args: MockHTTPHandler()

try:
    remote_selector = LayaToolSelector(
        tools=tools_list,
        base_url="http://localhost:8000",
        api_key="test-key",
        max_len=1024,
        head_max_len=256,
    )
    remote_decision = remote_selector.select("Search quantum computing")
    check("remote/tool_name", remote_decision.tool_name, "web_search")
    check("remote/confidence", remote_decision.confidence, 0.93)

    # Rejection of local hooks over remote
    remote_hooks_selector = LayaToolSelector(
        tools=tools_list,
        base_url="http://localhost:8000",
        hooks=["mock_hook"],
    )
    try:
        remote_hooks_selector.select("Query")
        FAIL.append("remote/hooks: expected ValueError")
    except ValueError as e:
        PASS.append("remote/hooks_rejected")
        check_true("remote/hooks_msg", "run in the local runner and cannot be sent" in str(e))
finally:
    laya.integrations.tools.urllib.request.build_opener = original_build_opener


# --------------------------------------------------------------- 9. Controls Signatures Check
params = inspect.signature(LayaToolSelector.__init__).parameters
for ctrl in PREDICT_CONTROLS:
    check_true(f"controls/predict_{ctrl}", ctrl in params)
for ctrl in DECISION_CONTROLS:
    check_true(f"controls/decision_{ctrl}", ctrl in params)
for ctrl in HOOK_CONTROLS:
    check_true(f"controls/hook_{ctrl}", ctrl in params)


# --------------------------------------------------------------- 10. Public Module Exports
integrations_mod = __import__("laya.integrations", fromlist=["__all__"])
check_true("export/integrations_selector", "LayaToolSelector" in integrations_mod.__all__)
check_true("export/integrations_decision", "ToolRouteDecision" in integrations_mod.__all__)
tools_mod = __import__("laya.integrations.tools", fromlist=["__all__"])
check_true("export/tools_selector", "LayaToolSelector" in tools_mod.__all__)
check_true("export/tools_decision", "ToolRouteDecision" in tools_mod.__all__)
check_true("export/top_level_clean", "LayaToolSelector" not in laya.__all__ and "ToolRouteDecision" not in laya.__all__)


# --------------------------------------------------------------- Report
print("\n" + "=" * 50)
print(f"LayaToolSelector Test Suite: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 50)

if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All tests passed cleanly!")
    sys.exit(0)
