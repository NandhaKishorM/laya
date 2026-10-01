"""The shape `pip install "laya[langchain]"` actually produces.

This PR installs the `langchain` extra in the `tests` job so that `tests/test_langchain.py` runs
against real Runnables. Nothing in the job proves that install happened, and the suite cannot tell
either: measured on this tree, `tests/test_langchain.py` prints `FAIL: 0` and exits 0 both ways --
`PASS: 87` with langchain-core, `PASS: 80` without it. The eight checks inside its trailing
`if _RUNNABLE_AVAILABLE:` block do not fail when the extra is missing, they disappear, and the
`else:` branch records one `PASS` for having skipped them. So the next edit to that install line, or
an extra that stops resolving, moves the whole LangChain surface back onto
`RunnableSerializable = object` and the job stays green.

The two columns disagree about what a router does:

    confidence_threshold='0.8'    real: 0.8 (coerced)   shim: '0.8' (kept as a str)
    .invoke() at that threshold   real: 'human'         shim: TypeError: '>' not supported
                                                        between instances of 'str' and 'float'
    criteria={'billing_agent': 1} real: ValidationError shim: accepted
    instructions=None             real: ValidationError shim: accepted

This file asserts the real column. It fails rather than skipping when langchain-core is absent -- a
skip is the failure mode described above. The batch behaviour this PR adds is checked in both
columns by `tests/test_langchain.py`; what only a real install can check is here: the pydantic
fields that gate a decision, and being a `Runnable` at all.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.langchain import (
    LayaEvaluator,
    LayaGuardrail,
    LayaRouter,
    LayaTriage,
    RunnableSerializable,
    _RUNNABLE_AVAILABLE,
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


def check_raises(name, fn, kind):
    try:
        value = fn()
    except Exception as error:
        check_true(name, isinstance(error, kind),
                   f"(raised {type(error).__name__}: {error}, wanted {kind.__name__})")
        return
    FAIL.append(f"{name}: accepted {value!r}, wanted {kind.__name__}")


def outcome(fn):
    """Report a failure instead of raising out of the suite: the fallback base class turns a
    coerced threshold into a TypeError at decision time, and that is a result to print."""
    try:
        return fn()
    except Exception as error:
        return f"{type(error).__name__}: {error}"


if not _RUNNABLE_AVAILABLE or RunnableSerializable is object:
    print("FAIL: 1")
    print("FAILED: langchain-core is not installed, so laya.integrations.langchain fell back to "
          "`RunnableSerializable = object`. Install the extra this suite is for: "
          "pip install -e \".[langchain]\" (docs/langchain.md: pip install \"laya[langchain]\")")
    sys.exit(1)

from langchain_core.runnables import Runnable  # noqa: E402
from pydantic import ValidationError  # noqa: E402


# --------------------------------------------------------------- Mock Agent
class MockAgent:
    """Answers every question with the same choice at a fixed confidence, no weights."""

    def __init__(self, confidence):
        self.confidence = confidence

    def predict(self, state, questions, **kwargs):
        return {"answers": {qid: {"choice": "billing_agent", "confidence": self.confidence,
                                  "answer_confidence": self.confidence,
                                  "urgency": 1, "needs_human": False}
                            for qid in questions}}


low, high = MockAgent(0.5), MockAgent(0.9)
CRITERIA = {"billing_agent": "invoices, refunds, billing", "tech": "bugs, crashes, errors"}

# --------------------------------------------------------------- 1. The base class is real
for cls in (LayaRouter, LayaGuardrail, LayaTriage, LayaEvaluator):
    check_true(f"base/{cls.__name__}", issubclass(cls, RunnableSerializable),
               f"(mro is {cls.__mro__[1].__name__}, not RunnableSerializable)")
    check_true(f"runnable/{cls.__name__}", issubclass(cls, Runnable),
               f"(not a langchain_core Runnable: mro {cls.__mro__[1].__name__})")
    check_true(f"runnable-api/{cls.__name__}", hasattr(cls, "batch"),
               "(a Runnable gets .batch() from its base)")

# --------------------------------------------------------------- 2. Fields validate and coerce
ROUTER_KW = dict(criteria=CRITERIA, confidence_threshold="0.8", fallback="human")
router = outcome(lambda: LayaRouter(agent=low, **ROUTER_KW))
check_true("construct/accepts-a-string-threshold", not isinstance(router, str),
           f"(got {router!r})")
check("coerce/threshold-type", outcome(lambda: type(router.confidence_threshold).__name__), "float")
check("coerce/threshold-value", outcome(lambda: router.confidence_threshold), 0.8)
check_raises("validate/criteria-int",
             lambda: LayaRouter(criteria={"billing_agent": 1}, agent=high), ValidationError)
check_raises("validate/instructions-none",
             lambda: LayaRouter(criteria=CRITERIA, instructions=None, agent=high), ValidationError)

# --------------------------------------------------------------- 3. And the coerced value decides
check("gate/below-threshold", outcome(lambda: router.invoke({"input": "I was billed twice"})),
      "human")
check("gate/above-threshold",
      outcome(lambda: LayaRouter(criteria=CRITERIA, confidence_threshold="0.8", fallback="human",
                                 agent=high).invoke({"input": "I was billed twice"})),
      "billing_agent")
check("gate/numeric-threshold-still-gates",
      outcome(lambda: LayaRouter(criteria=CRITERIA, confidence_threshold=0.8, fallback="human",
                                 agent=low).invoke({"input": "I was billed twice"})),
      "human")

# --------------------------------------------------------------- 4. batch(): each input, either path
class BatchingAgent(MockAgent):
    """The shape `Agent`/`Router` have: a second entry point that answers many states at once."""

    def __init__(self, confidence):
        super().__init__(confidence)
        self.batch_calls = []

    def predict_batch(self, states, questions, **kwargs):
        self.batch_calls.append(len(states))
        return [self.predict(state, questions) for state in states]


INPUTS = ["I was billed twice", "the app crashes on launch"]
# The threshold is a string on purpose: whether `batch` gates at all depends on pydantic having
# coerced it, which is the part only a real install can check.
BATCH_KW = dict(criteria=CRITERIA, confidence_threshold="0.55", fallback="human")
check("batch/runner without predict_batch answers each input",
      outcome(lambda: LayaRouter(agent=low, **BATCH_KW).batch(INPUTS)), ["human", "human"])
shared = BatchingAgent(0.5)
check("batch/runner with predict_batch answers each input",
      outcome(lambda: LayaRouter(agent=shared, **BATCH_KW).batch(INPUTS)), ["human", "human"])
check("batch/one call per batch", outcome(lambda: shared.batch_calls), [2])

# --------------------------------------------------------------- Summary
print(f"PASS: {len(PASS)}")
print(f"FAIL: {len(FAIL)}")
for f in FAIL:
    print(f"FAILED: {f}")

if FAIL:
    sys.exit(1)
