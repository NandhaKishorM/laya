"""ONNXAgent must accept `lang` / `lang_temperatures` like the PyTorch Agent.

A caller who configures per-language temperatures and passes `lang=` to `Agent.system_one`
gets calibrated confidence. Swapping the same call to `ONNXAgent` used to raise
`TypeError: unexpected keyword argument 'lang'`, and an ONNXAgent built with `lang_temperatures`
silently ignored them. This checks the parity without onnxruntime or a checkpoint: onnxruntime
is imported lazily inside `__init__`, so the class imports, and `_infer` is driven with a stub
session and a fake tokenizer (the same technique as tests/test_portability.py).

Run: python tests/test_onnx_lang_parity.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import clamp_temperature  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402

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


# ---------------------------------------------------------------- signature parity with the torch Agent
onnx_so = inspect.signature(ONNXAgent.system_one).parameters
check_true("signature/system_one accepts lang", "lang" in onnx_so,
           "params were %s" % list(onnx_so))
check_true("signature/predict is system_one (accepts lang too)",
           "lang" in inspect.signature(ONNXAgent.predict).parameters)
check_true("signature/__init__ accepts lang_temperatures",
           "lang_temperatures" in inspect.signature(ONNXAgent.__init__).parameters)
# the torch Agent is the contract both sides must satisfy
check_true("signature/matches the torch Agent",
           "lang" in inspect.signature(Agent.system_one).parameters)


# ---------------------------------------------------------------- a fake tokenizer + stub session drive _infer
class _FakeTok:
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubSession:
    """Returns fixed logits so decoding is deterministic; one row per question, K columns."""

    def __init__(self, logits, act_logits):
        self._logits, self._act = logits, act_logits

    def run(self, names, inputs):
        return [self._logits, self._act]


def _bare_onnx(lang_temperatures):
    a = ONNXAgent.__new__(ONNXAgent)
    a.model_id = "stub"
    a.cfg = {"max_len": 64, "head_max_len": 32}
    a.tok = _FakeTok()
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    # two questions -> two rows; three option columns cover the widest question (the choice)
    a.session = _StubSession(
        np.array([[2.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),   # logits[row, :k]
        np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float32),             # act_logits
    )
    a.lang_temperatures = {}
    for l, cfg in (lang_temperatures or {}).items():
        norm = l.split("-")[0].lower()
        a.lang_temperatures[norm] = {
            "temperature": [clamp_temperature(t) for t in cfg["temperature"]],
            "temperature_by_options": {k: clamp_temperature(v)
                                       for k, v in cfg.get("temperature_by_options", {}).items()},
        }
    return a


QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "support": "help", "sales": "buy"}},
    "flag": {"type": "noul", "instructions": "Urgent?"},
}

# A German override with a clearly different temperature than the base [1,1,1].
agent = _bare_onnx({"de": {"temperature": [3.0, 3.0, 3.0]}})

base = agent._infer("some state text", QUESTIONS)
de = agent._infer("some state text", QUESTIONS, lang="de")
missing = agent._infer("some state text", QUESTIONS, lang="fr")   # not configured -> base

p_base = base["answers"]["dept"]["probabilities"]
p_de = de["answers"]["dept"]["probabilities"]
p_missing = missing["answers"]["dept"]["probabilities"]

check_true("lang/override changes the calibrated probabilities",
           p_base["billing"] != p_de["billing"],
           "base=%r de=%r" % (p_base, p_de))
# temperature 3 > 1 softens the distribution: the argmax probability falls
check_true("lang/higher temperature softens the top probability",
           p_de["billing"] < p_base["billing"])
check("lang/unconfigured language falls back to base", p_missing, p_base)
# a hyphen subtag resolves to the base language
de_DE = agent._infer("some state text", QUESTIONS, lang="de-DE")
check("lang/hyphen subtag resolves to the override",
      de_DE["answers"]["dept"]["probabilities"], p_de)
# the noul question is scaled the same way
check_true("lang/noul is scaled too",
           base["answers"]["flag"]["noul"] != de["answers"]["flag"]["noul"])

# a malformed override is rejected up front, like the torch Agent
try:
    _bare_onnx_bad = ONNXAgent.__new__(ONNXAgent)
    # the parsing lives in __init__; assert the same guard exists in its source
    src = inspect.getsource(ONNXAgent.__init__)
    check_true("lang/__init__ validates the 3-float override",
               "must be a list of 3 floats" in src)
except Exception as e:  # noqa: BLE001
    FAIL.append("lang/guard check raised %r" % e)


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
