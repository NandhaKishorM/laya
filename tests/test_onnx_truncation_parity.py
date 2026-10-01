"""ONNXAgent must report state truncation in `usage` like the PyTorch Agent (#174).

`Agent.predict_batch` adds `state_tokens`, `state_tokens_dropped`, `truncated` and
`truncated_questions` to each result's usage, from the token budget `build_sequence` applied.
A caller who swaps to `ONNXAgent` must get the same fields with the same meaning. This drives
`_infer` with a stub session and a fake tokenizer (the technique in tests/test_onnx_lang_parity.py),
so it needs neither onnxruntime nor a checkpoint.

Run: python tests/test_onnx_truncation_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from laya.common import build_sequence  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


# ---------------------------------------------------------------- a fake tokenizer + stub session drive _infer
class _FakeTok:
    """One token per character, so a state's token count is its length."""
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [10 + (ord(c) % 40) for c in text]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


class _StubSession:
    def run(self, names, inputs):
        n = inputs["input_ids"].shape[0]
        return [np.zeros((n, 3), dtype=np.float32), np.zeros((n, 2), dtype=np.float32)]


MAX_LEN, HEAD_MAX_LEN = 128, 96


def _bare_onnx():
    a = ONNXAgent.__new__(ONNXAgent)
    a.model_id = "stub"
    a.cfg = {"max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}
    a.tok = _FakeTok()
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.lang_temperatures = {}
    a.session = _StubSession()
    return a


QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "money", "support": "help", "sales": "buy"}},
    "flag": {"type": "noul", "instructions": "Ok?"},
}

# The room each question leaves for the state: max_len minus the head and the closing [SEP].
ROOM = {qid: MAX_LEN - len(build_sequence(_FakeTok(), "", ONNXAgent._to_internal(q), MAX_LEN, HEAD_MAX_LEN)[0])
        for qid, q in QUESTIONS.items()}
assert ROOM["flag"] < ROOM["dept"], ROOM   # the cases below need two different budgets

agent = _bare_onnx()

# ---------------------------------------------------------------- empty questions: usage does not grow
check("empty/usage is unchanged", agent._infer("some state", {})["usage"],
      {"input_tokens": 0, "output_tokens": 0})

# ---------------------------------------------------------------- a state that fits every question
u = agent._infer("x" * ROOM["flag"], QUESTIONS)["usage"]
check("fits/keys match the torch Agent", sorted(u),
      sorted(["input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped", "truncated",
              "truncated_questions"]))
check("fits/state_tokens", u["state_tokens"], ROOM["flag"])
check("fits/nothing dropped", u["state_tokens_dropped"], 0)
check("fits/truncated is False (a bool)", (u["truncated"], type(u["truncated"])), (False, bool))
check("fits/no question listed", u["truncated_questions"], [])

# ---------------------------------------------------------------- a state only the narrower question truncates
n = ROOM["flag"] + 6
u = agent._infer("x" * n, QUESTIONS)["usage"]
check("one/state_tokens is the full state", u["state_tokens"], n)
check("one/dropped is the worst question's", u["state_tokens_dropped"], n - ROOM["flag"])
check("one/truncated", (u["truncated"], type(u["truncated"])), (True, bool))
check("one/only the narrower question listed", u["truncated_questions"], ["flag"])

# ---------------------------------------------------------------- a state every question truncates
u = agent._infer("x" * 500, QUESTIONS)["usage"]
check("all/state_tokens", u["state_tokens"], 500)
check("all/dropped is the max over questions", u["state_tokens_dropped"], 500 - ROOM["flag"])
check("all/every question listed, in order", u["truncated_questions"], ["dept", "flag"])

# ---------------------------------------------------------------- a conversation list keeps its tail, same report
convo = [{"role": "user", "content": "x" * 500}]
u = agent._infer(convo, QUESTIONS)["usage"]
check("list/truncated from the left is still reported", u["truncated"], True)

# ---------------------------------------------------------------- predict_batch reports each state on its own
res = agent.predict_batch(["x" * ROOM["flag"], "x" * (ROOM["flag"] + 6), "x" * 500], QUESTIONS)
check("batch/per-state truncated", [r["usage"]["truncated"] for r in res], [False, True, True])
check("batch/per-state questions", [r["usage"]["truncated_questions"] for r in res], [[], ["flag"], ["dept", "flag"]])
check("batch/per-state state_tokens", [r["usage"]["state_tokens"] for r in res], [ROOM["flag"], ROOM["flag"] + 6, 500])


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
