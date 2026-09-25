"""`laya-evals run --onnx`: the eval harness can score an ONNX export, not just the Router.

`OnnxRunner` adapts a single-checkpoint `ONNXAgent` to the harness contract (predict /
predict_batch with an optional per-example model). Weight-free throughout: the agent is a fake,
and the CLI end-to-end tests monkeypatch `laya.onnx_agent.ONNXAgent` -- importing that module
needs only numpy, onnxruntime is loaded inside its constructor -- so the real wiring from
`main(["run", ..., "--onnx", ...])` down to the written report JSON is exercised without
onnxruntime or a checkpoint download.

Run: python tests/test_evals_onnx.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evals_cli  # noqa: E402
from laya.evals import EvalError  # noqa: E402
from laya.evals_cli import OnnxRunner  # noqa: E402

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


class _FakeAgent:
    """Duck-typed ONNXAgent: records calls, answers every choice question with `label`."""

    def __init__(self, model_id_or_path, onnx_path=None, **kw):
        self.model_id = model_id_or_path
        self.onnx_path = onnx_path
        self.calls = []

    def _answer(self, questions):
        return {"model": "fake-onnx",
                "answers": {qid: {"type": "choice", "choice": "billing", "answer_confidence": 0.9}
                            for qid in questions},
                "usage": {"input_tokens": 3, "output_tokens": 0}}

    def predict(self, state, questions, **kw):
        self.calls.append(("predict", state))
        return self._answer(questions)


class _BatchAgent(_FakeAgent):
    def predict_batch(self, states, questions, batch_size=None):
        self.calls.append(("predict_batch", tuple(states), batch_size))
        return [self._answer(questions) for _ in states]


# ---------------------------------------------------------------- predict contract
agent = _FakeAgent("english-ckpt")
runner = OnnxRunner(agent)
Q = {"intent": {"type": "choice", "instructions": "i", "criteria": {"billing": "b"}}}
res = runner.predict("a state", Q)
check("predict/returns the agent answer", res["answers"]["intent"]["choice"], "billing")
check("predict/forwards state to the agent", agent.calls[-1], ("predict", "a state"))
runner.predict("s2", Q, model="english-ckpt")
check_true("predict/matching per-example model is accepted", True)

try:
    runner.predict("s3", Q, model="multilingual")
    check_true("predict/foreign model raises EvalError", False, "no exception")
except EvalError as exc:
    msg = str(exc)
    check_true("predict/foreign model raises EvalError", True)
    check_true("predict/error names the served checkpoint", "english-ckpt" in msg, msg)
    check_true("predict/error names the asked-for checkpoint", "multilingual" in msg, msg)

# ---------------------------------------------------------------- predict_batch contract
agent_nobatch = _FakeAgent("ckpt")
OnnxRunner(agent_nobatch).predict_batch(["a", "b", "c"], Q)
check("predict_batch/falls back to one predict per state",
      [c[1] for c in agent_nobatch.calls], ["a", "b", "c"])

bagent = _BatchAgent("ckpt")
out = OnnxRunner(bagent).predict_batch(["a", "b"], Q, batch_size=2)
check("predict_batch/delegates once when the agent can batch",
      bagent.calls, [("predict_batch", ("a", "b"), 2)])
check("predict_batch/delegated result passes through", len(out), 2)
try:
    OnnxRunner(bagent).predict_batch(["a"], Q, model="other")
    check_true("predict_batch/foreign model raises too", False, "no exception")
except EvalError:
    check_true("predict_batch/foreign model raises too", True)

# ---------------------------------------------------------------- CLI wiring
args = evals_cli._build_parser().parse_args(["run", "d.jsonl", "--onnx", "m.onnx"])
check("cli/--onnx parses on the run subcommand", args.onnx, "m.onnx")
args = evals_cli._build_parser().parse_args(["run", "d.jsonl"])
check("cli/--onnx defaults to None (torch Router path unchanged)", args.onnx, None)

tmp = tempfile.mkdtemp(prefix="laya_evals_onnx_")
dataset = os.path.join(tmp, "data.jsonl")
with open(dataset, "w") as f:
    f.write(json.dumps({"state": "charged twice, refund me",
                        "questions": Q, "expected": {"intent": "billing"}}) + "\n")
    f.write(json.dumps({"state": "billed again after cancelling",
                        "questions": Q, "expected": {"intent": "billing"}}) + "\n")
report_path = os.path.join(tmp, "report.json")

import laya.onnx_agent as onnx_agent_module  # noqa: E402  (numpy-only at import time)

_real_onnx_agent = onnx_agent_module.ONNXAgent
try:
    onnx_agent_module.ONNXAgent = _FakeAgent
    rc = evals_cli.main(["run", dataset, "--onnx", os.path.join(tmp, "laya.onnx"),
                         "--model", "english-ckpt", "--json", report_path])
    check("cli/run --onnx exits 0 on a passing dataset", rc, 0)
    with open(report_path) as f:
        report = json.load(f)
    check("cli/report records the onnx export path",
          report["config"]["onnx"], os.path.join(tmp, "laya.onnx"))
    check("cli/report keeps the forced checkpoint", report["config"]["model"], "english-ckpt")
    check("cli/metrics are computed from the ONNX answers",
          report["overall"]["choice_accuracy"], 1.0)

    # --model omitted: the export's checkpoint defaults to the english bundle repo.
    rc = evals_cli.main(["run", dataset, "--onnx", "laya.onnx"])
    check("cli/run without --model still scores", rc, 0)

    # batching reaches the runner's predict_batch through the harness
    onnx_agent_module.ONNXAgent = _BatchAgent
    rc = evals_cli.main(["run", dataset, "--onnx", "laya.onnx", "--model", "ckpt",
                         "--batch-size", "2"])
    check("cli/--batch-size runs the ONNX batch path", rc, 0)

    # a per-example model the single-checkpoint agent does not serve is an error, not a no-op
    # (no --model here: --model would authoritatively rewrite every row and mask the mismatch)
    mismatch = os.path.join(tmp, "mismatch.jsonl")
    with open(mismatch, "w") as f:
        row = json.loads(open(dataset).readline())
        row["model"] = "some-other-checkpoint"
        f.write(json.dumps(row) + "\n")
    rc = evals_cli.main(["run", mismatch, "--onnx", "laya.onnx"])
    check("cli/foreign per-example model exits 1", rc, 1)
finally:
    onnx_agent_module.ONNXAgent = _real_onnx_agent

# ---------------------------------------------------------------- torch path unchanged
check("api/RouterRunner still exists beside OnnxRunner", hasattr(evals_cli, "RouterRunner"), True)

# ---------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
