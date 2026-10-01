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
        self.calibration = kw.get("calibration")
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
    def predict_batch(self, states, questions, batch_size=None, sort_by_length=False):
        self.calls.append(("predict_batch", tuple(states), batch_size, sort_by_length))
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
      bagent.calls, [("predict_batch", ("a", "b"), 2, False)])
check("predict_batch/delegated result passes through", len(out), 2)

# The #294 grouping knob: core pads each pass to its longest state, so a batch of mixed lengths
# feeds the model mostly padding. `laya-evals run --sort-by-length` is how a scored run asks for
# the cheaper shape of the same pass.
grouped = _BatchAgent("ckpt")
OnnxRunner(grouped).predict_batch(["a", "b"], Q, batch_size=2, sort_by_length=True)
check("predict_batch/sort_by_length is forwarded to the agent that has it",
      grouped.calls, [("predict_batch", ("a", "b"), 2, True)])

untouched = _BatchAgent("ckpt")
OnnxRunner(untouched).predict_batch(["a", "b"], Q, batch_size=2)
check("predict_batch/an unsorted run does not invent the key",
      untouched.calls, [("predict_batch", ("a", "b"), 2, False)])

# An agent that cannot batch has no group to reorder, and asking must not break the run.
nofile = _FakeAgent("ckpt")
OnnxRunner(nofile).predict_batch(["a", "b", "c"], Q, batch_size=2, sort_by_length=True)
check("predict_batch/a no-batch agent still answers when grouping is asked",
      [c[1] for c in nofile.calls], ["a", "b", "c"])
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
args = evals_cli._build_parser().parse_args(["run", "d.jsonl", "--onnx", "m.onnx", "--calibration", "c.json"])
check("cli/--calibration parses on the run subcommand", args.calibration, "c.json")
args = evals_cli._build_parser().parse_args(["run", "d.jsonl"])
check("cli/--calibration defaults to None", args.calibration, None)

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
    check("cli/report omits calibration when unset", "calibration" not in report["config"], True)
    check("cli/report keeps the forced checkpoint", report["config"]["model"], "english-ckpt")
    check("cli/metrics are computed from the ONNX answers",
          report["overall"]["choice_accuracy"], 1.0)

    # --calibration without --onnx is rejected before model load
    rc_no_onnx = evals_cli.main(["run", dataset, "--calibration", "calib.json"])
    check("cli/--calibration without --onnx exits 2", rc_no_onnx, 2)

    # --calibration with --onnx is forwarded to ONNXAgent and recorded in report config
    calib_file = os.path.join(tmp, "calib.json")
    calib_report_path = os.path.join(tmp, "report_calib.json")
    rc = evals_cli.main(["run", dataset, "--onnx", os.path.join(tmp, "laya.onnx"),
                         "--model", "english-ckpt", "--calibration", calib_file,
                         "--json", calib_report_path])
    check("cli/run --onnx --calibration exits 0", rc, 0)
    with open(calib_report_path) as f:
        calib_report = json.load(f)
    check("cli/report records the calibration path",
          calib_report["config"]["calibration"], calib_file)

    # --model omitted: the export's checkpoint defaults to the english bundle repo.
    rc = evals_cli.main(["run", dataset, "--onnx", "laya.onnx"])
    check("cli/run without --model still scores", rc, 0)

    # batching reaches the runner's predict_batch through the harness
    onnx_agent_module.ONNXAgent = _BatchAgent
    rc = evals_cli.main(["run", dataset, "--onnx", "laya.onnx", "--model", "ckpt",
                         "--batch-size", "2"])
    check("cli/--batch-size runs the ONNX batch path", rc, 0)

    # ... and so does the grouping knob, all the way from the flag to the agent's own call. The
    # report is the witness: `sort_by_length_sent` is what the harness put on the wire, not what
    # the command line said, so a flag that parsed and was dropped fails here by name.
    grouped_path = os.path.join(tmp, "grouped.json")
    rc = evals_cli.main(["run", dataset, "--onnx", "laya.onnx", "--model", "ckpt",
                         "--batch-size", "2", "--sort-by-length", "--json", grouped_path])
    check("cli/--sort-by-length scores the same dataset", rc, 0)
    with open(grouped_path) as f:
        grouped_report = json.load(f)
    check("cli/--sort-by-length is sent to the runner",
          grouped_report["config"]["timing"]["sort_by_length_sent"], True)
    check("cli/an ungrouped run says it sent nothing",
          report["config"]["timing"]["sort_by_length_sent"], False)
    check("cli/grouping changes the shape of the run, not its score",
          grouped_report["overall"]["choice_accuracy"], report["overall"]["choice_accuracy"])

    # a per-example model the single-checkpoint agent does not serve is an error, not a no-op
    # (no --model here: --model would authoritatively rewrite every row and mask the mismatch)
    mismatch = os.path.join(tmp, "mismatch.jsonl")
    with open(mismatch, "w") as f:
        row = json.loads(open(dataset).readline())
        row["model"] = "some-other-checkpoint"
        f.write(json.dumps(row) + "\n")
    rc = evals_cli.main(["run", mismatch, "--onnx", "laya.onnx"])
    # Asking for a different checkpoint than the ONNX export serves is a caller mistake, so
    # exit 2 (docs/evals.md:28), not the 1 a quality failure uses.
    check("cli/foreign per-example model exits 2", rc, 2)
finally:
    onnx_agent_module.ONNXAgent = _real_onnx_agent

# ---------------------------------------------------------------- torch path unchanged
check("api/RouterRunner still exists beside OnnxRunner", hasattr(evals_cli, "RouterRunner"), True)

# ---------------------------------------------------------------- report
print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
