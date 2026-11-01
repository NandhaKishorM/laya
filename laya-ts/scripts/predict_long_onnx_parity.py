"""Matched split-graph ONNX parity; real Python ONNXAgent.predict_long, CPU only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--python-root", type=Path, required=True, help="Checkout of Python v0.4.18")
parser.add_argument("--checkpoint", type=Path, required=True, help="Original English checkpoint, including tokenizer/")
parser.add_argument("--bundle", type=Path, required=True, help="Split ONNX export of that exact checkpoint")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
REFERENCE = args.python_root.resolve()
sys.path.insert(0, str(REFERENCE))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import onnxruntime as ort
import transformers
from transformers import AutoTokenizer
import laya
from laya.onnx_agent import ONNXAgent
from laya.common import clamp_temperature

BUNDLE = args.bundle.resolve()
CHECKPOINT = args.checkpoint.resolve()
if laya.__version__ != "0.4.18":
    raise SystemExit("--python-root must point at the pinned Python 0.4.18 reference")
reference_commit = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
if reference_commit != "118d361a774c9a28cd02a78d6cf4d2af0614f422":
    raise SystemExit("--python-root must be checked out at v0.4.18 commit 118d361a774c9a28cd02a78d6cf4d2af0614f422")
if subprocess.check_output(["git", "-C", str(REFERENCE), "status", "--porcelain", "--", "laya"], text=True).strip():
    raise SystemExit("--python-root must have an unmodified laya/ source tree")
if (CHECKPOINT / "tokenizer/tokenizer.json").read_bytes() != (BUNDLE / "tokenizer.json").read_bytes():
    raise SystemExit("checkpoint and ONNX bundle must use the same tokenizer.json")
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1
encoder = ort.InferenceSession(str(BUNDLE / "encoder.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
head = ort.InferenceSession(str(BUNDLE / "head.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
calls = []
class SplitSession:
    def run(self, names, feeds):
        calls.append({key: value.tolist() for key, value in feeds.items()})
        hidden = encoder.run(None, {k: feeds[k] for k in ("input_ids", "attention_mask")})[0]
        hf = {k: feeds[k] for k in ("marker_pos", "marker_mask", "attention_mask")}
        hf.update(hidden_states=hidden, qtype=feeds["qtype"].reshape(-1, 1))
        return head.run(names, hf)

agent = ONNXAgent.__new__(ONNXAgent)
agent.cfg = json.loads((BUNDLE / "rl_agent_config.json").read_text())
agent.tok = AutoTokenizer.from_pretrained(str(CHECKPOINT / "tokenizer"), local_files_only=True)
agent.session = SplitSession()
agent.hooks = []
agent._hooks_mutex = threading.Lock()
agent.temperature_raw = agent.cfg["temperature"]
agent.temperature = [clamp_temperature(x) for x in agent.temperature_raw]
agent.temperature_by_options_raw = agent.cfg.get("temperature_by_options", {})
agent.temperature_by_options = {k: clamp_temperature(v) for k, v in agent.temperature_by_options_raw.items()}
agent.lang_temperatures = {}
agent.binning_map = None
questions = {
    "action": {"type": "choice", "instructions": "What should support do?", "criteria": ["refund", "investigate", "close"]},
    "urgent": {"type": "noul", "instructions": "Is urgent action required?"},
    "severity": {"type": "score", "instructions": "How severe is the problem?", "criteria": ["minor", "moderate", "severe"]},
}
long = "This customer has asked for an update on the account. " * 8 + "The account was charged twice. Please refund the duplicate payment immediately."
cases = [
    {"name": "short", "state": "Please refund the duplicate charge.", "questions": questions, "options": {}},
    {"name": "long-overlap-capped", "state": long, "questions": questions, "options": {"window": 64, "stride": 32, "batchSize": 2}},
    {"name": "structured", "state": {"ticket": long, "priority": "urgent"}, "questions": questions, "options": {"window": 96, "stride": 48, "batchSize": 2}},
]
for case in cases:
    calls.clear()
    opts = dict(case["options"])
    if "batchSize" in opts:
        opts["batch_size"] = opts.pop("batchSize")
    began = time.perf_counter()
    case["python"] = agent.predict_long(case["state"], case["questions"], **opts)
    case["python_inputs"] = list(calls)
    case["python_seconds"] = time.perf_counter() - began
    print(case["name"], case["python"]["usage"], flush=True)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
report = {
    "reference": {"tag": "v0.4.18", "commit": reference_commit,
                  "sources": {name: sha(REFERENCE / name) for name in ["laya/onnx_agent.py", "laya/agent.py", "laya/common.py", "laya/hooks.py"]}},
    "checkpoint": {"path": str(CHECKPOINT), "tokenizer_sha256": sha(CHECKPOINT / "tokenizer/tokenizer.json")},
    "bundle": str(BUNDLE),
    "sha256": {p.name: sha(p) for p in BUNDLE.iterdir() if p.is_file()},
    "environment": {"python": platform.python_version(), "onnxruntime": ort.__version__, "numpy": np.__version__, "transformers": transformers.__version__, "backend": "CPUExecutionProvider", "python_threads": 2},
    "method": "Real ONNXAgent.predict_long over an adapter joining the exact encoder/head graphs consumed by TS. No fake logits; no model-quality or latency benchmark.",
    "cases": cases,
}
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
