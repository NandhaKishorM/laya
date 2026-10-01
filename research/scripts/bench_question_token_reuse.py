"""Compare the actual patch against the original functions at the checked-out base commit."""
import argparse
import ast
import contextlib
import importlib
import json
import os
import pathlib
import sys
import statistics
import subprocess
import time
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402
import laya  # noqa: E402
from bench_long_context import FILLER, QUESTIONS  # noqa: E402
import laya.common as common  # noqa: E402

agent_module = importlib.import_module("laya.agent")
after_batch = laya.Agent.predict_batch


def original_function(base, file, name, namespace, cls=None):
    # Execute only the original function from this checkout, with its original dependencies.
    source = subprocess.check_output(["git", "show", base + ":" + file], cwd=ROOT, text=True)
    tree = ast.parse(source)
    nodes = tree.body
    if cls:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == cls).body
    fn = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, file + "@" + base, "exec"), namespace)
    return namespace[name]


@contextlib.contextmanager
def mode(name, before_batch):
    with patch.object(laya.Agent, "predict_batch", before_batch if name == "before" else after_batch):
        yield


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (time.perf_counter() - start) * 1000


def capture(agent, fn):
    batches, raw = [], []
    forward = agent._forward

    def saved(batch):
        batches.append({key: val.clone() for key, val in batch.items() if isinstance(val, torch.Tensor)})
        result = forward(batch)
        raw.append(tuple(val.copy() for val in result))
        return result

    with patch.object(agent, "_forward", saved):
        result = fn()
    return result, batches, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint directory or Hugging Face repo")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--baseline-ref", required=True, help="Trusted Git ref before question reuse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    before_build = original_function(args.baseline_ref, "laya/common.py", "build_sequence", dict(common.__dict__))
    before_batch = original_function(args.baseline_ref, "laya/agent.py", "predict_batch",
                                    dict(agent_module.__dict__, build_sequence=before_build), cls="Agent")
    agent = laya.Agent(args.model, revision=args.revision, device=args.device, fast=False, compile=False)
    # First seed-13 MASSIVE English regression case, combined with the long-context filler.
    # This is a latency workload; no dataset accuracy is claimed for the padded input.
    case = {"state": {"utterance": "wake me up at five am this week"},
            "instructions": "What is the user asking for in `utterance`?",
            "options": ["email_query", "takeaway_order", "qa_definition", "cooking_recipe", "qa_factoid",
                        "calendar_set", "music_settings", "social_query", "general_quirky", "datetime_query",
                        "qa_currency", "audio_volume_other", "transport_query", "alarm_remove", "general_greet",
                        "transport_traffic", "email_querycontact", "alarm_set", "transport_ticket", "takeaway_query"]}
    questions20 = {"intent": {"type": "choice", "instructions": case["instructions"],
                             "criteria": {key: key.replace("_", " ") for key in case["options"]}}}
    repeats = round(4800 / len(agent.tok(FILLER, add_special_tokens=False)["input_ids"]))
    state = FILLER * repeats + "\n\nActual request: " + common.serialize_state(case["state"])
    workloads = [("long_20_options_default", lambda: agent.predict_long(state, questions20)),
                 ("long_20_options_batch8", lambda: agent.predict_long(state, questions20, batch_size=8)),
                 ("long_4_options_default", lambda: agent.predict_long(state, QUESTIONS)),
                 ("single_short_20_options", lambda: agent.system_one(case["state"], questions20))]
    report = {"base": args.baseline_ref, "model": args.model, "model_revision": args.revision,
              "torch": torch.__version__, "device": str(agent.device), "gpu": torch.cuda.get_device_name(agent.device) if agent.device.type == "cuda" else None, "threads": 4,
              "repeats": args.repeats, "workloads": []}
    for name, fn in workloads:
        for which in ("before", "after"):
            with mode(which, before_batch):
                for _ in range(3):
                    fn()
        with mode("before", before_batch):
            expected, before_tensors, before_raw = capture(agent, fn)
        with mode("after", before_batch):
            actual, after_tensors, after_raw = capture(agent, fn)
        assert expected == actual
        assert len(before_tensors) == len(after_tensors)
        assert all(left[key].equal(right[key]) for left, right in zip(before_tensors, after_tensors) for key in left)
        assert all((x == y).all() for left, right in zip(before_raw, after_raw) for x, y in zip(left, right))
        counts = {}
        for which in ("before", "after"):
            seen = []
            encode = common.encode_text

            def counting(tok, text, **kwargs):
                seen.append(text)
                return encode(tok, text, **kwargs)

            # The original build_sequence has its own global namespace.
            with mode(which, before_batch), patch.object(common, "encode_text", counting), patch.dict(
                    before_build.__globals__, encode_text=counting):
                fn()
            counts[which] = len(seen)
        pairs = []
        for i in range(args.repeats):
            pair = {}
            for which in (("before", "after") if i % 2 == 0 else ("after", "before")):
                with mode(which, before_batch):
                    result, pair[which] = timed(fn, agent.device)
                assert result == expected
                assert agent.device.type == args.device.split(":")[0]
            pairs.append(pair)
        before = statistics.median(pair["before"] for pair in pairs)
        after = statistics.median(pair["after"] for pair in pairs)
        row = {"name": name, "windows": expected["usage"].get("windows", 1),
               "before_p50_ms": before, "after_p50_ms": after, "saving_percent": (before - after) / before * 100,
               "paired_saving_p50_ms": statistics.median(pair["before"] - pair["after"] for pair in pairs),
               "after_faster_pairs": sum(pair["after"] < pair["before"] for pair in pairs),
               "question_encode_calls": counts, "pairs_ms": pairs,
               "all_input_tensors_equal": True, "decision_logits_and_action_probabilities_equal": True, "outputs_equal": True}
        report["workloads"].append(row)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in row.items() if key != "pairs_ms"}), flush=True)


if __name__ == "__main__":
    main()
