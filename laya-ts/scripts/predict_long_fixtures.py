#!/usr/bin/env python3
"""Record the real Python ONNXAgent.predict_long contract without downloading models.

Generate with --python-root pointing at v0.4.18 (118d361a774c9a28cd02a78d6cf4d2af0614f422).
CI uses --check against the current Python tree: provenance may change, behavior may not.
Only tokenization and the ONNX session are synthetic; sequence building, decoding,
hook dispatch, OptionStabilityHook, budget checks and aggregation are production code.
This is a deterministic control-flow oracle, not a model-quality evaluation.
Requires numpy and CPU torch (imported by laya.common); no ONNX Runtime is needed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "laya-ts/fixtures/predict-long.json"
REFERENCE_COMMIT = "118d361a774c9a28cd02a78d6cf4d2af0614f422"


class CharacterTokenizer:
    """One Unicode codepoint per token; expansion deliberately models decode drift."""

    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 1, 2, 3, 0
    mask_token = "[M]"

    def __init__(self, expansion=1):
        self.expansion = expansion

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = [ord(char) + 10 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(chr(int(token) - 10) * self.expansion for token in ids)


class RecordingSession:
    def __init__(self, tie=False):
        self.tie = tie
        self.calls = []

    def run(self, names, inputs):
        ids, attention = inputs["input_ids"], inputs["attention_mask"]
        sums = ids.sum(axis=1).tolist()
        self.calls.append({"rows": len(ids), "tokens": attention.sum(axis=1).tolist(), "sums": sums})
        width = inputs["marker_mask"].shape[1]
        logits = [[2 if self.tie else 1 + total % 5] + [-slot for slot in range(1, width)]
                  for total in sums]
        return [np.asarray(logits, dtype=np.float32),
                np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(ids), 1))]


def bare_agent(onnx_agent, case):
    agent = onnx_agent.__new__(onnx_agent)
    agent.model_id = "fixture"
    agent.cfg = deepcopy(case["cfg"])
    agent.tok = CharacterTokenizer(case.get("expansion", 1))
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    agent.session = RecordingSession(case.get("tie", False))
    agent.hooks = ()
    agent.hooks_raise = True
    agent.hooks_concurrent = True
    agent.hooks_timeout = None
    agent._hooks_lock = None
    agent._hooks_mutex = threading.Lock()
    return agent


def cases():
    questions = {
        "route": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y", "c": "z"}},
        "urgent": {"type": "noul", "instructions": "?"},
        "grade": {"type": "score", "instructions": "?", "criteria": [0, {"level": "high"}]},
    }
    state = "ABCDEGHIKLMNOPQRSTUVXYZabcdeghikmnopqrstuvxyz0123456789"

    def make(name, **changes):
        return {"name": name, "state": state, "questions": deepcopy(questions),
                "cfg": {"max_len": 96, "head_max_len": 40},
                "options": {"window": 16, "stride": 8, "batchSize": 2}, **changes}

    out = [make("all-types"), make("short", state="hi"),
           make("empty-state", state=""), make("structured-state", state={"text": "hello", "ok": True}),
           make("conversation", state=[{"role": "user", "content": state}]),
           make("mask-removal", state=state + "[M]" + state),
           make("default-window-and-cap", state=state * 3, options={}),
           make("explicit-window-capped", options={"window": 160, "stride": 100, "batchSize": 2}),
           make("tie-first-window", tie=True),
           make("decode-expansion-truncates", state=state * 2, expansion=2,
                cfg={"max_len": 64, "head_max_len": 32}, options={"window": 64}),
           make("empty-questions", questions={}),
           make("unsupported-aggregate", options={"aggregate": "mean"})]
    for behavior in ("reverse-results", "installed-end-reverse", "clone-results", "annotate", "shared-answer",
                     "reverse-states", "clone-states", "rewrite-states", "empty-states",
                     "remove-caller-all", "remove-caller-some", "temporary-remove-all",
                     "temporary-remove-some", "rewrite-question-type", "clear-questions",
                     "skip", "skip-many", "usage-fields", "shorten-results", "replace-batch",
                     "replace-batch-added-removed"):
        out.append(make(behavior, behavior=behavior))
    out += [make("short-skip", state="hi", behavior="skip"),
            make("stability-long", behavior="stability"),
            make("stability-short", state="hi", behavior="stability"),
            make("budget-after-forward", state=state * 2, behavior="budget",
                 options={"window": 48, "hooksRaise": False}),
            make("short-budget-after-forward", state=state[:35], behavior="budget", options={}),
            make("in-place-question-budget", state=state * 2, behavior="widen",
                 options={"hooksRaise": False}, cfg={"max_len": 64, "head_max_len": 32},
                 questions={"route": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}})]
    for field, qid in (("type", "route"), ("choice", "route"), ("noul", "urgent"),
                       ("score", "grade"), ("legend", "grade"), ("probabilities", "route"),
                       ("confidence", "route"), ("answer_confidence", "route")):
        out.append(make("edit-" + field, behavior="edit", field=field, qid=qid))
    return out


def hooks_for(case, stability_hook, events, cached):
    behavior = case.get("behavior")

    def start(ctx):
        if behavior == "reverse-states":
            ctx.states.reverse()
        elif behavior == "clone-states":
            ctx.states = deepcopy(ctx.states)
        elif behavior == "rewrite-states":
            ctx.states = ["rewritten"]
        elif behavior == "empty-states":
            ctx.states = []
        elif behavior in ("temporary-remove-all", "temporary-remove-some"):
            ctx.questions["temporary"] = {"type": "noul", "instructions": "?"}
        elif behavior == "rewrite-question-type":
            ctx.questions["route"] = {"type": "noul", "instructions": "?"}
        elif behavior == "clear-questions":
            ctx.questions = {}
        elif behavior in ("skip", "skip-many"):
            ctx.skip([cached] * (len(ctx.states) if behavior == "skip-many" else 1))
        elif behavior == "budget":
            ctx.max_len = 48
        elif behavior == "widen":
            ctx.questions["route"]["criteria"].update({"opt%02d" % i: "d" * 20 for i in range(8)})
        events.append({"phase": "start", "states": deepcopy(ctx.states), "questions": list(ctx.questions)})

    def end(ctx):
        events.append({"phase": "end", "results": len(ctx.results) if ctx.results is not None else None,
                       "error": ctx.error is not None})
        if ctx.results is None:
            return
        if behavior == "reverse-results":
            ctx.results.reverse()
        elif behavior == "clone-results":
            ctx.results = deepcopy(ctx.results)
        elif behavior == "shorten-results":
            ctx.results.pop()
        elif behavior in ("remove-caller-all", "remove-caller-some", "temporary-remove-all", "temporary-remove-some"):
            qid = "temporary" if behavior.startswith("temporary") else "route"
            for row in ctx.results if behavior.endswith("all") else ctx.results[:1]:
                del row["answers"][qid]
        elif behavior == "annotate":
            for row in ctx.results:
                for answer in row["answers"].values():
                    answer["reliability"] = {"note": "observer"}
                    answer["action"]["note"] = "observer"
        elif behavior == "usage-fields":
            for index, row in enumerate(ctx.results):
                row["usage"].update({"extra": index + 1, "flag": True, "details": {str(index): index},
                                     "label": str(index), "output_tokens": 9})
        elif behavior == "edit":
            field, qid = case["field"], case["qid"]
            for row in ctx.results:
                answer = row["answers"][qid]
                if field == "type":
                    # A valid end-hook rewrite: all score fields are present. The
                    # captured question type still chooses the aggregation metric.
                    answer.update(type="score", score=0.5, legend={"0": "low", "1": "high"})
                elif field == "choice":
                    answer[field] = "edited"
                elif field in ("legend", "probabilities"):
                    answer[field][next(iter(answer[field]))] = "edited" if field == "legend" else 0.1234
                else:
                    answer[field] = 0.1234

    hooks = []
    if behavior == "stability":
        hooks.append(stability_hook(accept=None, escalate=None, probes="reorder",
                                    variants=["reversed"], seed="long-fixture"))
    return hooks, start, end


def record(onnx_agent, stability_hook, case):
    agent = bare_agent(onnx_agent, case)
    behavior = case.get("behavior")
    if behavior == "installed-end-reverse":
        agent.hooks = [SimpleNamespace(on_predict_end=lambda ctx: ctx.results.reverse())]
    if behavior == "shared-answer":
        original = agent._infer_batch

        def shared(*args, **kwargs):
            rows = original(*args, **kwargs)
            for row in rows:
                row["answers"]["route"] = rows[0]["answers"]["route"]
            return rows

        agent._infer_batch = shared
    if behavior in ("replace-batch", "replace-batch-added-removed"):
        def replacement(states, questions, batch_size=None, lang=None, **kwargs):
            if behavior == "replace-batch-added-removed":
                questions["temporary"] = {"type": "noul", "instructions": "?"}
            rows = agent._infer_batch(states, questions, batch_size=batch_size, lang=lang)
            if behavior == "replace-batch-added-removed":
                for row in rows:
                    del row["answers"]["temporary"]
            return rows

        agent.predict_batch = replacement
    events = []
    cached = {"model": "cache", "answers": {"cached": {"type": "noul", "noul": 0.8,
              "confidence": 0.8, "answer_confidence": 0.8, "action": {"act_probability": 0.5}}},
              "usage": {"input_tokens": 0, "output_tokens": 0}}
    initial_cache = deepcopy(cached)
    hooks, start, end = hooks_for(case, stability_hook, events, cached)
    options = {"batch_size" if key == "batchSize" else "hooks_raise" if key == "hooksRaise" else key: value
               for key, value in case["options"].items()}
    outcome = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            # Deliberately invoke the real public method, never a port or a copied helper.
            outcome["result"] = agent.predict_long(deepcopy(case["state"]), deepcopy(case["questions"]),
                                                    hooks=hooks, on_predict_start=start,
                                                    on_predict_end=end, **options)
        except (ValueError, TypeError) as exc:
            outcome["error"] = str(exc)
    outcome["warnings"] = [str(w.message) for w in captured]
    outcome["forwards"] = agent.session.calls
    outcome["events"] = events
    outcome["cache_unchanged"] = cached == initial_cache
    return outcome


def provenance(root):
    files = ["laya/onnx_agent.py", "laya/agent.py", "laya/common.py", "laya/hooks.py", "laya/confidence.py"]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", *files], text=True)
    return {"pinned_tag": "v0.4.18", "pinned_commit": REFERENCE_COMMIT, "commit": commit,
            "modified_reference_files": dirty.splitlines(), "python": platform.python_version(),
            "numpy": np.__version__, "method": "laya.onnx_agent.ONNXAgent.predict_long",
            "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}


def generate(root):
    sys.path.insert(0, str(root))
    from laya.hooks import OptionStabilityHook
    from laya.onnx_agent import ONNXAgent

    if Path(sys.modules[ONNXAgent.__module__].__file__).resolve() != root / "laya/onnx_agent.py":
        raise RuntimeError("Python imported a different laya tree than --python-root")
    return {"schema": "laya-predict-long/1", "reference": provenance(root),
            "cases": [{**case, "expected": record(ONNXAgent, OptionStabilityHook, case)} for case in cases()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, default=ROOT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        reference = provenance(args.python_root.resolve())
        if reference["commit"] != REFERENCE_COMMIT or reference["modified_reference_files"]:
            raise SystemExit("Recording requires a clean v0.4.18 reference at " + REFERENCE_COMMIT
                             + "; use --python-root for that checkout, or --check to compare current behavior")
    result = generate(args.python_root.resolve())
    if args.check:
        expected = json.loads(args.fixture.read_text(encoding="utf-8"))
        actual = {key: value for key, value in result.items() if key != "reference"}
        wanted = {key: value for key, value in expected.items() if key != "reference"}
        if json.loads(json.dumps(actual)) != wanted:
            changed = [case["name"] for case, old in zip(result["cases"], expected["cases"]) if case != old]
            raise SystemExit("predict_long fixtures differ from this Python reference: " + ", ".join(changed))
        print("PASS: Python ONNXAgent.predict_long matches all committed fixtures")
    else:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2)[:-2]
        encoded = ",\n".join("    " + json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                              for case in result["cases"])
        args.fixture.write_text(header + ',\n  "cases": [\n' + encoded + "\n  ]\n}\n", encoding="utf-8")
        print(f"Wrote {args.fixture}: {len(result['cases'])} cases")
    print(json.dumps(result["reference"], sort_keys=True))


if __name__ == "__main__":
    main()
