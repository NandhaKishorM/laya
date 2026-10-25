"""Generate model-free TypeScript fixtures from production Python stability hooks.

    python laya-ts/scripts/option_stability_fixtures.py --python-root REFERENCE
    python laya-ts/scripts/option_stability_fixtures.py --python-root REFERENCE --check

REFERENCE must contain laya/hooks.py and laya/common.py. The committed oracle records
its original source hashes and Git provenance from a local combination of the #1286
rename semantics and the raw score/report/calibration contract from #1289.
NumPy is required by the production hooks. The production render functions are
loaded unchanged from their AST so generating fixtures does not import torch or
a model.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "laya-ts/fixtures/option-stability.json"
OPTION_NAMES = {
    "nShuffles": "n_shuffles", "renameMaxOptions": "rename_max_options",
    "targetAccuracy": "target_accuracy", "maxEscalateAccuracy": "max_escalate_accuracy",
}


def python_options(options):
    return {OPTION_NAMES.get(key, key): value for key, value in options.items()}


def load_reference(root):
    path = root / "laya/hooks.py"
    spec = importlib.util.spec_from_file_location("stability_fixture_reference", path)
    hooks = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hooks
    spec.loader.exec_module(hooks)
    render_names = {"render_criterion", "_resolve_noul_labels", "render_options"}
    tree = ast.parse((root / "laya/common.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in render_names]
    if {node.name for node in functions} != render_names:
        raise RuntimeError("reference common.py is missing the production render helpers")
    scope = {"json": json, "Dict": dict, "List": list,
             "_DEFAULT_NOUL_LABELS": {"false": "false", "true": "true"}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / "laya/common.py"), "exec"), scope)
    return hooks, scope["render_options"]


def provenance(root):
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("laya/hooks.py", "laya/common.py")}
    manifest = root / "reference-provenance.json"
    if manifest.exists():
        source = json.loads(manifest.read_text(encoding="utf-8"))
        if source.get("files") != hashes:
            raise RuntimeError("reference files differ from their provenance manifest")
    else:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        source = {"commit": git("rev-parse", "HEAD"),
                  "modified_reference_files": git("status", "--porcelain", "--", "laya/hooks.py", "laya/common.py"),
                  "files": hashes}
    return {**source, "python": platform.python_version()}


def variants(hooks):
    cases = [
        ("two-dedup", ["yes", "no"], "", {}),
        ("three-utf8", ["alpha", "beta", "gamma"], "种子|café|🙂", {}),
        ("twenty-zero-shuffles", [f"v{i}" for i in range(20)], "twenty", {"nShuffles": 0}),
        # Enough draws to cross MT19937's 624-word state boundary, not only its first block.
        ("twenty-six-twenty-shuffles", [f"v{i}" for i in range(26)], "twenty-six", {"nShuffles": 20}),
        ("twenty-seven-no-letters", [f"v{i}" for i in range(27)], "twenty-seven", {}),
        ("fast-include", ["a", "b", "c"], "seed", {"include": ["reversed", "letters_reversed"]}),
        ("custom-include", ["a", "b", "c", "d"], "seed", {"include": ["shuffle3", "letters_shuffle"]}),
        ("rename-only", ["a", "b", "c"], "seed", {"reorder": False}),
        ("reorder-only", ["a", "b", "c"], "seed", {"relabel": False}),
        ("no-probes", ["a", "b", "c"], "seed", {"relabel": False, "reorder": False}),
        ("filtered-away", ["a", "b", "c"], "seed", {"nShuffles": 0, "include": ["shuffle3"]}),
        ("unicode-labels", ["\ue000", "🙂", "é"], "nul\u0000seed", {"nShuffles": 5}),
    ]
    return [{"name": name, "labels": labels, "seed_key": seed, "options": options,
             "expected": [{"name": n, "order": order, "relabel": rel}
                          for n, order, rel in hooks.make_variants(labels, seed, **python_options(options))]}
            for name, labels, seed, options in cases]


def hook_case(hooks, render_options, name, criteria, *, options=None, layout="sequential",
              order=None, states=None, supports=None):
    options = {"accept": 0.8, "escalate": 0.6, **(options or {})}
    q = {"type": "choice", "instructions": "Choose a suitable action.", "criteria": criteria}
    if order is not None:
        q["option_order"] = order
    questions = {"q": q}
    original = deepcopy(questions)
    states = ["fixture state"] if states is None else states
    hook = hooks.OptionStabilityHook(**python_options(options))
    ctx = hooks.PredictContext(states=deepcopy(states), questions=questions,
                               agent=SimpleNamespace(config={"option_layout": layout}))
    hook.on_predict_start(ctx)
    assert questions == original, "the reference mutated caller-owned questions"

    expanded = []
    for key, question in ctx.questions.items():
        crit = question["criteria"]
        canonical = dict.fromkeys(crit) if isinstance(crit, list) else crit
        rendered = render_options({"t": "choice", "crit": canonical})
        physical = question.get("option_order", list(range(len(rendered))))
        expanded.append({"key": key, "rendered": [rendered[i] for i in physical]})

    # The reference's map is only used to synthesize model rows in each probe's
    # displayed label space. Expected public results still come from on_predict_end.
    plan = hook._plans[ctx.run_id]["q"]["variants"]
    labels = list(criteria)
    target = labels[0]
    others = labels[1:]
    model_results = []
    for state_index in range(len(states)):
        answers = {}
        for i, (_, key, back, _) in enumerate(plan):
            support = supports[i % len(supports)] if supports else (0.8, 0.25, 0.7)[(i + state_index) % 3]
            if not others:
                support = 1.0
            chosen = target if support >= 0.5 or not others else others[0]
            forward = {original: shown for shown, original in back.items()}
            probabilities = {forward[label]: (support if label == target else
                             1.0 - support if label == others[0] else 0.0) for label in labels}
            answers[key] = {"type": "choice", "choice": forward[chosen], "probabilities": probabilities}
        usage = {"input_tokens": 100 * len(answers), "output_tokens": len(answers),
                 "truncated_questions": list(answers),
                 "options": {key: {"total": len(labels), "distinct": len(labels)} for key in answers}}
        model_results.append({"answers": answers, "usage": usage})
    ctx.results = deepcopy(model_results)
    hook.on_predict_end(ctx)
    assert questions == original, "end hook mutated caller-owned questions"
    assert ctx.run_id not in hook._plans, "reference leaked its completed plan"
    for result in ctx.results:
        report = result["answers"]["q"]["reliability"]
        if report["soft_stability"] is not None:
            assert hooks.decide_reliability(report["soft_stability"], options["accept"], options["escalate"]) == report["decision"]
    return {"name": name, "states": states, "questions": original, "layout": layout,
            "options": options, "expanded": expanded,
            "model_results": model_results, "expected_results": ctx.results}


def hook_cases(hooks, render_options):
    make = lambda name, criteria, **kw: hook_case(hooks, render_options, name, criteria, **kw)
    described = {"accept": "Accept the request", "review": "Review the request", "reject": "Reject the request"}
    cases = [
        make("described-full", described),
        make("list-fast", list(described), options={"variants": "fast"}),
        make("undescribed", {"accept": None, "review": "", "reject": " \t\n"}, options={"variants": "fast"}),
        make("mixed-values", {"missing": None, "blank": " ", "zero": 0, "false": False,
                              "object": {"detail": "café", "priority": 2}, "array": ["x", 3]},
             options={"variants": "fast"}),
        make("numeric-keys-order", {"1": "first", "2": "second", "10": "tenth"}, order=[2, 0, 1]),
        make("numeric-list", ["1", "2", "10"], options={"variants": "fast"}),
        make("prototype-labels", {"__proto__": "base", "constructor": "create", "toString": "print"}),
        make("unicode-labels", {"\ue000": "private", "🙂": "smile", "é": "accent"}),
        make("custom-variants", described, options={"variants": ["shuffle3", "letters_shuffle"], "seed": "种子🙂"}),
        make("zero-shuffles", described, options={"nShuffles": 0}),
        make("extra-shuffles", described, options={"nShuffles": 5}),
        make("uncalibrated", described, options={"accept": None, "escalate": None, "variants": "fast"}),
        make("parallel-auto", described, layout="parallel"),
        make("parallel-explicit-reorder", described, layout="parallel", options={"probes": "reorder"}),
        make("parallel-relabel-off", described, layout="parallel", options={"relabel": False}),
        make("rename-cap", {f"v{i}": f"description {i}" for i in range(20)}, layout="parallel", options={"variants": "fast"}),
        make("over-rename-cap", {f"v{i}": None for i in range(21)}, layout="parallel"),
        make("alphabet-cap", {f"v{i}": None for i in range(27)}, layout="parallel", options={"renameMaxOptions": 27}),
        make("one-option", {"only": None}),
        make("batch-seed", described, states=["first", "second"], options={"variants": "fast"}),
        make("explicit-structured-state", described, states=[{"user": "你好", "enabled": True}], options={"seed": "shared seed"}),
        make("rounding-exact-halves", {"yes": "allow", "no": "deny"},
             options={"variants": "fast"}, supports=[0.9, 0.03125, 0.15625]),
        make("rounding-binary-neighbors", {"yes": "allow", "no": "deny"},
             options={"variants": "fast"}, supports=[0.9, 0.12345, 0.99995]),
    ]
    for name, support in [("below-accept", [0.7999, 0.8, 0.8]),
                          ("at-accept", [0.8, 0.8, 0.8]),
                          ("above-accept", [0.8001, 0.8, 0.8]),
                          ("below-escalate", [0.5999, 0.6, 0.6]),
                          ("at-escalate", [0.6, 0.6, 0.6])]:
        case = make(name, {"yes": "allow", "no": "deny"}, options={"variants": "fast"}, supports=support)
        report = case["expected_results"][0]["answers"]["q"]["reliability"]
        assert report["soft_stability"] == sum(support) / 3, "reference needs #1289 raw scores"
        cases.append(case)
    renamed = next(case for case in cases if case["name"] == "undescribed")["expanded"][-1]["rendered"]
    assert renamed == ["A: reject", "B: review", "C: accept"], "reference needs #1286 label semantics"
    return cases


def calibrations(hooks):
    cases = [
        ("near-equal-cuts", [0.79996, 0.8, 0.80004], [False, False, True], {"targetAccuracy": 1.0}),
        ("mixed-neighbors", [0.59996, 0.6, 0.60004, 0.79996, 0.8, 0.80004], [False, True, False, True, True, True], {}),
        ("equal-score-cohort", [0.8, 0.8, 0.9], [True, False, True], {"targetAccuracy": 1.0}),
        ("accept-nothing", [0.2, 0.8, 0.9], [False, False, False], {}),
        ("all-correct", [0.2, 0.8, 0.9], [True, True, True], {}),
        ("edges", [0.0, 1.0], [False, True], {"maxEscalateAccuracy": 0.0}),
    ]
    result = []
    for name, scores, correct, options in cases:
        thresholds = hooks.calibrate_reliability(scores, correct, **python_options(options))
        result.append({"name": name, "scores": scores, "correct": correct, "options": options,
                       "thresholds": thresholds,
                       "decisions": [hooks.decide_reliability(score, *thresholds) for score in scores]})
    assert result[0]["thresholds"][0] == 0.80004, "reference needs #1289 distinct raw cuts"
    return result


def generate(root):
    hooks, render_options = load_reference(root)
    cases = hook_cases(hooks, render_options)
    decisions = [{"score": score, "accept": accept, "escalate": escalate,
                  "expected": hooks.decide_reliability(score, accept, escalate)}
                 for accept, escalate in [(None, None), (0.8, 0.6), (1.01, 0.0), (0.0, 0.0)]
                 for score in [0.0, math.nextafter(0.6, 0.0), 0.6, math.nextafter(0.8, 0.0), 0.8,
                               math.nextafter(0.8, 1.0), 1.0]]
    selected = [case for case in cases if case["name"] in ("below-accept", "at-accept", "above-accept")]
    scores = [case["expected_results"][0]["answers"]["q"]["reliability"]["soft_stability"] for case in selected]
    correct = [False, False, True]
    thresholds = hooks.calibrate_reliability(scores, correct, target_accuracy=1.0)
    return {"schema": "laya-option-stability/1", "reference": provenance(root),
            "variant_cases": variants(hooks), "hook_cases": cases,
            "calibration_cases": calibrations(hooks), "decision_cases": decisions,
            "pipeline": {"cases": [case["name"] for case in selected], "correct": correct,
                         "options": {"targetAccuracy": 1.0}, "thresholds": thresholds,
                         "decisions": [hooks.decide_reliability(score, *thresholds) for score in scores]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, default=ROOT)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true", help="compare behavior with the committed oracle without writing")
    args = parser.parse_args()
    result = generate(args.python_root.resolve())
    if args.check:
        expected = json.loads(args.fixture.read_text(encoding="utf-8"))
        # A newer source commit may implement the same contract. Compare behavior;
        # print actual provenance, retaining the original oracle's provenance.
        actual = {key: value for key, value in result.items() if key != "reference"}
        wanted = {key: value for key, value in expected.items() if key != "reference"}
        if json.loads(json.dumps(actual)) != wanted:
            raise SystemExit("option stability fixtures differ from this Python reference")
        print("PASS: Python reference matches all committed option stability fixtures")
    else:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        # Keep one complete case per line: the oracle stays small while its named
        # cases and source provenance remain easy to locate and inspect.
        fields = []
        for key, value in result.items():
            if key.endswith("_cases"):
                encoded = "[\n" + ",\n".join("    " + json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                                               for case in value) + "\n  ]"
            else:
                encoded = json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\n  ")
            fields.append(f"  {json.dumps(key)}: {encoded}")
        args.fixture.write_text("{\n" + ",\n".join(fields) + "\n}\n", encoding="utf-8")
        print(f"Wrote {args.fixture} ({args.fixture.stat().st_size} bytes)")
    print(json.dumps(result["reference"], ensure_ascii=False, sort_keys=True))
    print(f"Cases: {len(result['variant_cases'])} variants, {len(result['hook_cases'])} hook calls, "
          f"{len(result['calibration_cases'])} calibrations, {len(result['decision_cases'])} decisions, 1 fit/replay pipeline")


if __name__ == "__main__":
    main()
