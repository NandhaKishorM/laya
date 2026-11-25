#!/usr/bin/env python3
"""Record internal/gateway/testdata/policy.json: the tier policy and table row of
examples/29_presets_model_router.py on synthetic answers, with the example's own formatting.

    python laya-go/samples/scripts/gen_router_policy_golden.py          # write
    python laya-go/samples/scripts/gen_router_policy_golden.py --check  # fail if the file differs
"""
import argparse
import ast
import json
import pathlib
import sys

EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / "examples" / "29_presets_model_router.py"
OUT = pathlib.Path(__file__).resolve().parents[1] / "internal" / "gateway" / "testdata" / "policy.json"


def _str(node):
    try:
        v = ast.literal_eval(node)
    except ValueError:
        return ""
    return v if isinstance(v, str) else ""


def _load_example():
    """Take route_tier and the table-row expression from the example's own source via ast."""
    src = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "route_tier"]
    assert len(fn) == 1, "route_tier not found in %s" % EXAMPLE
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(EXAMPLE), "exec"), ns)
    dom = row_expr = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "domain"):
            dom = ast.Expression(n.value)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print" and n.args
                and isinstance(n.args[0], ast.BinOp) and _str(n.args[0].left).startswith("   %-17s %.2f / 3")):
            row_expr = ast.Expression(n.args[0])
    assert dom is not None and row_expr is not None, "table row expression not found in %s" % EXAMPLE
    dom_c = compile(ast.fix_missing_locations(dom), str(EXAMPLE), "eval")
    row_c = compile(ast.fix_missing_locations(row_expr), str(EXAMPLE), "eval")

    def row(label, a):
        env = {"label": label, "a": a}
        env["domain"] = eval(dom_c, {}, env)
        return eval(row_c, {}, env)

    return ns["route_tier"], row


route_tier, row = _load_example()


def case(d, probs, tools, sens):
    return {"difficulty": {"score": d}, "domain": {"probabilities": probs},
            "needs_tools": {"noul": tools}, "is_sensitive": {"noul": sens}}


DOMAINS = {"code": 0.1, "math_or_logic": 0.2, "writing": 0.3, "factual_lookup": 0.3, "data_analysis": 0.05,
           "chitchat": 0.05}
CASES = [
    ("small", case(0.4, DOMAINS, 0.1, 0.1)),
    ("hard", case(2.2, DOMAINS, 0.2, 0.3)),
    ("at-difficulty-threshold", case(1.5, DOMAINS, 0.2, 0.3)),
    ("just-under-threshold", case(1.4999, DOMAINS, 0.2, 0.3)),
    ("tools", case(0.2, DOMAINS, 0.5, 0.49)),
    ("sensitive-overrides-tools", case(0.2, DOMAINS, 0.9, 0.5)),
    ("tools-just-under", case(0.2, DOMAINS, 0.4999, 0.1)),
    ("tools-overrides-hard", case(2.2, DOMAINS, 0.9, 0.1)),
    ("sensitive-overrides-hard", case(2.5, DOMAINS, 0.1, 0.9)),
    ("sensitive-easy", case(0.1, DOMAINS, 0.0, 0.97)),
]


def render():
    out = []
    for name, a in CASES:
        tier, why = route_tier(a)
        out.append({"name": name, "difficulty": a["difficulty"]["score"], "domains": a["domain"]["probabilities"],
                    "needs_tools": a["needs_tools"]["noul"], "is_sensitive": a["is_sensitive"]["noul"],
                    "tier": tier, "why": why, "row": row("label", a)})
    return json.dumps(out, ensure_ascii=False, indent=1) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("%s differs" % OUT)
            return 1
        print("%s matches" % OUT)
        return 0
    OUT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
