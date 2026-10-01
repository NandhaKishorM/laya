"""`docs/structured.md`'s tables must be the compiler's behavior, in both directions.

The page promises a "documented subset" (`laya/structured.py:146` says the mapping is one), and the
subset table is what a reader trusts. It had drifted both ways at once: it promised `title` as an
"option label", which no code path reads at all (weight-free proof: the compiled plan is
byte-identical with and without the key), while `Optional[X]` -- the `anyOf`/`oneOf` unwrap and the
`type: ["x", "null"]` list form -- had been supported for months and appeared nowhere. 13 of the
module's 19 `SchemaError` messages were also undocumented, the three limits collapsed into one row
that said only "the limit is named in the message".

So the tables are rewritten as data the test can run, and this suite runs them: every schema cell is
compiled, every rejection cell is raised, and the key set and message set are compared against what
`laya/structured.py` actually reads and actually raises -- derived with `ast`, never transcribed. A
row that no longer matches the code fails; a key or a message added to the code without a row fails;
and a row added to the docs that the code does not honour fails.

Needs no weights, no GPU, no network, and no pydantic (the `tests` job installs neither). The
pydantic-side claims -- `Literal[...]` renders to `enum`, `Optional[...]` to `anyOf`, and
`model_json_schema()` putting a `title` on every field -- are prose rationale here, checked locally
against pydantic in the PR's witness log.

Run: `python tests/test_structured_docs.py`
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

PASS: List[str] = []
FAIL: List[str] = []


def check(what: str, got: Any, want: Any) -> None:
    if got == want:
        PASS.append(what)
    else:
        # %s, not concatenation: several details below are lists of names, and a gate that raises on
        # the way to reporting a failure leaves the row it failed on unsaid.
        FAIL.append("%s: got %r, want %r" % (what, got, want))


def check_true(what: str, cond: bool, detail: Any = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def read(path: str) -> str:
    # newline="" so a CRLF checkout and an LF checkout parse identically
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


DOCS = os.path.join("docs", "structured.md")
MODULE = os.path.join("laya", "structured.py")

SUBSET_HEADER = "| JSON schema | Laya question | Returned value |"
PROP_HEADER = "| property schema | message |"
CALL_HEADER = "| call | message |"
API_HEADER = "| function | purpose |"


def split_row(line: str) -> List[str]:
    # a message cell may contain a `|`? none do; splitting on the plain separator is what the page's
    # own rendering requires, so a stray pipe in a cell is itself a finding.
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_table(text: str, header: str) -> List[List[str]]:
    lines = text.split("\n")
    try:
        start = lines.index(header)
    except ValueError:
        return []
    rows = []
    for line in lines[start + 2:]:                        # skip the |---| separator
        if not line.startswith("|"):
            break
        rows.append(split_row(line))
    return rows


def cell_value(cell: str) -> str:
    """The backticked body of a table cell."""
    m = re.match(r"^`(.*)`$", cell.strip())
    if not m:
        raise ValueError("cell is not a single backticked value: %r" % cell)
    return m.group(1)


SAFE: Dict[str, Any] = {"__builtins__": {"range": range, "round": round, "len": len}}


def run_cell(text: str, extra: Optional[Dict[str, Any]] = None) -> Any:
    """Evaluate a documented cell. The namespace is literals plus what the page names."""
    ns = dict(SAFE)
    ns.update(extra or {})
    return eval(text, ns)                                 # noqa: S307 - the text is this repo's own docs


# ------------------------------------------------------------------- what the compiler really does

def property_keys(tree: ast.AST) -> List[str]:
    """Every key `laya/structured.py` looks up on a *property* schema dict, by ast.

    Three spellings matter: `prop.get("k")`, `prop["k"]`, and `if "k" in prop`. Reading only one of
    them is how a copy of a registry goes stale elsewhere in this repo; here the risk is reading only
    one of the three spellings and reporting a key as unread.
    """
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "prop" and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id == "prop" and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, ast.Name) \
                        and comp.id == "prop":
                    keys.add(node.left.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd) \
                and isinstance(node.left, ast.Set) and isinstance(node.right, ast.Call) \
                and getattr(node.right.func, "id", None) == "set" \
                and isinstance(node.right.args[0], ast.Name) and node.right.args[0].id == "prop":
            for el in node.left.elts:
                if isinstance(el, ast.Constant):
                    keys.add(el.value)
    return sorted(keys)


def error_messages(tree: ast.AST) -> List[str]:
    """Every `SchemaError(...)` message literal in the module, with the format specifiers intact."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) \
                and (getattr(node.exc.func, "id", None) or getattr(node.exc.func, "attr", None)) == "SchemaError" \
                and node.exc.args:
            first = node.exc.args[0]
            while isinstance(first, ast.BinOp):           # "text %s" % (a,)
                first = first.left
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.append(first.value)
    return out


def fragments(message: str) -> List[str]:
    """The static parts of a documented message, long enough to identify a raise site.

    A code literal is `"%s: a free string cannot ..."`; the docs carry the raised text, so the two
    are compared on the parts no placeholder covers. Short tails (`""` after a trailing `%s`, a bare
    `": "`) match anything and are dropped.
    """
    parts = [p.strip() for p in re.split(r"%[-0-9.]*[sdrr]", message) if p.strip()]
    return [p for p in parts if len(p) > 12 or p in ("got",)]


def matches(literal: str, message: str) -> bool:
    frags = fragments(literal)
    return bool(frags) and all(f in message for f in frags)


def main() -> int:
    check_true("docs/structured.md exists", os.path.exists(DOCS), "missing")
    check_true("laya/structured.py exists", os.path.exists(MODULE), "missing")
    if FAIL:
        return report()

    text = read(DOCS)
    tree = ast.parse(read(MODULE))
    sys.path.insert(0, os.getcwd())
    import laya
    import laya.structured as S

    code_keys = property_keys(tree)
    literals = error_messages(tree)

    subset = parse_table(text, SUBSET_HEADER)
    props = parse_table(text, PROP_HEADER)
    calls = parse_table(text, CALL_HEADER)
    api = parse_table(text, API_HEADER)

    # ------------------------------------------------------------------ the tables parsed at all
    check_true("subset table/parsed rows", len(subset) == 9, "got %d" % len(subset))
    check_true("subset table/every row has three cells",
               all(len(r) == 3 for r in subset), [len(r) for r in subset])
    check_true("rejections table/parsed rows", len(props) == 13, "got %d" % len(props))
    check_true("rejections table/every row has two cells",
               all(len(r) == 2 for r in props), [len(r) for r in props])
    check_true("entry point table/parsed rows", len(calls) == 6, "got %d" % len(calls))
    check_true("api table/parsed rows", len(api) >= 6, "got %d" % len(api))

    # ----------------------------------------------------------- the subset table, row by row
    ns = {"plan_from_json_schema": S.plan_from_json_schema, "decide": S.decide}
    documented_keys = set()
    for row in subset:
        try:
            prop = run_cell(cell_value(row[0]))
        except Exception as exc:                          # a cell that will not even parse
            FAIL.append("subset row/%s: cell raises %s: %s" % (row[0], type(exc).__name__, exc))
            continue
        want_kind = cell_value(row[1])
        if not isinstance(prop, dict):
            FAIL.append("subset row/%s: a schema cell must be an object, got %s"
                        % (row[0], type(prop).__name__))
            continue
        documented_keys.update(prop.keys())
        try:
            field = S.plan_from_json_schema(
                {"type": "object", "properties": {"field": prop}})[0]
        except Exception as exc:
            FAIL.append("subset row/%s: the documented schema raises %s: %s"
                        % (row[0], type(exc).__name__, exc))
            continue
        check("subset row/%s question" % cell_value(row[0]), field.question["type"], want_kind)

    # the wording row has to prove the key does what the table says it does
    desc_row = [r for r in subset if "description" in r[0]]
    check_true("subset table/one row documents `description`", len(desc_row) == 1,
               "got %d" % len(desc_row))
    if len(desc_row) == 1:
        try:
            prop = run_cell(cell_value(desc_row[0][0]))
            wording = S.plan_from_json_schema(
                {"type": "object", "properties": {"f": prop}})[0].question["instructions"]
            without = {k: v for k, v in prop.items() if k != "description"}
            dropped = S.plan_from_json_schema(
                {"type": "object", "properties": {"f": without}})[0].question["instructions"]
            check("subset row/description becomes the instructions", wording, prop["description"])
            check_true("subset row/description is load-bearing", dropped != wording,
                       "the key changes nothing: %r" % (wording,))
        except Exception as exc:
            FAIL.append("subset row/description: %s: %s" % (type(exc).__name__, exc))

    # --------------------------------------------------- every documented key is read, and reverse
    check_true("subset keys/all read by the compiler",
               documented_keys <= set(code_keys),
               "documented but never looked up: %s" % sorted(documented_keys - set(code_keys)))
    rejected_keys = set()
    for row in props:
        try:
            prop = run_cell(cell_value(row[0]))
        except Exception:
            continue                                      # the `"boolean"` row is not a dict
        if isinstance(prop, dict):
            rejected_keys.update(prop.keys())
    check_true("compiler keys/all documented",
               set(code_keys) <= (documented_keys | rejected_keys),
               "read by laya/structured.py and absent from the tables: %s"
               % sorted(set(code_keys) - documented_keys - rejected_keys))
    check_true("docs name `title` as not read",
               re.search(r"`title` is \*\*not\*\* read", text) is not None,
               "the page has to say which keys it deliberately ignores")
    check_true("`title` really is unread", "title" not in code_keys,
               "the compiler now reads title; move it into the subset table")

    # ------------------------------------------------------ the rejections table, row by row
    raised = []
    for row in props:
        raw = cell_value(row[0])
        want = cell_value(row[1])
        try:
            prop = run_cell(raw)
        except Exception as exc:
            FAIL.append("rejections row/%s: cell raises %s: %s" % (raw, type(exc).__name__, exc))
            continue
        try:
            S.plan_from_json_schema({"type": "object", "properties": {"name": prop}})
            FAIL.append("rejections row/%s: documented as a rejection, compiles fine" % raw)
        except S.SchemaError as exc:
            raised.append(str(exc))
            check("rejections row/%s message" % raw, str(exc), want)
        except Exception as exc:
            FAIL.append("rejections row/%s: raised %s, not SchemaError: %s"
                        % (raw, type(exc).__name__, exc))

    for row in calls:
        raw, want = cell_value(row[0]), cell_value(row[1])
        try:
            run_cell(raw, ns)
            FAIL.append("entry point row/%s: documented as a rejection, returns" % raw)
        except S.SchemaError as exc:
            raised.append(str(exc))
            check("entry point row/%s message" % raw, str(exc), want)
        except Exception as exc:
            FAIL.append("entry point row/%s: raised %s, not SchemaError: %s"
                        % (raw, type(exc).__name__, exc))

    # every message the module can raise has a row, and no row documents a message it cannot raise
    unmapped = [m for m in raised if not any(matches(lit, m) for lit in literals)]
    check_true("docs/raised messages exist in the compiler", not unmapped, unmapped)
    orphans = []
    for lit in literals:
        hits = [m for m in raised if matches(lit, m)]
        if len(hits) != 1:
            orphans.append("%s -> %d rows" % (lit, len(hits)))
    check_true("compiler/every SchemaError message is documented", not orphans, orphans)
    check("messages/one row each", len(raised), len(literals))

    # ------------------------------------------------------------------ limits and projections
    limits = dict((m.group(1), int(m.group(2)))
                  for m in re.finditer(r"`(MAX_\w+) = (\d+)`", text))
    check("limits line/names found", sorted(limits),
          ["MAX_OPTIONS", "MAX_PROPERTIES", "MAX_SCORE_LEVELS"])
    for name, value in sorted(limits.items()):
        check("limits/%s matches the module" % name, getattr(S, name), value)

    exact = re.search(r"an `enum: (\[[^\]]*\])` returns `([^`]*)`, not `([^`]*)`", text)
    check_true("projection/sentence parses", exact is not None, "the wording this gate reads changed")
    if exact:
        values = json.loads(exact.group(1))
        got = S.answers_to_json({"field": {"choice": exact.group(2)}},
                                {"type": "object", "properties": {"field": {"enum": values}}})
        check("projection/an enum keeps its type", got, {"field": values[1]})
        check_true("projection/the value is not the label",
                   got["field"] != exact.group(2).strip('"'), repr(got))

    bounded = re.search(r"a bounded integer returns a\s*\n?level between `minimum` and `maximum`",
                        text)
    check_true("projection/sentence about a bounded integer parses", bounded is not None,
               "the wording this gate reads changed")
    if bounded:
        schema = {"type": "object", "properties": {"field": {"type": "integer",
                                                             "minimum": 3, "maximum": 7}}}
        # `score` is the 0-based level index (5 levels: 0..4 for minimum=3, maximum=7), never
        # the absolute field value -- see `DecisionModel._decode_answers` in laya/agent.py.
        for score, want in ((0, 3), (2, 5), (4, 7)):
            got = S.answers_to_json({"field": {"type": "score", "score": score}}, schema)
            check("projection/minimum + argmax for score %d" % score, got, {"field": want})

    noul = re.search(r"a boolean is `noul >= ([\d.]+)`", text)
    check_true("projection/sentence about a boolean parses", noul is not None,
               "the wording this gate reads changed")
    if noul:
        cut = float(noul.group(1))
        schema = {"type": "object", "properties": {"field": {"type": "boolean"}}}
        for value, want in ((cut, True), (cut - 0.001, False)):
            got = S.answers_to_json({"field": {"type": "noul", "noul": value}}, schema)
            check("projection/noul %.3f is %s" % (value, want), got, {"field": want})

    # a nullable field with no answer is absent, which is what the subset table's last column claims
    nullish = [r for r in subset if "anyOf" in r[0]]
    check_true("subset table/one row documents `anyOf`", len(nullish) == 1, "got %d" % len(nullish))
    if len(nullish) == 1:
        try:
            schema = {"type": "object", "properties": {"field": run_cell(cell_value(nullish[0][0]))}}
            check("projection/Optional with no answer omits the key",
                  S.answers_to_json({}, schema), {})
            check("projection/Optional still answers",
                  S.answers_to_json({"field": {"choice": "x"}}, schema), {"field": "x"})
        except Exception as exc:
            FAIL.append("subset row/anyOf: %s: %s" % (type(exc).__name__, exc))

    # ---------------------------------------------------------------------- the API table
    def resolve(name: str) -> Any:
        """What a documented name refers to: `laya.x`, `agent.x`/`router.x`, or the module's own."""
        parts = name.split(".")
        if parts[0] == "laya":
            obj: Any = laya
            rest = parts[1:]
        elif parts[0] in ("agent", "router"):
            obj = laya.Agent if parts[0] == "agent" else laya.Router
            rest = parts[1:]
        else:
            obj, rest = S, parts
        for part in rest:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    for row in api:
        for name in re.findall(r"`([\w.]+)\(", row[0]):
            check_true("api row/%s resolves" % name, callable(resolve(name)),
                       "no such callable")
    check("decide/keyword-only arguments",
          [p.name for p in inspect.signature(S.decide).parameters.values()
           if p.kind is p.KEYWORD_ONLY],
          ["questions", "return_details", "min_confidence"])
    check_true("decide/forwards predict kwargs",
               any(p.kind is p.VAR_KEYWORD for p in inspect.signature(S.decide).parameters.values()),
               "the page promises `**predict_kwargs`")
    # ...and the other direction: a public name of this module cannot go undocumented. The rule is
    # per-kind, because a loose one is a fake: `plan_from_json_schema` also appears in the rejection
    # tables, so "the name is somewhere on the page" passed with its API row deleted (mutant m5).
    api_column = " ".join(row[0] for row in api)
    for name in S.__all__:
        target = getattr(S, name, None)
        if inspect.isclass(target):
            check_true("exports/%s is described on the page" % name, name in text,
                       "a class no section explains")
        else:
            check_true("exports/%s has an API row" % name,
                       re.search(r"\b%s\(" % re.escape(name), api_column) is not None,
                       "not named as a callable in `The API`'s first column")

    # the module docstring teaches the same subset; it must not teach a key the code ignores
    doc = ast.get_docstring(tree) or ""
    check_true("module docstring/says the mapping is a documented subset",
               "documented subset" in doc, repr(doc[:80]))
    check_true("module docstring/does not promise `title`", "`title`" not in doc and " title " not in doc,
               "the docstring names title; the compiler does not read it")

    return report()


def report() -> int:
    print("%d checks passed, %d failed" % (len(PASS), len(FAIL)))
    for line in FAIL:
        print("FAIL %s" % line)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
