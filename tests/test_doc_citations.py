"""Every prose pointer from this repository's text into its code has to resolve.

Two forms are used, and each fails in its own way:

  * ``laya/router.py::_question_schema`` -- names a symbol, so it cannot rot, but it does go
    wrong: a rename leaves a citation to a function that no longer exists and nothing in the tree
    notices. This resolves each one against the cited file's ``ast``, including the owner when the
    prose claims one.
  * ``laya/router.py:141`` -- rots on its own, silently, every time the code above it grows.

Measured across 372 tracked text files on this branch before anything was fixed: 11 ``:NNN``
citations name a symbol in the same breath as the number, and **7 of those 11 are wrong** -- 3
stale numbers and 4 claims about an owner that does not exist. ``Agent._to_internal`` is cited at
``laya/agent.py:736-748`` in two places while the function is at 1109-1130, and
``_resolve_noul_labels`` at ``laya/common.py:92`` while it is at 113-124; line 141 is inside
``_check_checkpoint_policy``, a few lines from a ``warnings.warn``. Three sites call the schema key
``Router._question_schema``, but it is a module function at ``laya/router.py:221`` with no class
around it, so ``getattr(Router, "_question_schema", None)`` is ``None``; one cites
``OnnxAgent._decode_answers`` where the class has always been ``ONNXAgent``.

The line numbers are the failure a reader pays for. A citation exists so the reviewer can open the
file and see the claim; landing on an unrelated line costs them the search the pointer was written
to avoid, and reads as a claim nobody checked. So the three stale sites and the three ``Router.``
sites are rewritten into the ``::symbol`` form -- which eight citations elsewhere in the tree
already used, and twenty-three do now -- and this suite holds both forms to the tree: a ``::``
citation must resolve, and a ``:NNN`` citation written next to a symbol must fall inside it and
name that symbol's real owner. The two ``_decode_answers`` citations keep their line ranges on
purpose: they point at fragments of a long function, they are correct today, and they are the
subjects that keep the second check from going vacuous.

Deterministic and cheap: parse the tree, compare, print. No weights, no GPU, no network.

Run: python tests/test_doc_citations.py
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.relpath(os.path.abspath(__file__), ROOT).replace(os.sep, "/")

PASS: List[str] = []
FAIL: List[str] = []


def check_true(what: str, cond: bool, detail: object = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def _shown(named: Sequence["Symbol"]) -> str:
    return ", ".join(str(s) for s in named)


class Symbol:
    """A name a ``::`` citation could mean, and the scope that defines it."""

    __slots__ = ("name", "parent", "start", "end")

    def __init__(self, name: str, parent: Optional[str], start: int, end: int) -> None:
        self.name, self.parent, self.start, self.end = name, parent, start, end

    def __str__(self) -> str:
        return "%s%s (%d-%d)" % ("%s." % self.parent if self.parent else "",
                                 self.name, self.start, self.end)


DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _collect(stmts: Sequence[ast.stmt], parent: Optional[str], out: List[Symbol]) -> None:
    for node in stmts:
        if isinstance(node, DEFS):
            out.append(Symbol(node.name, parent, node.lineno, node.end_lineno or node.lineno))
            _collect(node.body, node.name, out)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(target, ast.Name):
                    out.append(Symbol(target.id, parent, node.lineno,
                                      node.end_lineno or node.lineno))
        elif isinstance(node, ast.If):
            # A module-level guard (``if TYPE_CHECKING:``) still defines what it holds.
            _collect(node.body, parent, out)
            _collect(node.orelse, parent, out)


_CACHE: Dict[str, List[Symbol]] = {}


def symbols(rel: str) -> List[Symbol]:
    if rel not in _CACHE:
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        try:
            out: List[Symbol] = []
            _collect(ast.parse(open(path, encoding="utf-8").read()).body, None, out)
        except (OSError, SyntaxError, UnicodeDecodeError):
            out = []
        _CACHE[rel] = out
    return _CACHE[rel]


DBLCOLON = re.compile(r'(?P<file>[A-Za-z0-9_.\-/]+\.py)::(?P<name>[A-Za-z_][A-Za-z0-9_]*'
                      r'(?:\.[A-Za-z_][A-Za-z0-9_]*)?)')
LINECITE = re.compile(r'(?P<file>[A-Za-z0-9_.\-/]+\.py):(?P<a>\d+)(?:\s*-\s*(?P<b>\d+))?')
IDENT = re.compile(r'`(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`')

# What may sit between a backticked symbol and the citation after it. Punctuation and spacing
# cover both live idioms -- "`Foo.bar` (`file:12`)" and "`Foo.bar`, (`file:12`)" -- and the
# connectors cover "`Router.predict_batch` -- see laya/router.py:961". Longer than that is a new
# clause, and binding across a clause is how a checker starts inventing violations.
GAP_LIMIT = 24
CONNECTORS = {"see", "via", "at", "in", "as", "is", "where", "shows", "says", "and"}

# A site this suite knows is stale but must not fail on, because the pull request that owns those
# lines is in review. The key is (citing file, cited file, symbol); the row retires when that
# change merges, and the check starts applying to the site again.
DEFERRED: Dict[Tuple[str, str, str], str] = {
    ("tests/test_example_server_per_call_controls.py", "laya/router.py", "predict_batch"):
        "#1086 rewrites both comments that carry this citation (its hunks are @@ -22,15 and "
        "@@ -270,9), so a sibling change across the same lines mid-review is worse than waiting. "
        "Once #1086 merges, delete this row and rewrite `laya/router.py:961` as "
        "`laya/router.py::Router.predict_batch`.",
}

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", ".mypy_cache",
             ".pytest_cache", "site-packages"}


def target_of(cited: str) -> Optional[str]:
    """Resolve a cited path in the tree, tolerating the ``laya/x.py`` and ``x.py`` spellings."""
    for cand in ({cited, cited.split("/", 1)[-1]} if "/" in cited else {cited}):
        if cand.endswith(".py") and os.path.isfile(os.path.join(ROOT, cand.replace("/", os.sep))):
            return cand
    return None


def text_files() -> List[str]:
    try:
        listed = [f for f in subprocess.check_output(["git", "-C", ROOT, "ls-files"],
                                                     text=True).split("\n") if f]
    except (OSError, subprocess.SubprocessError):
        listed = []
        for base, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            listed.extend(os.path.relpath(os.path.join(base, n), ROOT).replace(os.sep, "/")
                          for n in names)
    return sorted(f for f in listed if f != SELF and f.endswith((".py", ".md"))
                  and not any(d in f.split("/") for d in SKIP_DIRS))


def bound_symbol(text: str, pos: int) -> Optional[str]:
    """The backticked identifier a citation is attached to, or None when it is attached to none."""
    matches = list(IDENT.finditer(text, 0, pos))
    if not matches:
        return None
    gap = text[matches[-1].end():pos]
    if len(gap) > GAP_LIMIT or gap.count("`") > 1:
        return None
    if any(word not in CONNECTORS for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', gap)):
        return None
    return matches[-1].group("name")


def cites_inside(lo: int, hi: int, named: Sequence["Symbol"]) -> bool:
    """Whether the cited range falls inside the symbol it names -- a fragment of a function is a
    legitimate citation, so overlap is the test, not containment."""
    return any(s.start <= hi and lo <= s.end for s in named)


def positive_controls() -> None:
    """Prove the two detectors fire, against a symbol read from the tree rather than a literal.

    Without this, a checker that always returns "fine" passes exactly as loudly as one that works.
    The spans come from `laya/agent.py` as it is now, so the fixtures cannot themselves go stale.
    """
    fn = [s for s in symbols("laya/agent.py") if s.name == "_to_internal"]
    check_true("the control has a function to cite", bool(fn),
               "no `_to_internal` in laya/agent.py")
    if not fn:
        return
    first, last = fn[0].start, fn[0].end
    check_true("control: a range inside the named function is accepted",
               cites_inside(first, last, fn), "%d-%d rejected" % (first, last))
    check_true("control: a range 300 lines below it is rejected",
               not cites_inside(last + 1, last + 300, fn), "")
    check_true("control: line 1 of the cited file is rejected", not cites_inside(1, 1, fn), "")

    stale = "`Agent._to_internal` (`laya/agent.py:%d`)" % (last + 300)
    got = bound_symbol(stale, stale.index("laya/agent.py"))
    check_true("control: the `symbol` (`file:NNN`) idiom binds",
               got == "Agent._to_internal", got)
    far = ("`Agent._to_internal` is the function that normalizes instructions before hashing, "
           "and it lives at (`laya/agent.py:%d`)" % last)
    got = bound_symbol(far, far.index("laya/agent.py"))
    check_true("control: a citation two clauses away does not bind", got is None, got)
    form = "`laya/agent.py::Agent._to_internal`"
    got = bound_symbol(form, form.index("Agent._to_internal"))
    check_true("control: a `::` citation is not read as a line citation", got is None, got)


def main() -> int:
    positive_controls()
    files = text_files()
    check_true("the scan covers the tree", len(files) > 300, "%d text files" % len(files))

    dbl_total = 0
    dbl_owner_claims = 0
    unresolved: List[str] = []
    bound = 0
    drifted: List[str] = []
    deferred: List[Tuple[Tuple[str, str, str], str]] = []
    unbound = 0

    for rel in files:
        try:
            text = open(os.path.join(ROOT, rel.replace("/", os.sep)),
                        encoding="utf-8", errors="ignore").read()
        except OSError:
            continue

        def where(pos: int, _rel: str = rel, _text: str = text) -> str:
            return "%s:%d" % (_rel, _text.count("\n", 0, pos) + 1)

        for m in DBLCOLON.finditer(text):
            tgt = target_of(m.group("file"))
            if tgt is None:
                continue
            dbl_total += 1
            owner, _, leaf = m.group("name").rpartition(".")
            named = [s for s in symbols(tgt) if s.name == leaf]
            if not named:
                unresolved.append("%s cites %s, which %s does not define"
                                  % (where(m.start()), m.group(0), tgt))
            elif owner:
                dbl_owner_claims += 1
                if not any(s.parent == owner for s in named):
                    unresolved.append("%s cites %s, but %s defines it as %s"
                                      % (where(m.start()), m.group(0), tgt, _shown(named)))

        for m in LINECITE.finditer(text):
            tgt = target_of(m.group("file"))
            if tgt is None:
                continue
            name = bound_symbol(text, m.start())
            if name is None:
                unbound += 1
                continue
            owner, _, leaf = name.rpartition(".")
            named = [s for s in symbols(tgt) if s.name == leaf]
            if not named:
                unbound += 1
                continue
            bound += 1
            key = (rel, tgt, leaf)
            if key in DEFERRED:
                deferred.append((key, "%s -> %s (names %s)"
                                 % (where(m.start()), m.group(0), name)))
                continue
            if owner and not any(s.parent == owner for s in named):
                drifted.append("%s calls it %s, but %s defines it as %s"
                               % (where(m.start()), name, tgt, _shown(named)))
                continue
            lo, hi = int(m.group("a")), int(m.group("b") or m.group("a"))
            if not cites_inside(lo, hi, named):
                drifted.append("%s puts %s at %s:%d, but it runs %s"
                               % (where(m.start()), name, tgt, lo, _shown(named)))

    check_true("every `file::symbol` citation resolves to a definition",
               not unresolved, unresolved)
    check_true("the `file::symbol` check has subjects", dbl_total >= 12,
               "only %d citations of that form" % dbl_total)
    check_true("no `file:NNN` citation lands outside the symbol named beside it",
               not drifted, drifted)
    check_true("the line-citation check has subjects", bound >= 3,
               "only %d citations bind to a named symbol" % bound)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    print("%d text files scanned from %s" % (len(files), ROOT))
    print("%d `file::symbol` citation(s) resolve, %d of them claiming an owner"
          % (dbl_total, dbl_owner_claims))
    print("%d `file:NNN` citation(s) name a symbol beside them; %d of those land inside it"
          % (bound, bound - len(deferred) - len(drifted)))
    for key, site in deferred:
        print("deferred: %s -- %s" % (site, DEFERRED[key]))
    print("not checkable from this repository (not asserted either way):")
    print("  - %d `file:NNN` citation(s) with no symbol named beside them -- a drift there is "
          "invisible, which is why the `::` form is the one to write" % unbound)
    print("  - citations to files outside the tree, and to non-Python targets")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
