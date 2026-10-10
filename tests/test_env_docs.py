"""Every ``LAYA_*`` the package reads has to be written down where a user can find it.

Two names drifted out of reach of the documentation and nothing noticed:

  * ``LAYA_MAX_CONCURRENT`` appears only inside ``laya/serve.py``'s module docstring.
    That table is the developer-facing copy; no page renders it, and the server table in
    ``docs/docker.md`` lists nine variables and not this one -- so the admission-control
    knob behind a ``503`` is invisible to the deployment that hits it.
  * ``LAYA_MPS_AMP_MIN_ROWS`` is named nowhere at all, one sentence away from its
    documented ``LAYA_CUDA_AMP`` / ``LAYA_CPU_AMP`` siblings, even though it changes which
    forwards run in fp16 on Apple Silicon and therefore which probabilities come back.

And a name being written down is not the same as what it accepts being written down correctly.
``LAYA_CUDA_AMP`` and ``LAYA_CPU_AMP`` are the two names in the package whose accepted values are
a literal tuple, and both of the places that list them are wrong: ``docs/docker.md``'s
``LAYA_CPU_AMP`` row says "``bf16`` opts the CPU forward into bf16; anything else leaves it fp32"
while the CPU comparison accepts ``bf16`` and ``bfloat16`` alike, and ``_cuda_amp_dtype``'s own
docstring -- the text a maintainer editing the function reads -- says
"LAYA_CUDA_AMP=fp16|bf16 when set ... anything else is ignored" while the function accepts four
spellings. The error is in the direction that hides a working knob: ``LAYA_CPU_AMP=bfloat16``
turns bf16 on today and the page promises it will not, so the deployment that trusts the page
never asks for a precision it is already entitled to.

A review note fixes each of these once. The names live in three places -- the package, the
markdown, and the compose files -- and a list written down in any one of them goes stale
the moment an ``os.environ`` read is added in another, so this derives the set instead of
transcribing it: it walks ``laya/`` with ``ast`` and compares what it finds against what
the docs mention, in both directions.

A default column goes stale the same way and is harder to see, because the row still reads
correctly. Both surfaces that document the package's own defaults say ``LAYA_DEFAULT_MODEL``
falls back to ``english``. Since 0.4.0 it falls back to ``multilingual``: the breaking routing
change moved ``Router(default=...)`` and four pages with it, but not ``docs/http-api.md`` nor the
table in ``laya/serve.py``'s docstring, and the server leaves the keyword out when the variable is
unset precisely so that the value "cannot drift" from ``Router``'s own. ``tests/test_serve.py``
already pins the resolved ``router.default``; it never reads a table, so nothing compared the two
columns to the code they describe. The operator who trusts the stale cell is asking for the
checkpoint that change exists to stop -- off English it collapses while staying confident, so no
downstream ``min_confidence`` gate catches it. So the default column is derived too, from three
read shapes, and held on the two surfaces that describe the package rather than a container.

Why the whole string-constant sweep rather than only ``os.environ.get("LAYA_...")``: the
narrow form misses two of the fourteen names the package actually consumes, because
``_env_bool("LAYA_AUTO_TASK", ...)`` and ``os.environ.get(_ENV_KEY)`` hide the literal from
a call-shape matcher. Over-catching is the safe direction -- a name the package puts in a
string is a name a user can set.

The same ``ast`` + file-walk approach as ``tests/test_doc_tables.py``: no weights, no GPU,
no network, deterministic.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict, List, Set

PASS: List[str] = []
FAIL: List[str] = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "laya")
NAME = re.compile(r"\bLAYA_[A-Z0-9_]+\b")

# Markdown a user reads for configuration. The package's own docstrings are deliberately
# not here: rendering them is not wired up, which is how LAYA_MAX_CONCURRENT went missing.
DOC_FILES = ("README.md", "BENCHMARKS.md", "CONTRIBUTING.md")
SKIP_DIRS = {".git", ".github", ".cache", ".venv", "site", "__pycache__", "node_modules",
             "dist", "build", ".pytest_cache", ".ruff_cache"}

# Documented elsewhere already, by a pull request that has not merged. Each entry has to
# stay *undocumented* to remain valid, so the row retires itself the moment the page lands
# and cannot become a permanent hole in the sweep.
DEFERRED: dict = {}

# The half-precision spellings a document could offer for the two AMP names. Kept as the closed
# universe of things a prose site can mean rather than as an accepted list: the accepted list is
# derived from laya/agent.py below, and a spelling outside this set (`fp32`, `int8`) is not a
# value anyone offers -- it is what the forward falls back to, and pages name it for that.
DTYPE_SPELLINGS = ("fp16", "float16", "bf16", "bfloat16")
AMP_VARS = ("LAYA_CUDA_AMP", "LAYA_CPU_AMP")
BACKTICK = re.compile(r"`([^`\n]+)`")
# A sentence that claims its own list is the whole vocabulary. Only these phrases make a doc
# falsifiable by the code's set; "`bf16` is the CPU counterpart" names one spelling and promises
# nothing about the others, so it stays a passing mention instead of a failure. `\s` rather than a
# space because prose wraps, and the package's own docstring breaks "anything" from "else" across
# a line -- which is how a first version of this sweep read green while never comparing it.
COMPLETE_CLAIM = re.compile(r"\banything\s+(?:else|other)\b|\bnothing\s+else\b|"
                            r"\b(?:recognized|recognised|accepted|valid)\s+spellings?\b", re.I)


def check(what: str, got: object, want: object) -> None:
    if got == want:
        PASS.append(what)
    else:
        FAIL.append("%s: got %r, want %r" % (what, got, want))


def check_true(what: str, cond: bool, detail: object = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def read(path: str) -> str:
    # newline="" so a CRLF checkout and an LF checkout parse identically
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read().replace("\r\n", "\n")


def walk(base: str, *dirs: str):
    """Every file under base/*, with vendored and generated trees cut out."""
    start = os.path.join(base, *dirs)
    if os.path.isfile(start):
        yield start
        return
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            yield os.path.join(dirpath, fn)


def package_names() -> Dict[str, Set[str]]:
    """LAYA_* names appearing in a string constant anywhere under laya/."""
    found: Dict[str, Set[str]] = {}
    for path in walk(ROOT, "laya"):
        if not path.endswith(".py"):
            continue
        tree = ast.parse(read(path), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name in NAME.findall(node.value):
                    found.setdefault(name, set()).add(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    return found


def environ_call_names() -> Set[str]:
    """The narrow capture: names passed straight to os.environ.get/[]. Only used to show
    that it misses part of the set, which is why package_names() is the one that gates."""
    found: Set[str] = set()
    for path in walk(ROOT, "laya"):
        if not path.endswith(".py"):
            continue
        tree = ast.parse(read(path), filename=path)
        for node in ast.walk(tree):
            args: List[ast.expr] = []
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Attribute) \
                    and node.func.value.attr == "environ" and node.args:
                args = [node.args[0]]
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                    and node.value.attr == "environ":
                args = [node.slice]
            for arg in args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.update(NAME.findall(arg.value))
    return found


def markdown_names() -> Dict[str, Set[str]]:
    found: Dict[str, Set[str]] = {}
    paths = [os.path.join(ROOT, f) for f in DOC_FILES if os.path.isfile(os.path.join(ROOT, f))]
    paths += [p for p in walk(ROOT, "docs") if p.endswith(".md")]
    for path in paths:
        for name in NAME.findall(read(path)):
            found.setdefault(name, set()).add(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    return found


def non_markdown_names() -> Dict[str, Set[str]]:
    """Everywhere a variable can be consumed outside prose: Python, compose, Dockerfile,
    nix, shell. A name that appears only in markdown is documented but wired to nothing.
    `tests/` is excluded on purpose -- a test that greps the docs cannot be the reason a
    variable exists.
    """
    found: Dict[str, Set[str]] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and d != "tests")
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            if fn.endswith((".md", ".pyc", ".json", ".safetensors", ".bin", ".png", ".ico")):
                continue
            try:
                text = read(path)
            except (UnicodeDecodeError, OSError):
                continue
            for name in NAME.findall(text):
                found.setdefault(name, set()).add(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    return found


def amp_accepted(var: str) -> Set[str]:
    """The spellings laya/agent.py compares `var` against, read out of the comparisons themselves.

    ``LAYA_CUDA_AMP`` and ``LAYA_CPU_AMP`` are the only names in the package whose accepted values
    are a literal tuple -- everything else is an int, a path, or ``_env_bool``'s shared truthy
    set -- so the vocabulary is a property of the code that can be recovered rather than a list
    written down here. Same window as the sibling gate in ``tests/test_packaging.py``, narrowed to
    dtype spellings: without the filter, an unrelated ``in (...)`` inside 300 characters of the
    lookup would widen the set, and a widened set only ever hides a page that overpromises.
    """
    src = read(os.path.join(PACKAGE, "agent.py"))
    out: Set[str] = set()
    for m in re.finditer(r'environ\.get\("%s"' % var, src):
        for group in re.findall(r"\bin \(([^)]*)\)", src[m.end():m.end() + 300]):
            out.update(t for t in re.findall(r'"([^"]+)"', group) if t in DTYPE_SPELLINGS)
    return out


def prose_units():
    """Yield ``(location, text)`` for each chunk of prose that can carry a claim about a value.

    One unit per markdown table row, then per paragraph, then per package string constant that
    names an AMP variable. Row granularity is the point: the configuration table is where a
    variable's accepted spellings are defined, and a whole-file match would let the CUDA row's
    ``fp16`` satisfy the CPU row's claim about ``bfloat16`` -- the two sets differ, and that
    difference is the entire content of the two options.
    """
    paths = [os.path.join(ROOT, f) for f in DOC_FILES if os.path.isfile(os.path.join(ROOT, f))]
    paths += [p for p in walk(ROOT, "docs") if p.endswith(".md")]
    for path in paths:
        # posix separators because the anchors below name `docs/docker.md` and `laya/agent.py`
        # literally, and CI runs this suite on `windows-latest` too.
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        para: List[str] = []
        start = 1
        for i, line in enumerate(read(path).splitlines(), 1):
            if line.startswith("|"):
                yield "%s:%d" % (rel, i), line
                continue
            if not line.strip():
                if para:
                    yield "%s:%d" % (rel, start), "\n".join(para)
                para, start = [], i + 1
                continue
            if not para:
                start = i
            para.append(line)
        if para:
            yield "%s:%d" % (rel, start), "\n".join(para)
    for path in walk(ROOT, "laya"):
        if not path.endswith(".py"):
            continue
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        for node in ast.walk(ast.parse(read(path), filename=path)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and any(v in node.value for v in AMP_VARS):
                yield "%s:%d" % (rel, node.lineno), node.value


def offered(text: str, var: str) -> Set[str]:
    """Spellings this unit presents as values of `var`: ``VAR=fp16|bf16`` and backticked tokens.

    Backticks are what separates a spelling being offered from a precision being described.
    ``README.md`` writes "MPS autocasts in fp16 too ... it says float16 even when every call stays
    below the gate" in the same paragraph that names ``LAYA_CPU_AMP``, and neither of those tokens
    is a value that variable takes -- they are torch dtypes a returned field reports. Requiring the
    backtick keeps that paragraph a description instead of turning it into a promise this gate
    would fail it over.
    """
    out: Set[str] = set()
    for m in re.finditer(r"%s=([A-Za-z0-9|]+)" % var, text):
        out.update(t for t in m.group(1).split("|") if t in DTYPE_SPELLINGS)
    if var in text:
        out.update(t for t in BACKTICK.findall(text) if t in DTYPE_SPELLINGS)
    return out


# ------------------------------------------------------------------ a documented default
# Two cells promised ``english`` for ``LAYA_DEFAULT_MODEL`` while ``Router(default="multilingual")``
# has answered those requests since 0.4.0: the breaking routing change moved the fallback and the
# two environment tables were not part of it. Every other row on both pages is right, so this is
# not a page nobody reads -- it is a page that reads well and went stale, which is the failure a
# prose review cannot catch and a comparison against the code can. The comparison derives the
# code's answer from three read shapes rather than transcribing a table of defaults here:
#
#   R1  the read site carries it: ``os.environ.get("LAYA_X", D)`` and ``_env_bool("LAYA_X", D)``.
#   R2  a resolver reads with no literal default and answers the unset path with one, found through
#       the guard on the variable that read assigned (`raw = ...` then `if not raw: return DEFAULT_X`).
#   R3  a resolver sends *nothing* when unset -- it returns ``{}`` or ``None`` and the caller leaves
#       the keyword out of ``Router(**options)`` -- so the answer is ``Router``'s own parameter
#       default. The keyword comes from the one-key dict literal the resolver returns when the
#       variable *is* set, or from the `options["key"] = value` line that forwards it.
#
# Only the two surfaces that document *the package's* defaults are held to this. ``docs/docker.md``
# and ``docs/cli-mcp.md`` document the Compose service and the stdio server, whose defaults are
# allowed to differ, which is the row in "what this cannot check" below.
ROUTER_CLASS = "Router"
PACKAGE_DEFAULT_SURFACES = ("laya/serve.py docstring", "docs/http-api.md")
TABLE_RULE = re.compile(r"[ =]+")
DEFAULT_GUTTER = 4


def is_read(node):
    """``(name, default_node)`` when this call is a ``LAYA_*`` read, else None.

    ``_env_bool`` is in the set because ``LAYA_AUTO_TASK`` and ``LAYA_PRELOAD`` go through it, and
    the narrow ``os.environ`` matcher this file already reports as insufficient misses them.
    """
    if not (isinstance(node, ast.Call) and node.args):
        return None
    func = node.func
    environ = isinstance(func, ast.Attribute) and func.attr == "get" \
        and isinstance(func.value, ast.Attribute) and func.value.attr == "environ"
    if not (environ or (isinstance(func, ast.Name) and func.id == "_env_bool")):
        return None
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
        return None
    found = NAME.findall(first.value)
    if len(found) != 1:
        return None
    return found[0], (node.args[1] if len(node.args) > 1 else None)


def own_nodes(func):
    """Nodes in this function's own body, in source order, nested functions left out.

    Nesting matters twice over: ``create_app`` would otherwise "read" every name its request
    handlers read and be judged ambiguous, and a caller's ``options["k"] = v`` line has to be seen
    after the ``v = F(...)`` line that gives it meaning.
    """
    out, stack = [], [func]
    while stack:
        node = stack.pop(0)
        out.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    and child is not func:
                continue
            stack.append(child)
    return out


def literal_of(node, consts):
    """A constant, or a module-level constant's value; ``ast.Constant`` covers None and "".

    Returns ``(found, value)`` rather than ``value`` because ``False`` and ``0`` are answers the
    documentation has to be able to agree with.
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return True, consts[node.id]
    return False, None


def module_consts(tree):
    out = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name) \
                and isinstance(stmt.value, ast.Constant):
            out[stmt.targets[0].id] = stmt.value.value
    return out


def read_names(func):
    return {site[0] for node in own_nodes(func) for site in [is_read(node)] if site}


def router_param_defaults():
    """``Router.__init__``'s parameter defaults, as written in laya/router.py.

    ``None`` is dropped: it is the same "no opinion" the resolver expressed, so it cannot be the
    scalar a default column names.
    """
    tree = ast.parse(read(os.path.join(ROOT, "laya", "router.py")))
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == ROUTER_CLASS]:
        for func in cls.body:
            if isinstance(func, ast.FunctionDef) and func.name == "__init__":
                names = [a.arg for a in func.args.args[1:]]
                defaults = func.args.defaults
                pad = len(names) - len(defaults)
                out = {}
                for i, dflt in enumerate(defaults):
                    ok, value = literal_of(dflt, {})
                    if ok and value is not None:
                        out[names[pad + i]] = value
                return out
    return {}


def forwarded_keyword(func, funcs, name):
    """The ``Router`` keyword a send-nothing resolver answers for, recovered from the caller.

    Two shapes, both real in laya/serve.py: ``options.update(F(...))`` where F returns a one-key
    dict literal, and ``value = F(...)`` followed by ``options["key"] = value``. The second one is
    only a keyword when the same function then does ``Router(**options)``, and which local that is
    comes out of the call rather than out of its name."""
    for node in own_nodes(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict) \
                and len(node.value.keys) == 1:
            key = node.value.keys[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    for holder in funcs.values():
        if holder is func:
            continue
        nodes = own_nodes(holder)
        expanded = set()
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == ROUTER_CLASS:
                expanded.update(kw.value.id for kw in node.keywords
                                if kw.arg is None and isinstance(kw.value, ast.Name))
        called = {}
        for node in nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Call) \
                    and isinstance(node.value.func, ast.Name):
                inner = funcs.get(node.value.func.id)
                if inner is not None:
                    called[node.targets[0].id] = inner
        for node in nodes:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) \
                    and target.value.id in expanded \
                    and isinstance(target.slice, ast.Constant) \
                    and isinstance(target.slice.value, str) \
                    and isinstance(node.value, ast.Name):
                inner = called.get(node.value.id)
                if inner is not None and read_names(inner) == {name}:
                    return target.slice.value
    return None


def code_defaults():
    """What laya/serve.py does when a ``LAYA_*`` is unset.

    Returns ``(derived, unchecked, conflicts)``: the names answered with a single value, the names
    this file could not answer with the reason for each, and the names read with two different
    literal defaults at two read sites.
    """
    tree = ast.parse(read(os.path.join(PACKAGE, "serve.py")))
    consts = module_consts(tree)
    funcs = {}
    for func in [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        funcs.setdefault(func.name, func)

    read_defaults: Dict[str, Set[object]] = {}
    for node in ast.walk(tree):
        site = is_read(node)
        if not site:
            continue
        ok, value = literal_of(site[1], consts) if site[1] is not None else (False, None)
        # "" is not a default a page can name; it is the read's way of saying "unset", and the
        # function's own guard decides what that means.
        if ok and value is not None and value != "":
            read_defaults.setdefault(site[0], set()).add(value)
    # Two reads of one name that disagree about what unset means is a code bug, and naming either
    # answer would let the page match the code and still be wrong, so it is failed on.
    conflicts = {name: sorted(repr(v) for v in values)
                 for name, values in read_defaults.items() if len(values) > 1}
    derived = {name: next(iter(values)) for name, values in read_defaults.items()
               if len(values) == 1}

    unchecked = {}
    router = router_param_defaults()
    for name in sorted({n for func in funcs.values() for n in read_names(func)}):
        if name in derived or name in conflicts:
            continue
        candidates = [f for f in funcs.values() if name in read_names(f)]
        if len(candidates) != 1:
            unchecked[name] = "read by %d functions, so no single unset path to follow" % len(candidates)
            continue
        func = candidates[0]
        locals_ = set()
        for node in own_nodes(func):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and any(site and site[0] == name
                            for c in ast.walk(node.value)
                            for site in [is_read(c)] if isinstance(c, ast.Call)):
                locals_.add(node.targets[0].id)
        unset = None
        for node in own_nodes(func):
            if not isinstance(node, ast.If):
                continue
            if not locals_ & {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}:
                continue
            returns = [s for s in node.body if isinstance(s, ast.Return)]
            if returns:
                unset = returns[0].value
                break
        if unset is None:
            unchecked[name] = "no default at the read and no guard on the read's own variable"
            continue
        ok, value = literal_of(unset, consts)
        if ok and value is not None:
            derived[name] = value
            continue
        keyword = forwarded_keyword(func, funcs, name)
        if keyword is None:
            unchecked[name] = "the unset path sends nothing and no keyword names it"
        elif keyword not in router:
            unchecked[name] = "the unset path hands %r to %s, which defaults it to no value" \
                              % (keyword, ROUTER_CLASS)
        else:
            derived[name] = router[keyword]
    return derived, unchecked, conflicts


def docstring_default_table():
    """The environment table in laya/serve.py's module docstring, as ``{name: default cell}``.

    Hand-aligned, and its name column overflows the rule above it, so slicing at the rule's spans
    loses the long names. The default column is the one every row keeps to the right, so each row's
    text after its final run of two or more spaces is taken, and kept only when it starts in that
    column's gutter -- which is also what tells a wrapped meaning cell from a wrapped default.
    """
    doc = ast.get_docstring(ast.parse(read(os.path.join(PACKAGE, "serve.py")))) or ""
    lines = doc.splitlines()
    rules = [i for i, line in enumerate(lines) if TABLE_RULE.fullmatch(line) and "=" in line]
    if len(rules) < 3:
        return {}
    gutter = [m.start() for m in re.finditer(r"=+", lines[rules[0]])][-1] - DEFAULT_GUTTER
    rows, current = {}, None
    for line in lines[rules[1] + 1:rules[2]]:
        runs = list(re.finditer(r"\s{2,}", line))
        piece = line[runs[-1].end():].strip() if runs else ""
        on_column = bool(runs) and runs[-1].end() >= gutter and bool(piece)
        found = NAME.findall(line[:40])
        if len(found) == 1 and line.startswith("``"):
            current = found[0]
            rows[current] = [piece] if on_column else []
        elif current and on_column:
            rows[current].append(piece)
    return {name: " ".join(cells) for name, cells in rows.items()}


def markdown_default_table(rel_path):
    """A markdown configuration table's default column, for rows whose first cell is one name."""
    rows = {}
    for line in read(os.path.join(ROOT, *rel_path.split("/"))).splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        found = NAME.findall(cells[0])
        if len(found) == 1 and cells[0].strip("`") == found[0]:
            rows[found[0]] = cells[-1]
    return rows


def default_text(value):
    """How a scalar default reads in a table: booleans as the 1/0 the pages spell, 0.0 as 0."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def default_cell(cell):
    """A default column's text, stripped of the markup and of the gloss some rows carry.

    ``(`` ``english`` ``)``, ``0 (off)`` and ``none`` all state one value; the parentheses are the
    table's own convention for "this is the value nobody set anything to", and the trailing pair on
    ``0 (off)`` is a note about that value rather than a second value. A row whose whole cell is
    parenthesised is therefore read inside the parentheses, and one that ends with them keeps the
    part before.
    """
    text = cell.strip().strip("`").strip()
    if re.fullmatch(r"\(.*\)", text):
        return text[1:-1].strip()
    return re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()


def main() -> int:
    pkg = package_names()
    docs = markdown_names()
    tree = non_markdown_names()

    # ------------------------------------------------- the capture itself has to be working
    # If the walk or the parser silently found nothing, every check below would pass vacuously.
    witnesses = {"LAYA_DEVICE", "LAYA_CUDA_AMP", "LAYA_CPU_AMP", "LAYA_MAX_CONCURRENT",
                 "LAYA_MPS_AMP_MIN_ROWS", "LAYA_SHA256_DIGESTS", "LAYA_API_KEY"}
    check_true("sweep/reads the names the package is known to consume",
               witnesses <= set(pkg), sorted(witnesses - set(pkg)))
    check_true("sweep/found a plausible number of names", len(pkg) >= len(witnesses), len(pkg))
    # LAYA_AUTO_TASK goes through _env_bool() and LAYA_DEVICE through a module constant in
    # laya/mcp/device.py, so the call-shaped matcher is not enough on its own.
    missed = set(pkg) - environ_call_names()
    check_true("sweep/the narrow os.environ matcher misses names a string sweep finds",
               "LAYA_AUTO_TASK" in missed, sorted(missed))

    # ------------------------------------------------- direction 1: consumed -> documented
    missing = {n: v for n, v in pkg.items() if n not in docs}
    retired = {n: v for n, v in missing.items() if n in DEFERRED}
    unexplained = {n: sorted(v) for n, v in missing.items() if n not in DEFERRED}
    check_true("every LAYA_* the package reads is documented", not unexplained, unexplained)
    check("the only undocumented names are the deferred ones", sorted(retired), sorted(DEFERRED))

    # ------------------------------------------------- direction 2: documented -> consumed
    # A configuration page describing a variable nothing reads is worse than a gap: the
    # reader sets it and the program ignores it. The consumer may be the package, a compose
    # file's interpolation, the Dockerfile, or the Nix module.
    docs_only = {}
    for name, where in docs.items():
        consumers = tree.get(name, set())
        if not consumers:
            docs_only[name] = sorted(where)
    check_true("every documented LAYA_* is consumed somewhere in the tree",
               not docs_only, docs_only)

    # the two rows this suite was written for, pinned to the page that carries them, so a
    # later edit that deletes one fails by name rather than only as a set difference
    anchors = {"LAYA_MAX_CONCURRENT": "docs/docker.md",
               "LAYA_MPS_AMP_MIN_ROWS": "README.md"}
    for name, page in sorted(anchors.items()):
        check_true("%s is documented on %s" % (name, page),
                   page in docs.get(name, set()), sorted(docs.get(name, set())))

    # ------------------------------------------------- what a documented name accepts
    # Direction 1: no page may offer a spelling the code ignores. That is the failure that costs an
    # operator outright -- they set `LAYA_CPU_AMP=fp16` because the table named it, and the forward
    # stays in fp32 while the configuration reads as honored.
    # Direction 2: where a unit *defines* a variable (the table row whose first cell is its name) or
    # claims its list is the whole vocabulary ("anything else is ignored"), the spellings it names
    # have to be all of them. That is the failure this repository has in both sites today, and it is
    # the one a reader cannot work around: a working spelling documented as inert is a precision
    # nobody asks for.
    # Both directions read the accepted set out of laya/agent.py rather than from a list here, so
    # the gate cannot fall behind the code the way the two prose sites did.
    cuda, cpu = amp_accepted("LAYA_CUDA_AMP"), amp_accepted("LAYA_CPU_AMP")
    accepted = {"LAYA_CUDA_AMP": cuda, "LAYA_CPU_AMP": cpu}
    check_true("amp/the dtype vocabularies came from laya/agent.py",
               bool(cuda) and bool(cpu) and cuda != cpu,
               "cuda=%s cpu=%s -- if the lookups stop being literal `in (...)` tuples this needs "
               "retargeting; if they start agreeing, the CPU option's cross-reference is stale"
               % (sorted(cuda), sorted(cpu)))

    rows_sites, claim_sites = set(), set()
    for loc, text in prose_units():
        named = [v for v in AMP_VARS if v in text]
        if not named:
            continue
        # A unit that names both variables may name any spelling either one accepts. The union is
        # what keeps a sentence about both devices from reading as a promise about the CPU option,
        # which is the narrower of the two sets.
        allowed = set().union(*(accepted[v] for v in named))
        for var in named:
            over = sorted(offered(text, var) - allowed)
            check_true("amp/%s offers no spelling %s ignores" % (loc, var), not over,
                       "%s is offered against a set laya/agent.py compares %s against (%s): %s"
                       % (over, var, sorted(accepted[var]), text.strip()[:120]))
        first_cell = text.split("|")[1].strip() if text.startswith("|") else ""
        for var in named:
            if first_cell != "`%s`" % var:
                continue
            rows_sites.add((loc.split(":")[0], var))
            check("amp/%s row names every spelling %s accepts" % (loc, var),
                  sorted(offered(text, var)), sorted(accepted[var]))
        # Attributing "anything else" to a variable only means something when the unit names one.
        # A unit naming both, with one list between them, would fail for the wrong reason.
        if len(named) == 1 and COMPLETE_CLAIM.search(text) and offered(text, named[0]):
            claim_sites.add((loc.split(":")[0], named[0]))
            check("amp/%s's completeness claim lists what %s accepts" % (loc, named[0]),
                  sorted(offered(text, named[0])), sorted(accepted[named[0]]))
    # Non-vacuity by name rather than by count: the sweep has to be standing on the sites this was
    # written for, so a page that reworded its row, or a docstring that dropped its sentence,
    # cannot switch a direction off and leave the file green. A count would have passed both -- the
    # two docker.md rows alone satisfy any floor -- and the two drift sites are a row and a
    # docstring, which is exactly the pair a count cannot tell apart. The anchors name the *file*
    # for the claims rather than the file and variable, so extracting the CPU comparison into a
    # helper or inlining it again is a refactor and not a failure.
    check_true("amp/the defining rows are the two dtype rows",
               {("docs/docker.md", v) for v in AMP_VARS} <= rows_sites,
               "rows compared: %s" % sorted(rows_sites))
    check_true("amp/every dtype vocabulary has a completeness claim in front of it",
               {v for _f, v in claim_sites} == set(AMP_VARS),
               "claims compared: %s" % sorted(claim_sites))
    check_true("amp/the package's own docstrings are in the claim sweep",
               any(f == "laya/agent.py" for f, _v in claim_sites),
               "claims compared: %s -- if laya/agent.py stops carrying a sentence that lists an "
               "accepted dtype, the prose site this sweep was written for moved" % sorted(claim_sites))

    # ------------------------------------------------- what a documented default says
    derived, unchecked, conflicts = code_defaults()
    check("defaults/no name is read with two different literal defaults", conflicts, {})
    surfaces = {"laya/serve.py docstring": docstring_default_table(),
                "docs/http-api.md": markdown_default_table("docs/http-api.md")}
    for name, value in sorted(derived.items()):
        for surface in PACKAGE_DEFAULT_SURFACES:
            table = surfaces[surface]
            if name not in table:
                continue
            check("defaults/%s's %s cell says what the code does" % (surface, name),
                  default_cell(table[name]), default_text(value))
    # Non-vacuity by name. Without these the whole section could be standing on nothing: a table
    # that stopped parsing returns {}, every `continue` above fires, and the file is green.
    check_true("defaults/the drifted cell is derived from Router, not asserted from a copy",
               derived.get("LAYA_DEFAULT_MODEL") == "multilingual",
               "got %r -- if the routing fallback moved again, the two pages this was written for "
               "moved with it or are now stale; check them before changing this line"
               % derived.get("LAYA_DEFAULT_MODEL"))
    counts = {s: len(surfaces[s]) for s in PACKAGE_DEFAULT_SURFACES}
    check_true("defaults/both package surfaces still carry a default column",
               all(count >= 12 and "LAYA_PRELOAD" in surfaces[s]
                   for s, count in counts.items()), counts)
    check_true("defaults/LAYA_DEFAULT_MODEL is a row on both package surfaces",
               all("LAYA_DEFAULT_MODEL" in surfaces[s] for s in PACKAGE_DEFAULT_SURFACES),
               {s: sorted(surfaces[s]) for s in PACKAGE_DEFAULT_SURFACES})
    check_true("defaults/at least eight defaults were derived, not skipped",
               len(derived) >= 8, sorted(derived))
    documented = {name for surface in PACKAGE_DEFAULT_SURFACES for name in surfaces[surface]}
    for name in sorted(documented - set(derived)):
        # A name the server never reads has no unset path here to follow, so it joins the reported
        # set rather than passing quietly -- but only if nothing more specific already explains it,
        # which is why this cannot turn a failed derivation into a silent skip.
        unchecked.setdefault(name, "laya/serve.py does not read it, so this file has no unset "
                                   "path to follow for it")
    unaccounted = sorted(name for surface in PACKAGE_DEFAULT_SURFACES
                         for name in surfaces[surface]
                         if name not in derived and name not in unchecked)
    check_true("defaults/every documented default is derived or named as unchecked",
               not unaccounted, unaccounted)

    # ------------------------------------------------- what this cannot check
    unbacked = [
        ("a documented default matches the code default",
         "held to laya/serve.py and laya/router.py for %d of the names on the two package surfaces; "
         "%s stay out because their unset answer is not a value a table can print. Also out: "
         "docs/docker.md, which documents the Compose service's defaults and deliberately differs "
         "from the package (LAYA_PRELOAD is 0 there and 1 in laya/serve.py), and docs/cli-mcp.md, "
         "which documents the stdio server's"
         % (len(derived), sorted(set(surfaces["docs/http-api.md"]) - set(derived)))),
        ("a name only mentioned in a comment",
         "comments are not string constants, so an undocumented read in a comment is invisible "
         "here; the same is true of a name built by concatenation, and of an accepted spelling "
         "that only a comment lists"),
        ("the values a variable accepts",
         "held to laya/agent.py for LAYA_CUDA_AMP and LAYA_CPU_AMP, the only names whose values "
         "are a literal tuple; a value validated by type or by _env_bool()'s shared truthy set "
         "has no per-name vocabulary for prose to be compared against"),
    ]
    check_true("uncheckable claims are reported, not skipped", len(unbacked) == 3, unbacked)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    if not FAIL:
        print("all %d LAYA_* names the package reads appear in the docs" % len(pkg))
        print("all %d documented names are consumed somewhere in the tree" % len(docs))
        for var in AMP_VARS:
            print("laya/agent.py compares %s against %s" % (var, sorted(accepted[var])))
        print("%d row(s) and %d completeness claim(s) held to that set: %s / %s"
              % (len(rows_sites), len(claim_sites), sorted(rows_sites), sorted(claim_sites)))
        print("%d documented default(s) held to laya/serve.py and laya/router.py" % len(derived))
        for name, why in sorted(unchecked.items()):
            print("not derived: %s -- %s" % (name, why))
        for name, why in sorted(DEFERRED.items()):
            print("deferred: %s -- %s; delete the DEFERRED row once it is documented" % (name, why))
        print("not checkable from this repository (not asserted either way):")
        for name, why in unbacked:
            print("  - %s: %s" % (name, why))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
