"""Every ``LAYA_*`` the package reads has to be written down where a user can find it.

Two names drifted out of reach of the documentation and nothing noticed:

  * ``LAYA_MAX_CONCURRENT`` appears only inside ``laya/serve.py``'s module docstring.
    That table is the developer-facing copy; no page renders it, and the server table in
    ``docs/docker.md`` lists nine variables and not this one -- so the admission-control
    knob behind a ``503`` is invisible to the deployment that hits it.
  * ``LAYA_MPS_AMP_MIN_ROWS`` is named nowhere at all, one sentence away from its
    documented ``LAYA_CUDA_AMP`` / ``LAYA_CPU_AMP`` siblings, even though it changes which
    forwards run in fp16 on Apple Silicon and therefore which probabilities come back.

A review note fixes both once. The names live in three places -- the package, the
markdown, and the compose files -- and a list written down in any one of them goes stale
the moment an ``os.environ`` read is added in another, so this derives the set instead of
transcribing it: it walks ``laya/`` with ``ast`` and compares what it finds against what
the docs mention, in both directions.

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
DEFERRED = {
    "LAYA_SHA256_DIGESTS": "#528 (docs(security)) documents digest verification",
}


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
                    found.setdefault(name, set()).add(os.path.relpath(path, ROOT))
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
            found.setdefault(name, set()).add(os.path.relpath(path, ROOT))
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
                found.setdefault(name, set()).add(os.path.relpath(path, ROOT))
    return found


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

    # ------------------------------------------------- what this cannot check
    unbacked = [
        ("a documented default matches the code default",
         "docs/docker.md documents the Compose service's default, which is deliberately not "
         "the package default (LAYA_PRELOAD is 0 there and 1 in laya/serve.py)"),
        ("a name only mentioned in a comment",
         "comments are not string constants, so an undocumented read in a comment is invisible "
         "here; the same is true of a name built by concatenation"),
        ("the values a variable accepts",
         "LAYA_CUDA_AMP=fp32 is silently ignored rather than rejected; checking the accepted "
         "set against prose is a different sweep"),
    ]
    check_true("uncheckable claims are reported, not skipped", len(unbacked) == 3, unbacked)

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL " + f)
    if not FAIL:
        print("all %d LAYA_* names the package reads appear in the docs" % len(pkg))
        print("all %d documented names are consumed somewhere in the tree" % len(docs))
        for name, why in sorted(DEFERRED.items()):
            print("deferred: %s -- %s; delete the DEFERRED row once it is documented" % (name, why))
        print("not checkable from this repository (not asserted either way):")
        for name, why in unbacked:
            print("  - %s: %s" % (name, why))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
