"""The CVE audit job must see every name the package declares, not just the core five.

`.github/workflows/security.yml` builds its audit lists from `pyproject.toml` and hands them to
`pip-audit`. For a while it did that with a regex anchored on `dependencies = [`, which matches the
`[project]` array and none of `[project.optional-dependencies]` -- so the ten names behind the
`serve`, `fast`, `mcp`, `structured`, `onnx`, `langchain` and `langgraph` extras, and the 57
transitive names they pull in, were outside the audit. `Dockerfile` ends with
`pip install ".[serve]" && pip check`: the HTTP server's dependencies ship in the published image.

Since #645/#646 the audit is split in two. The blocking scope is what ships: the core plus the
`[serve]` extra, audited strictly, where a finding fails the job. The remaining optional
integrations are audited in a second, advisory step: findings are reported as an annotation but do
not fail unrelated PRs, while a failure of the audit itself -- a specifier pip-audit cannot
resolve, a resolver or tool crash -- still does.

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11, same as
`tests/test_packaging.py`. The job that runs the extractor pins `python-version: "3.11"`, which is
checked here, and the lifted code is only executed where tomllib exists -- on 3.10 that check
degrades to a note and never to a pass.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, NOTES = [], [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        # %s, not concatenation: several details below are lists of names, and a gate that raises
        # on the way to reporting a failure leaves the mutant's name unsaid.
        FAIL.append("%s: %s" % (name, detail))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def section(text, header):
    """Lines of one top-level-`  ` job/table, from `header` to the next same-indent key."""
    lines = text.splitlines()
    depth = len(header) - len(header.lstrip())
    out, seen = [], False
    for line in lines:
        if seen and (line.strip() and (len(line) - len(line.lstrip())) <= depth
                     and not line.lstrip().startswith("#")):
            break
        if line.rstrip() == header.rstrip():
            seen = True
        if seen:
            out.append(line)
    return out


# ---------------------------------------------------------------- the declared sets
pyproject = read("pyproject.toml")


def array(body):
    return re.findall(r'"([^"]+)"', body)


core_block = re.search(r"^dependencies = \[(.*?)\]", pyproject, re.S | re.M)
core = array(core_block.group(1)) if core_block else []

# Every `name = [ ... ]` under [project.optional-dependencies], before the next table header.
od = pyproject.split("[project.optional-dependencies]", 1)
extras = {}
if len(od) > 1:
    table = od[1].split("\n[", 1)[0]
    for m in re.finditer(r"^([A-Za-z0-9_-]+) = \[(.*?)\]", table, re.S | re.M):
        extras[m.group(1)] = array(m.group(2))

check_true("pyproject/core parses", len(core) == 5, core)
check("pyproject/extra tables found", sorted(extras),
      ["crewai", "fast", "langchain", "langgraph", "llamaindex", "mcp", "onnx", "serve",
       "structured"])
declared = list(core) + [s for names in extras.values() for s in names]
expected = sorted(set(declared))
check("pyproject/extras add names the core does not have",
      sorted({re.split(r"[<>=!;\[ ]", s)[0] for s in declared}
             - {re.split(r"[<>=!;\[ ]", s)[0] for s in core}),
      ["crewai", "fastapi", "langchain-core", "langgraph", "llama-index-core", "mcp", "onnx",
       "onnxruntime", "onnxscript", "pydantic", "python-multipart", "tilelang", "uvicorn"])

# The two scopes: shipped (blocking) is the core plus [serve]; everything else is advisory.
shipped = sorted(set(core) | set(extras.get("serve", [])))
advisory = sorted(set(declared) - set(shipped))
check_true("pyproject/scope split covers every declared name",
           sorted(set(shipped) | set(advisory)) == expected and not set(shipped) & set(advisory),
           "shipped and advisory must partition the declared specs")

# The blocking scope rests on one declaration-level fact: the image installs the [serve] extra
# and no other (#694 review). If the Dockerfile ever reads `pip install ".[serve,onnx]"`, onnx
# would ship in the image while sitting in the advisory bucket, and nothing would go red.
dockerfile = read("Dockerfile")
image_extras = sorted({e.strip()
                       for group in re.findall(r"pip install[^\n]*?\.\[([^\]]*)\]", dockerfile)
                       for e in group.split(",") if e.strip()})
check("Dockerfile/the image installs only the [serve] extra", image_extras, ["serve"])

# ---------------------------------------------------------------- the job's own extractors
deps_job = "\n".join(section(read(os.path.join(".github", "workflows", "security.yml")),
                             "  deps:"))
check_true("security.yml/has a deps job", "pip-audit" in deps_job, deps_job[:80])


def lift_extractor(filename):
    """The heredoc that writes one audit list, dedented so it can run."""
    m = re.search(r"python - <<'PY' > %s\n(.*?)\n\s*PY\n" % re.escape(filename), deps_job, re.S)
    if not m:
        return ""
    body = m.group(1)
    indents = [len(l) - len(l.lstrip()) for l in body.splitlines() if l.strip()]
    return "\n".join(l[min(indents):] if l.strip() else "" for l in body.splitlines())


check_true("security.yml/exactly two heredocs write the audit lists",
           len(re.findall(r"<<'PY'", deps_job)) == 2
           and "> requirements-audit-core.txt" in deps_job
           and "> requirements-audit-extras.txt" in deps_job,
           "one extractor per scope; any other count means a scope silently went unaudited")
core_extractor = lift_extractor("requirements-audit-core.txt")
extras_extractor = lift_extractor("requirements-audit-extras.txt")

# Why the extractors are read out of the workflow instead of re-implemented here: the failure mode
# this file exists for is the two surfaces drifting, so a check that carries its own copy of the
# parsing rules would drift in exactly the same direction and stay green.
for name, extractor in (("core", core_extractor), ("extras", extras_extractor)):
    check_true("extractor/%s reads the tables with tomllib, not a regex on one array" % name,
               "tomllib" in extractor and "dependencies = \\[" not in extractor, extractor[:200])
    check_true("extractor/%s walks optional-dependencies" % name,
               "optional-dependencies" in extractor, extractor[:200])
check_true("extractor/keeps the declared-not-installed reason",
           "local version" in deps_job and "+cpu" in deps_job,
           "auditing an install would trip --strict on torch's 2.14.0+cpu")
check_true("extractor/audits both files strictly",
           re.search(r"pip-audit --strict[^\n]*-r requirements-audit-core\.txt", deps_job)
           and re.search(r"pip-audit --strict[^\n]*-r requirements-audit-extras\.txt", deps_job),
           "advisorial applies to findings, not to how the audit runs")
check_true("extractor/prints the scope it audited",
           re.search(r'echo "[^"]*requirements-audit-core\.txt', deps_job) is not None
           and re.search(r'echo "[^"]*requirements-audit-extras\.txt', deps_job) is not None,
           "when a scope goes red the log has to say how many names were in it")

pin = re.search(r"python-version:\s*[\"'](\d+)\.(\d+)[\"']", deps_job)
check_true("security.yml/the deps job pins python >= 3.11",
           pin is not None and tuple(map(int, pin.groups())) >= (3, 11),
           "tomllib, which the extractors read the tables with, is 3.11+")

# Findings advisory, failures blocking: no blanket continue-on-error, and the step must tell a
# findings report (its "known vulnerabilities" count) apart from the audit itself breaking.
check_true("security.yml/extras findings are advisory without swallowing tooling errors",
           "continue-on-error" not in deps_job
           and "known vulnerabilities" in deps_job
           and "::warning::" in deps_job
           and "::error::" in deps_job,
           "#646: a findings report warns, a broken audit (no report) must still fail the job")
check_true("security.yml/the advisory annotation names the advisories themselves",
           re.search(r"ids=\$\(grep[^\n]*(GHSA|PYSEC)[^\n]*pip-audit-extras\.log", deps_job) is not None
           and "${ids" in deps_job,
           "a triager reads WHAT was found from the annotation, not by opening the log (#694 review)")

ran = False
if sys.version_info >= (3, 11) and core_extractor and extras_extractor:
    ran = True

    def run_extractor(source):
        return subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                              capture_output=True, text=True)

    core_run = run_extractor(core_extractor)
    check_true("extractor/core runs against the real pyproject.toml",
               core_run.returncode == 0, core_run.stderr.strip()[:300])
    extras_run = run_extractor(extras_extractor)
    check_true("extractor/extras runs against the real pyproject.toml",
               extras_run.returncode == 0, extras_run.stderr.strip()[:300])

    core_emitted = [l.strip() for l in core_run.stdout.splitlines() if l.strip()]
    extras_emitted = [l.strip() for l in extras_run.stdout.splitlines() if l.strip()]
    check("extractor/core emits the shipped names, [serve] included", sorted(set(core_emitted)),
          shipped)
    check("extractor/extras emits every remaining declared name", sorted(set(extras_emitted)),
          advisory)
    # Specifiers carried through, not stripped: `fastapi` alone would audit whatever PyPI serves
    # today and hide the floor the package actually supports.
    check_true("extractor/preserves each version specifier",
               all(e in declared for e in core_emitted + extras_emitted),
               [e for e in core_emitted + extras_emitted if e not in declared])
    check_true("extractor/collapses the langchain/langgraph duplicate",
               len(core_emitted) == len(set(core_emitted))
               and len(extras_emitted) == len(set(extras_emitted)),
               [s for s in core_emitted + extras_emitted
                if (core_emitted + extras_emitted).count(s) > 1])
    check_true("extractor/the scopes do not double-audit a name",
               not set(core_emitted) & set(extras_emitted),
               sorted(set(core_emitted) & set(extras_emitted)))
    for name in ("fastapi", "uvicorn", "python-multipart"):
        check_true("extractor/audits %s in the blocking scope, which the image installs" % name,
                   any(re.split(r"[<>=!;\[ ]", s)[0] == name for s in core_emitted),
                   "Dockerfile installs it through the `serve` extra")
else:
    NOTES.append("python %d.%d has no tomllib: the extractors were not executed here, only their "
                 "source was checked (the job pins 3.11+)" % sys.version_info[:2])
# Where the interpreter can run the extractors, it must have. Without this the skip above is a
# hole a mutant can open -- `if False` would leave the file green with checks missing.
check_true("extractor/ran wherever tomllib exists",
           ran or sys.version_info < (3, 11), "the executable checks were skipped on a 3.11+ run")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for note in NOTES:
    print("  NOTE " + note)
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
