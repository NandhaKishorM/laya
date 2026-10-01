"""Packaging metadata must match what the dependencies actually need (#34).

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11.
"""
import ast
import fnmatch
import os
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def version_tuple(text):
    return tuple(int(part) for part in text.split("."))


pyproject = read("pyproject.toml")
setup_py = read("setup.py")

requires_python = re.search(r'requires-python\s*=\s*"[>=~^]*\s*([\d.]+)"', pyproject)
check_true("pyproject/declares requires-python", requires_python is not None)
floor = version_tuple(requires_python.group(1)) if requires_python else (0, 0)

classifier_versions = [
    version_tuple(v)
    for v in re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', pyproject)
]
check_true("pyproject/advertises specific Python versions", len(classifier_versions) > 0)
below_floor = [".".join(str(p) for p in v) for v in classifier_versions if v < floor]
check("classifiers/none below requires-python", below_floor, [])

# Checkpoints run on answerdotai/ModernBERT-large, which transformers only knows from 4.48.
transformers_floor = re.search(r'"transformers>=([\d.]+)"', pyproject)
check_true("pyproject/pins a transformers floor", transformers_floor is not None)
check_true(
    "transformers/floor covers ModernBERT",
    transformers_floor is not None and version_tuple(transformers_floor.group(1)) >= (4, 48),
    "ModernBERT support starts in transformers 4.48",
)

for field in ("python_requires", "install_requires", "classifiers"):
    check_true(
        "setup.py/does not duplicate %s" % field,
        field not in setup_py,
        "metadata belongs in pyproject.toml only",
    )

# The release job checks the git tag against pyproject, but laya.__version__ is what the server
# and SDK report at runtime, so the two strings must not drift apart.
static_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
check_true("pyproject/declares a static version", static_version is not None)
init_version = re.search(r'^__version__\s*=\s*"([^"]+)"', read(os.path.join("laya", "__init__.py")), re.M)
check_true("laya/__init__ declares a literal __version__", init_version is not None)
check(
    "version/pyproject matches laya.__version__",
    static_version.group(1) if static_version else None,
    init_version.group(1) if init_version else None,
)
for label, match in (("pyproject", static_version), ("laya/__init__", init_version)):
    value = match.group(1) if match else ""
    parts = value.split(".")
    check_true("%s/version is X.Y.Z" % label, len(parts) == 3 and all(p.isdigit() for p in parts), "got %r" % value)

workflow = read(os.path.join(".github", "workflows", "ci.yml"))
ci_versions = [version_tuple(v) for v in re.findall(r'"(\d+\.\d+)"', workflow)]
stale = [".".join(str(p) for p in v) for v in ci_versions if v < floor]
check("ci/tests no Python below requires-python", stale, [])

# Classifiers on PyPI are a support claim. If a version is listed there, CI must run it
# (3.12 was advertised while the matrix jumped 3.11 -> 3.13).
missing_from_ci = [
    ".".join(str(p) for p in v) for v in classifier_versions if v not in ci_versions
]
check("ci/tests every advertised Python version", missing_from_ci, [])

# Every test in tests/ must be wired into CI workflows (ci.yml or docker.yml),
# unless explicitly exempted with a documented rationale (#399).
def _clean_command_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    try:
        return " ".join(shlex.split(line, comments=True))
    except ValueError:
        return re.sub(r"(?:\s+|^)#.*$", "", line).strip()


def _invoked_workflow_tests(yaml_text):
    """Extract test files that are actually executed by python or pytest in run: steps."""
    run_lines = []
    in_run = False
    run_indent = 0
    for line in yaml_text.splitlines():
        indent = len(line) - len(line.lstrip())
        match = re.match(r"^(\s*)-\s+run:\s*(\|?>?)(.*)$", line) or re.match(r"^(\s*)run:\s*(\|?>?)(.*)$", line)
        if match:
            in_run = True
            run_indent = indent
            inline_cmd = match.group(3).strip()
            cleaned = _clean_command_line(inline_cmd)
            if cleaned:
                run_lines.append(cleaned)
            continue
        if in_run:
            if line.strip() and indent <= run_indent:
                in_run = False
            else:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    cleaned = _clean_command_line(stripped)
                    if cleaned:
                        if stripped.endswith("\\") and not cleaned.endswith("\\"):
                            cleaned += " \\"
                        run_lines.append(cleaned)

    # Merge backslash continuation lines (e.g. multi-line pytest argument lists)
    merged_lines = []
    buf = []
    for line in run_lines:
        if line.endswith("\\"):
            buf.append(line[:-1].strip())
        else:
            if buf:
                buf.append(line)
                merged_lines.append(" ".join(buf))
                buf = []
            else:
                merged_lines.append(line)
    if buf:
        merged_lines.append(" ".join(buf))

    # Match executable python or pytest invocations, ignoring mentions in echo/cat/test
    invoked = set()
    for cmd in merged_lines:
        for part in re.split(r";|&&|\|\||\|", cmd):
            part = part.strip()
            if re.search(r"\b(?:pytest|python(?:\d+(?:\.\d+)?)?\s+-m\s+pytest)\b", part):
                invoked.update(re.findall(r"\btests/(test_[a-zA-Z0-9_]+\.py)\b", part))
            elif re.search(r"\bpython(?:\d+(?:\.\d+)?)?\s+.*?tests/(test_[a-zA-Z0-9_]+\.py)\b", part):
                invoked.update(re.findall(r"\btests/(test_[a-zA-Z0-9_]+\.py)\b", part))
            else:
                # `unittest discover` runs a whole glob rather than naming files, so expand it
                # against the tree. A lane that discovers `test_zh_*.py` has wired every suite
                # the glob matches, and naming them again in the workflow would only drift.
                found = re.search(r"\bunittest\s+discover\b.*?-s\s+tests\b"
                                  r".*?-p\s+'?\"?([^'\" ]+)", part)
                if found:
                    invoked.update(fnmatch.filter(os.listdir(os.path.join(ROOT, "tests")),
                                                  found.group(1)))
    return invoked


docker_workflow = read(os.path.join(".github", "workflows", "docker.yml"))
registered_test_files = _invoked_workflow_tests(workflow) | _invoked_workflow_tests(docker_workflow)

EXEMPT_TEST_SUITES = {
    "test_local_e2e.py": "Requires local checkpoints under ~/laya_models (AGENTS.md)",
    "test_mcp_local_e2e.py": "Requires local checkpoints under ~/laya_models (AGENTS.md)",
    "test_onnx.py": "Requires onnx extra; skip-guarded on lane without it (AGENTS.md)",
    "test_fast.py": "Requires CUDA and tilelang extra (AGENTS.md)",
    "test_server_example.py": "Requires cached or downloaded weights for examples/server.py",
}

all_test_files = [
    f for f in os.listdir(os.path.join(ROOT, "tests"))
    if f.startswith("test_") and f.endswith(".py")
]
untested_suites = [
    f for f in sorted(all_test_files)
    if f not in EXEMPT_TEST_SUITES and f not in registered_test_files
]
check("ci/wires every non-exempt test suite", untested_suites, [])


# --------------------------------------------------------------- markdown links
# Nothing checked these, and the README ships to PyPI and to the model card. Only
# targets inside the repository are checked: external URLs would make the suite depend
# on the network, which it must not. The README's Router link pointed at a heading that
# had been renamed, so it silently went nowhere for as long as the rename was in.
def _slug(heading):
    """GitHub's heading anchor: drop anything that is not word/space/hyphen, lowercase,
    then spaces to hyphens."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def _headings(path):
    found = set()
    for line in read(path).splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if match:
            found.add(_slug(match.group(1)))
    return found


_md = []
for _dirpath, _dirnames, _filenames in os.walk("."):
    # `.pytest_cache` ships a README of its own and `.venv` is where CONTRIBUTING tells
    # contributors to install; neither is part of the repository.
    _dirnames[:] = [d for d in _dirnames
                    if d not in (".git", "__pycache__", "node_modules", ".pytest_cache", ".venv")]
    _md.extend(os.path.normpath(os.path.join(_dirpath, f))
               for f in _filenames if f.endswith(".md"))
_md = sorted(_md)
_heading_cache = {p: _headings(p) for p in _md}

check_true("md/at least the README, BENCHMARKS and docs are scanned", len(_md) >= 8, _md)
check_true("md/no build directory scanned",
           not any(".pytest_cache" in p for p in _md), _md)

_broken_files, _broken_anchors = [], []
for _path in _md:
    for _label, _target in re.findall(r"\[([^\]]*)\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)",
                                      read(_path)):
        if _target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        _tpath, _, _frag = _target.partition("#")
        _dest = os.path.normpath(os.path.join(os.path.dirname(_path), _tpath)) if _tpath else _path
        if _tpath and not os.path.exists(_dest):
            _broken_files.append("%s: [%s](%s)" % (_path, _label[:30], _target))
            continue
        if _frag and _dest.endswith(".md") and _frag not in _heading_cache.get(_dest, set()):
            _broken_anchors.append("%s: [%s](%s)" % (_path, _label[:30], _target))

check("md/no link to a file that does not exist", _broken_files, [])
check("md/no anchor that matches no heading", _broken_anchors, [])
# ---------------------------------------------------------------- Compose layout
# `compose.http.yaml` is an override, so it is merged onto `compose.yaml` rather than
# read on its own. These checks are textual because the suite takes no third-party
# dependency and PyYAML is not one; the real validation is `docker compose config`,
# which the Docker workflow runs for every file combination.
http = read("compose.http.yaml")
base = read("compose.yaml")
cuda = read("compose.cuda.yaml")
dockerfile = read("Dockerfile")

check_true("compose.http/declares laya-serve", "laya-serve:" in http, http[:200])
check_true("compose.http/runs the server command",
           'command: ["laya-serve"]' in http, "laya-serve is not the container command")
check_true("compose.http/publishes a port", re.search(r"^\s*ports:", http, re.M) is not None)
# Host and container port must come from the same variable, or they drift apart and the
# published port stops reaching the server.
port_map = re.search(r'-\s*"(?:\$\{LAYA_BIND_ADDRESS:-[^}]+\}:)?(\$\{[A-Z_]+:-(\d+)\}):(\$\{[A-Z_]+:-(\d+)\})"', http)
# The API is unauthenticated until LAYA_API_KEY is set, so exposure beyond the host is opt-in.
check_true("compose.http/publishes on loopback unless LAYA_BIND_ADDRESS is set",
           '"${LAYA_BIND_ADDRESS:-127.0.0.1}:' in http)
check_true("compose.http/has a healthcheck on /health",
           "healthcheck:" in http and "/health" in http)
check_true("compose.http/port mapping is present", port_map is not None, http[:300])
if port_map:
    check("compose.http/host port equals container port", port_map.group(2), port_map.group(4))
    check("compose.http/both sides use the same variable", port_map.group(1), port_map.group(3))
check_true("compose.http/the server reads the same variable",
           'LAYA_PORT: "${LAYA_PORT:-8000}"' in http)
# The block enumerates what it forwards, so a variable the server reads and the file omits is
# unreachable for the documented compose path -- `docker run -e` still works, compose does not.
check_true("compose.http/forwards the resident-checkpoint cap",
           'LAYA_MAX_LOADED: "${LAYA_MAX_LOADED:-}"' in http)
check_true("compose.http/forwards the FastAPI root path",
           'LAYA_ROOT_PATH: "${LAYA_ROOT_PATH:-}"' in http)
check_true("compose.http/shares the model cache",
           "model-cache:/home/laya/.cache/huggingface" in http)
# The base service is what `docker compose run --rm laya` uses; publishing it a port or
# changing its command would be a breaking change to the quickstart.
check_true("compose.http/leaves the quickstart service alone",
           "laya:" not in http, "compose.http.yaml overrides the base `laya` service")

# `pip install .` alone puts no `laya-serve` in the image, so the extra is load-bearing.
check_true("Dockerfile/installs the serve extra", '".[serve]"' in dockerfile, dockerfile[:400])
check_true("Dockerfile/still runs pip check", "pip check" in dockerfile)

# torch 2.14's eager Triton kernels compile on the first CUDA inference and need a C compiler the
# slim runtime image does not have (#365). The kill switch keeps the stock kernels.
check_true("Dockerfile/runtime stage disables torch's native Triton JIT (#365)",
           re.search(r"^\s*TORCH_DISABLE_NATIVE_JIT=1", dockerfile.partition("AS runtime")[2], re.M) is not None,
           "without TORCH_DISABLE_NATIVE_JIT=1 a GPU image serves 500s while /health stays green")

# Overrides for `laya` never reach `laya-serve`, a separate service. If the CUDA override does
# not repeat the args for laya-serve, that service silently serves on CPU.
check_true("compose.cuda/covers laya-serve too",
           re.search(r"^\s{2}laya-serve:", cuda, re.M) is not None,
           "compose.cuda.yaml does not mention laya-serve, so GPU serving would be CPU")
check("compose.cuda/repeats the torch index for the base service",
      len(re.findall(r'TORCH_INDEX: "\$\{LAYA_TORCH_INDEX:-cu128\}"', cuda)), 2)
check("compose.cuda/repeats the device reservation for both services",
      len(re.findall(r"driver: nvidia", cuda)), 2)
check_true("compose.cuda/no stale reference to a missing file",
           "compose.http.yaml on the HTTP preview branch" not in cuda,
           "compose.cuda.yaml still describes compose.http.yaml as living on another branch")

# Every file the Docker workflow validates must exist.
for name in ("compose.yaml", "compose.example.yml", "compose.cuda.yaml", "compose.http.yaml",
             "compose.spark.yaml", "compose.modelscope.yaml"):
    check_true("compose/%s exists" % name, os.path.exists(name))

# The ModelScope bake is a build argument, so a deployment that wants it has to carry the
# arguments into *both* services -- `laya-serve` is its own service and an override for `laya`
# never reaches it, which is the same trap compose.cuda.yaml documents. And the image has to stay
# offline: a baked snapshot is keyed by a ModelScope commit the Hub cannot confirm, so going online
# downloads the same weights again instead of serving the baked copy.
ms = read("compose.modelscope.yaml")
check_true("compose.modelscope/covers laya-serve too",
           re.search(r"^\s{2}laya-serve:", ms, re.M) is not None,
           "compose.modelscope.yaml does not mention laya-serve, so served checkpoints come from "
           "the Hub and a host without Hugging Face access cannot serve at all")
# One argument selects the checkpoint, and it defaults to the multilingual one, the checkpoint
# every Laya caller routes to without being asked.
check("compose.modelscope/repeats the prefetch args for both services",
      len(re.findall(r'MODELSCOPE_MODEL: "\$\{MODELSCOPE_MODEL:-multilingual\}"', ms)), 2)
check("compose.modelscope/repeats the revision for both services",
      len(re.findall(r'MODELSCOPE_REVISION: "\$\{MODELSCOPE_REVISION:-master\}"', ms)), 2)
check("compose.modelscope/keeps every service on the baked cache",
      len(re.findall(r'HF_HUB_OFFLINE: "\$\{HF_HUB_OFFLINE:-1\}"', ms)), 2)
check_true("compose.modelscope/preloads only what can load offline",
           'LAYA_MODELS: "${LAYA_MODELS:-multilingual}"' in ms,
           "LAYA_PRELOAD=1 with the family-wide default would fail on the first checkpoint that "
           "was not baked")

# The default build must stay exactly what it was: no prefetch, so a plain
# `docker build .` hits the Hub as before, and the RUN is a no-op for the empty argument.
dockerfile_runtime = dockerfile.partition("AS runtime")[2]
check_true("Dockerfile/prefetch args default to off",
           re.search(r'^ARG MODELSCOPE_MODEL=""', dockerfile_runtime, re.M) is not None,
           "a non-empty default would change every existing build")
check_true("Dockerfile/copies the prefetch script",
           "docker/prefetch_modelscope.py" in dockerfile, "the RUN below references a missing file")
check_true("Dockerfile/prefetch step is conditional",
           re.search(r'^\s*RUN if \[ -n "\$MODELSCOPE_MODEL" \]; then', dockerfile_runtime, re.M) is not None,
           "an unconditional RUN would make every build depend on modelscope.cn")
check_true("Dockerfile/prefetch hands the cache to the runtime user",
           "chown -R laya:laya /home/laya/.cache" in dockerfile_runtime,
           "the tokenizer-compatibility fix writes into the snapshot on first load, and the image "
           "runs as UID 10001")
check_true("docker/prefetch_modelscope.py exists", os.path.exists("docker/prefetch_modelscope.py"))


# --------------------------------------------------------------- nix: the deployment layer
# Nix builds and evaluates nothing here in CI (`grep -rn nix .github/workflows/` is empty), so
# these textual checks are the only gate on the two files a NixOS host deploys from. Same style
# as the compose checks above: no nix binary, no third-party dependency.

nix_pkg = read(os.path.join("nix", "package.nix"))
nix_version = re.search(r'^\s*version\s*=\s*"([^"]+)"', nix_pkg, re.M)
check_true("nix/package.nix declares a version", nix_version is not None, nix_pkg[:200])
# The release job compares the tag to pyproject, and pyproject is compared to laya.__version__
# above. This is the third declaration of the same string -- it sat at 0.3.4 for sixteen
# releases because no reader existed.
check("version/nix matches pyproject",
      nix_version.group(1) if nix_version else None,
      static_version.group(1) if static_version else None)

# `services.laya-serve.models` is joined into LAYA_MODELS, which laya.serve splits and hands to
# Router.preload() -- and preload normalises every name (laya/router.py:370), so the server takes
# core's aliases. A closed `enum` in the module can only copy that list and fall behind it: it
# refused ten of the thirteen spellings the same value accepts, and it refused them at nix
# evaluation time, on the way to starting a service that would have been happy.
nix_module = read(os.path.join("nix", "laya-serve.nix"))
models_block = re.search(r"models = lib\.mkOption \{(.*?)\n    \};", nix_module, re.S)
check_true("nix/module has a models option", models_block is not None, nix_module[:200])
block = models_block.group(1) if models_block else ""
type_line = re.search(r"^\s*type\s*=\s*(.+?)\s*$", block, re.M)
check_true("nix/models type declaration found", type_line is not None, block[:200])
declared_type = type_line.group(1) if type_line else ""
check_true("nix/models declares no closed enum over checkpoint names",
           "types.enum" not in declared_type, "type = %s" % declared_type)

# The accepted name set, read out of core rather than transcribed: DEFAULT_MODELS' keys and the
# alias table. ast, not import -- this suite takes no third-party dependency, and torch is one.
def dict_keys(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
    # Returning nothing rather than raising keeps the failure a named check: the population
    # guard below reports it. A gate that dies on the way to reporting leaves the cause unsaid.
    return []


router_src = read(os.path.join("laya", "router.py"))
canonical = dict_keys(router_src, "DEFAULT_MODELS")
aliases = dict_keys(router_src, "_ALIASES")
accepted = sorted(set(canonical) | set(aliases))
# Non-vacuity: the derivation has to have found both tables, or every comparison against
# `accepted` below would pass by matching an empty set.
check_true("nix/core name tables are both populated",
           len(canonical) >= 1 and len(aliases) >= 1,
           "DEFAULT_MODELS=%r _ALIASES=%r -- laya/router.py changed shape" % (canonical, aliases))
check("nix/models default is exactly core's canonical checkpoints",
      sorted(n.strip('"') for n in
             re.findall(r'default = \[([^\]]*)\]', block, re.S)[0].split())
      if re.search(r"default = \[", block) else None,
      sorted(canonical))

# Whatever the module tells its reader to type must be what core accepts, and the module runs no
# validator of its own -- so a name added to _ALIASES has to appear here or the option's own
# description goes stale on the day the alias lands.
description = re.search(r"description = ''(.*?)''", block, re.S)
check_true("nix/models has a description", description is not None, block[:200])
described = description.group(1) if description else ""
undocumented = [n for n in accepted if "`%s`" % n not in described]
check_true("nix/models describes every name core accepts", not undocumented,
           "accepted by laya.router.normalise_name but absent from the option text: %s"
           % undocumented)
# And the module must still be the thing that sets LAYA_MODELS, or the checks above describe a
# wire that no longer exists.
check_true("nix/module still joins models into LAYA_MODELS",
           re.search(r'LAYA_MODELS = lib\.concatStringsSep "," cfg\.models;', nix_module) is not None,
           "the models option no longer feeds LAYA_MODELS; these checks need retargeting")

# ---- every knob laya reads from the environment has to be reachable from the module
# `laya.serve` has no config file and no CLI flag for any of it: the process configures itself
# from LAYA_* and nothing else. On a NixOS host the unit's environment is the only thing that can
# hand those variables over, so a name the module never assigns is a control that host cannot ask
# for.
#
# This set is derived from the whole package, not from `laya/serve.py`. Reading serve.py alone is
# a scope error that passed: the three runtime knobs below were invisible to it. A deployment unit
# configures a *process*, and the process is `laya`.
#
# Both regexes carry `[A-Z0-9_]` for the same reason: `LAYA_SHA256_DIGESTS`. `[A-Z_]+` matches a
# prefix of that name, so a narrower pattern reports no gap rather than the one it cannot see.
# `_ENV_KEY = "LAYA_DEVICE"` is how laya/mcp/device.py names the one variable it reads, and since
# #574 laya.serve reads the device through it, so the constant counts as a read.
_READ_PATTERNS = (r'environ\.get\("(LAYA_[A-Z0-9_]+)"', r'_env_bool\("(LAYA_[A-Z0-9_]+)"',
                  r'environ\["(LAYA_[A-Z0-9_]+)"\]', r'_ENV_KEY = "(LAYA_[A-Z0-9_]+)"')


def env_reads():
    """Every `LAYA_*` name the package looks up, by walking laya/ rather than listing files."""
    found = set()
    for root, dirs, files in os.walk(os.path.join(ROOT, "laya")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if not name.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, name), ROOT)
            src = read(rel)
            for pat in _READ_PATTERNS:
                found.update(re.findall(pat, src))
    return found


read_names = env_reads()
# An assignment only. The `models` description names `LAYA_MODELS` in prose, and prose that
# mentions a variable sets nothing.
assigned = sorted(set(re.findall(r'\b(LAYA_[A-Z0-9_]+)\s*=', nix_module))
                  | set(re.findall(r'export (LAYA_[A-Z0-9_]+)=', nix_module)))

# Non-vacuity, twice over: an empty derivation passes both directions below for free, and a
# digit-blind pattern reproduces the original miss while still looking like ten names found.
check_true("nix/package's environment reads were found", len(read_names) >= 10,
           "got %d -- retarget this if the lookup shape changes" % len(read_names))
check_true("nix/the derivation reaches a digit-bearing env var",
           any(any(c.isdigit() for c in n) for n in read_names),
           "sorted: %s" % sorted(read_names))

# One name is knowingly left unwired, and it is named rather than hidden by a narrower
# derivation: `LAYA_SHA256_DIGESTS` carries a JSON object, and this module has no way to prove a
# shell-quoting claim holds -- nothing in CI evaluates a NixOS module, so every check here is
# textual. Listing the exception keeps the gap asserted at exactly one name.
UNWIRED = {"LAYA_SHA256_DIGESTS"}
check("nix/module reaches every env var laya reads",
      sorted(read_names - set(assigned) - UNWIRED), [])
# The other direction is the silent failure: a misspelled name is a perfectly good string,
# systemd exports it, no Python ever looks at it, and the operator's setting does nothing.
check("nix/module assigns no env var laya never reads",
      [n for n in assigned if n not in read_names], [])
# And the exception list has to stay the size of the real gap: an entry that got wired up, or a
# name core stopped reading, is a stale excuse that would hide the next one.
check("nix/unwired exceptions are all still unwired and still real",
      sorted(n for n in UNWIRED if n not in read_names or n in assigned), [])

# An option that nothing reads is a promise the unit does not keep, and the mirror case -- a
# setting the unit applies with no option to turn it -- is a host that cannot change it.
declared = re.findall(r"^    ([a-zA-Z]+) = lib\.mkOption \{", nix_module, re.M)
check_true("nix/module's options were found", len(declared) >= 5,
           "got %r -- retarget this if the option block changes shape" % (declared,))
check("nix/every declared option is used by the unit",
      [n for n in declared if "cfg.%s" % n not in nix_module], [])


def option_text(opt):
    _b = re.search(r"^    %s = lib\.mkOption \{(.*?)\n    \};" % opt, nix_module, re.S | re.M)
    return _b.group(1) if _b else ""


def option_type(opt):
    """The declared `type = ...;` line, and nothing else from the block.

    Read it off the type, not off the option: `mpsAmpMinRows`'s description spells out
    `ints.positive` to explain why it is `ints.positive`, and a check that searched the whole
    block was satisfied by that sentence while the type line said `ints.unsigned`. Prose naming a
    constraint is not the constraint.
    """
    _t = re.search(r"^\s*type = (.+);$", option_text(opt), re.M)
    return _t.group(1) if _t else ""


# Every new knob is opt-in: unset means the unit exports nothing and the runtime's own default
# applies, so a host that ignores them gets today's behaviour byte for byte.
for opt in ("rootPath", "logLevel", "maxConcurrent", "cudaAmp", "cpuAmp", "mpsAmpMinRows",
            "maxLoaded", "maxTokenBudget", "revision", "defaultModel"):
    _t = option_text(opt)
    check_true("nix/module declares %s" % opt, _t != "", "option not found")
    check_true("nix/%s is opt-in (nullOr, default null)" % opt,
               _t.count("nullOr") == 1 and re.search(r"^\s*default = null;", _t, re.M) is not None,
               _t.strip()[:120])
    check_true("nix/%s is guarded by a != null optionalAttrs" % opt,
               re.search(r"lib\.optionalAttrs \(cfg\.%s != null\)" % opt, nix_module) is not None,
               "the unit would export the variable even when the host left it unset")

# The same lesson as `models`, stated for the whole module: no option may carry a closed list of
# names that somebody else validates. uvicorn checks the log level, laya checks the checkpoint
# name, and a copy here can only fall behind them -- which is how the module came to refuse ten
# spellings of a value the server accepts.
check_true("nix/module declares no closed enum over names it does not own",
           "types.enum" not in nix_module, "an enum in this module is a copy of someone "
           "else's list; defer to the thing that validates the value")

# So the AMP vocabularies are read out of `laya.agent` instead, from the comparisons that follow
# each lookup rather than from a list written down here: a spelling core starts accepting shows up
# in the set on its own, and this module's prose has to name it. The sets differ by device -- CPU
# takes bf16 only, CUDA takes fp16 too -- and that asymmetry is the whole content of the two
# options, so the check runs in both directions like the env-var one above. A `fp16` added to the
# `cpuAmp` description is a promise the runtime does not keep.
agent_src = read(os.path.join("laya", "agent.py"))


def amp_tokens(var):
    anchor = re.search(r'environ\.get\("%s"' % var, agent_src)
    if anchor is None:
        return None                      # reported by the non-vacuity guard below
    window = agent_src[anchor.end():anchor.end() + 300]
    return sorted({t for group in re.findall(r'\bin \(([^)]*)\)', window)
                   for t in re.findall(r'"([^"]+)"', group)})


AMP_MEANING = {"cudaAmp": "LAYA_CUDA_AMP", "cpuAmp": "LAYA_CPU_AMP"}
for _opt, _var in sorted(AMP_MEANING.items()):
    accepted = amp_tokens(_var)
    check_true("nix/%s's dtype list came from laya/agent.py" % _opt, bool(accepted),
               "no `in (...)` comparison follows `environ.get(\"%s\")`; retarget this" % _var)
    _d = re.search(r"description = ''(.*?)''", option_text(_opt), re.S)
    described = _d.group(1) if _d else ""
    check_true("nix/%s names every dtype core accepts for %s" % (_opt, _var),
               all("`%s`" % t in described for t in accepted),
               "core compares %s against %s; the option text says: %s"
               % (_var, accepted, described.strip()[:160]))
    offered = [t for t in ("fp16", "float16", "bf16", "bfloat16")
               if "`%s`" % t in described and t not in accepted]
    check_true("nix/%s offers no dtype core ignores for it" % _opt, not offered,
               "%s is not in the set laya.agent compares %s against" % (offered, _var))

# `mpsAmpMinRows` has no vocabulary, but its type is a claim: laya clamps a value below 1 up to 1,
# so `ints.positive` is what keeps the module from accepting an input the service reinterprets.
check_true("nix/mpsAmpMinRows's type refuses what the runtime would clamp",
           "positive" in option_type("mpsAmpMinRows"),
           "type = %s -- ints.unsigned or ints.atLeast 0 would let 0 through to be rewritten"
           % option_type("mpsAmpMinRows"))

# Both AMP options tell the operator that the transport refuses whitespace, and that is a claim
# about the type, not the prose: the runtime lower-cases but does not trim, so `"bf16 "` is a
# silently inert setting. Checked where the claim is made.
for _opt in ("cudaAmp", "cpuAmp"):
    _t = option_type(_opt)
    check_true("nix/%s's type carries its own no-whitespace claim" % _opt,
               "strMatching" in _t and "nullOr" in _t,
               "type = %s -- the description promises a single token" % _t)

# The device split is the entire content of those two options: CUDA takes a half-precision
# spelling, CPU does not. Asserted from core rather than from the prose, so the day that stops
# being true the gate says so instead of the option text quietly overpromising.
_cuda, _cpu = amp_tokens("LAYA_CUDA_AMP"), amp_tokens("LAYA_CPU_AMP")
check_true("nix/CUDA and CPU accept different dtype spellings", _cuda != _cpu,
           "cuda=%r cpu=%r -- if laya honours the same set on both devices now, the `cpuAmp` "
           "description's cross-reference needs retargeting" % (_cuda, _cpu))


# --------------------------------------------------------------- declared extras
# The runtime error in laya/structured.py tells users to install `laya[structured]`, and the
# docs and README repeat it. A reference to an extra pyproject.toml does not declare is a dead
# end for anyone who follows it, so every `laya[...]` in the code and docs must resolve (#348).
_extra_section = pyproject.split("[project.optional-dependencies]")[1].split("\n[")[0]
_declared_extras = set(re.findall(r"^([a-z][\w-]*)\s*=\s*\[", _extra_section, re.M))
_referenced_extras = {}
for _dirpath, _dirnames, _filenames in os.walk("."):
    # `.venv` is where CONTRIBUTING tells contributors to install, and it is not the repository.
    _dirnames[:] = [d for d in _dirnames
                    if d not in (".git", "__pycache__", "node_modules", ".pytest_cache", ".venv")]
    for _f in _filenames:
        if not _f.endswith((".py", ".md", ".yml", ".toml")):
            continue
        _p = os.path.normpath(os.path.join(_dirpath, _f))
        for _group in re.findall(r"laya\[([a-z][\w,-]*)\]", read(_p)):
            for _extra in _group.split(","):
                _referenced_extras.setdefault(_extra.strip(), set()).add(_p)

check("extras/every referenced extra is declared",
      sorted(set(_referenced_extras) - _declared_extras), [])

# ------------------------------------------------------- push concurrency (#399)
# github.ref is refs/heads/main for every push, so a group of <workflow>-${{ github.ref }}
# is one group for the whole branch. Rapid merges then cancel each other. Pull requests
# stay on github.ref (a new commit still cancels the obsolete run); pushes use github.sha.
# docs.yml has a concurrency group and is included once that group uses this expression.
# It still deploys GitHub Pages on github.ref so an older build cannot publish over a
# newer one (#493 left that group ref-scoped on purpose).
_PER_COMMIT = "github.event_name == 'pull_request' && github.ref || github.sha"
_BARE_REF_GROUP = re.compile(
    r"(?m)^[ \t]*group:\s+\S+-\$\{\{\s*github\.ref\s*\}\}(?:\s+#.*)?\s*$"
)


def _concurrency_block(text):
    match = re.search(r"(?m)^concurrency:\n(?:[ \t]+[^\n]*\n)*", text)
    return match.group(0) if match else ""


_concurrency_workflows = ["ci.yml", "docker.yml", "security.yml"]
_docs_workflow = read(os.path.join(".github", "workflows", "docs.yml"))
_docs_concurrency = _concurrency_block(_docs_workflow)
if _docs_concurrency and _PER_COMMIT in _docs_concurrency:
    _concurrency_workflows.append("docs.yml")

for _wf in _concurrency_workflows:
    _text = _docs_workflow if _wf == "docs.yml" else read(os.path.join(".github", "workflows", _wf))
    _block = _docs_concurrency if _wf == "docs.yml" else _concurrency_block(_text)
    _stem = _wf[:-4]
    check_true(
        "%s/per-commit concurrency for pushes to main" % _wf,
        ("group: %s-${{ %s }}" % (_stem, _PER_COMMIT)) in _block,
        "want group: %s-${{ %s }}" % (_stem, _PER_COMMIT),
    )
    check_true(
        "%s/concurrency group is not github.ref alone" % _wf,
        _block != "" and _BARE_REF_GROUP.search(_block) is None,
        "a ref-only group collapses every push to main into one run",
    )

# ------------------------------------------------- the eval gate is not cancelled mid-run
# Scoped to evals.yml on purpose. A general "cancel-in-progress may only be false or name
# pull_request" rule would also have to be right about every workflow a future change adds, and
# nothing in the repository establishes it; the workflows that run on pull requests are already
# covered by the per-commit group checks above.
#
# What is true here, and only here: no evals trigger makes an in-flight run obsolete.
# `github.ref` is the default branch for `schedule` and for a `workflow_dispatch` on it, so the
# weekly baseline shared a group with a manual re-run, as do two dispatches on one ref, and
# `cancel-in-progress: true` made whichever started second kill the first. `release: published`
# never collided: its ref is the tag (`refs/tags/<tag_name>`), so each release had its own group.
# The job spends 60 minutes downloading weights and the dataset, and a cancelled run uploads no
# report, so a cancellation reads as a clean gate.
_EVALS_WORKFLOW = read(os.path.join(".github", "workflows", "evals.yml"))
# Non-vacuity, independent of the concurrency block: if evals.yml is renamed or its triggers
# change, this reports instead of the check below passing on a file it did not understand.
check_true("evals.yml still declares its three triggers",
           "schedule:" in _EVALS_WORKFLOW
           and "release:" in _EVALS_WORKFLOW
           and "workflow_dispatch:" in _EVALS_WORKFLOW)

_EVALS_CANCEL = re.search(
    r"(?m)^[ \t]*cancel-in-progress:\s*(.+?)\s*$", _concurrency_block(_EVALS_WORKFLOW))
# An absent key means the Actions default, which is false, so nothing is cancelled. A bare `true`
# is the only value that discards a run in flight.
check_true(
    "evals.yml/cancels no in-flight run",
    _EVALS_CANCEL is None or _EVALS_CANCEL.group(1).strip().lower() in ("false", "no", "off"),
    "cancel-in-progress: %s discards a 60-minute run that has already started; a cancelled run "
    "uploads no report, so it reads as a clean gate"
    % (_EVALS_CANCEL.group(1) if _EVALS_CANCEL else "true"),
)

# --------------------------------------------------------------- the API reference
# `laya.__all__` is what `from laya import *` ships and what the README tells people to call, so
# an export no page under docs/ names cannot be looked up at all. `cached_embed_fn` was one: the
# README documents wrapping an embedder in it, each of its three shortlisting siblings has a
# `:::` directive in reference/helpers.md, and it appeared on no docs page.
#
# Like every other check in this file, this one reads source under the directory it walks and
# imports nothing: no `laya.__getattr__` name resolves, torch never loads, and a computed `__all__`
# returns [] for the guard below to report instead of an import quietly yielding whatever the
# environment happens to have installed.
def _exported_names():
    tree = ast.parse(read(os.path.join("laya", "__init__.py")))
    for node in tree.body:
        targets = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in targets):
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                return [e.value for e in value.elts if isinstance(e, ast.Constant)]
            return []          # computed `__all__`: the guard below reports it rather than passing
    return []


_docs_pages = [p for p in _md if p.split(os.sep)[0] == "docs"]
_docs_text = "".join(read(p) for p in _docs_pages)
_export_list = _exported_names()
# Guards, not formalities: an empty page list or an unreadable `__all__` would let the sweep below
# pass without checking anything.
check_true("docs/reference pages found under docs/", len(_docs_pages) >= 8, _docs_pages[:3])
check_true("docs/__all__ read as a literal from laya/__init__.py",
           len(_export_list) >= 20, len(_export_list))

_unreferenced = [n for n in _export_list if not re.search(r"\b%s\b" % re.escape(n), _docs_text)]
check_true("docs/every exported name appears on a docs page",
           not _unreferenced, "named nowhere in docs/: %s" % _unreferenced)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all packaging tests passed")
sys.exit(1 if FAIL else 0)
