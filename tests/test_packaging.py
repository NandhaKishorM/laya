"""Packaging metadata must match what the dependencies actually need (#34).

Text parsing, not tomllib: the floor is 3.10 and tomllib arrives in 3.11.
"""
import os
import re
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
    # `.pytest_cache` ships a README of its own and is not part of the repository.
    _dirnames[:] = [d for d in _dirnames
                    if d not in (".git", "__pycache__", "node_modules", ".pytest_cache")]
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
port_map = re.search(r'-\s*"(\$\{[A-Z_]+:-(\d+)\}):(\$\{[A-Z_]+:-(\d+)\})"', http)
check_true("compose.http/port mapping is present", port_map is not None, http[:300])
if port_map:
    check("compose.http/host port equals container port", port_map.group(2), port_map.group(4))
    check("compose.http/both sides use the same variable", port_map.group(1), port_map.group(3))
check_true("compose.http/the server reads the same variable",
           'LAYA_PORT: "${LAYA_PORT:-8000}"' in http)
check_true("compose.http/shares the model cache",
           "model-cache:/home/laya/.cache/huggingface" in http)
# The base service is what `docker compose run --rm laya` uses; publishing it a port or
# changing its command would be a breaking change to the quickstart.
check_true("compose.http/leaves the quickstart service alone",
           "laya:" not in http, "compose.http.yaml overrides the base `laya` service")

# `pip install .` alone puts no `laya-serve` in the image, so the extra is load-bearing.
check_true("Dockerfile/installs the serve extra", '".[serve]"' in dockerfile, dockerfile[:400])
check_true("Dockerfile/still runs pip check", "pip check" in dockerfile)

# A Compose override replaces a service's `build` block whole. If the CUDA override does
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
for name in ("compose.yaml", "compose.example.yml", "compose.cuda.yaml", "compose.http.yaml"):
    check_true("compose/%s exists" % name, os.path.exists(name))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all packaging tests passed")
sys.exit(1 if FAIL else 0)
