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
init = read(os.path.join("laya", "__init__.py"))

# ------------------------------------------------------------------ setup.py
# Metadata lives in pyproject.toml only; the legacy shim is gone.
check("repo/has no setup.py", os.path.exists(os.path.join(ROOT, "setup.py")), False)

# ------------------------------------------------------------------ Python floor
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

# ------------------------------------------------------------------ version, single-sourced
dynamic = re.search(r'dynamic\s*=\s*\[([^\]]*)\]', pyproject)
check_true("pyproject/declares a dynamic version", dynamic is not None and "version" in dynamic.group(1))
check("pyproject/has no static version", re.search(r'^version\s*=\s*"', pyproject, re.M), None)

attr = re.search(r'version\s*=\s*\{\s*attr\s*=\s*"([^"]+)"\s*\}', pyproject)
check_true("pyproject/dynamic version points at an attr", attr is not None)
check("pyproject/dynamic attr is laya.__version__", attr.group(1) if attr else None, "laya.__version__")

init_version = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
check_true("laya/__init__ defines a literal __version__", init_version is not None)
check(
    "laya/__init__ version is X.Y.Z",
    bool(init_version) and all(p.isdigit() for p in init_version.group(1).split(".")) and len(init_version.group(1).split(".")) == 3,
    True,
)

# ------------------------------------------------------------------ license
license_expr = re.search(r'^license\s*=\s*"([^"]+)"', pyproject, re.M)
check("pyproject/uses an SPDX license string", license_expr.group(1) if license_expr else None, "Apache-2.0")
check_true("pyproject/does not use the deprecated license table", "license = {" not in pyproject)
license_files = re.search(r'license-files\s*=\s*\[([^\]]*)\]', pyproject)
check_true("pyproject/lists license-files", license_files is not None and "LICENSE" in license_files.group(1))
# PEP 639: a license expression and "License ::" trove classifiers cannot coexist.
check("classifiers/has no license classifier", re.findall(r'"(License :: [^"]+)"', pyproject), [])

# ------------------------------------------------------------------ package discovery
check_true("pyproject/uses packages.find", "[tool.setuptools.packages.find]" in pyproject)
find_block = re.search(r'\[tool\.setuptools\.packages\.find\](.*?)(\n\[|\Z)', pyproject, re.S)
check_true(
    "pyproject/includes the laya package",
    find_block is not None and "laya*" in find_block.group(1),
)

# ------------------------------------------------------------------ transformers floor
# Checkpoints run on answerdotai/ModernBERT-large, which transformers only knows from 4.48.
transformers_floor = re.search(r'"transformers>=([\d.]+)"', pyproject)
check_true("pyproject/pins a transformers floor", transformers_floor is not None)
check_true(
    "transformers/floor covers ModernBERT",
    transformers_floor is not None and version_tuple(transformers_floor.group(1)) >= (4, 48),
    "ModernBERT support starts in transformers 4.48",
)

# ------------------------------------------------------------------ CI floor
workflow = read(os.path.join(".github", "workflows", "ci.yml"))
ci_versions = [version_tuple(v) for v in re.findall(r'"(\d+\.\d+)"', workflow)]
stale = [".".join(str(p) for p in v) for v in ci_versions if v < floor]
check("ci/tests no Python below requires-python", stale, [])

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all packaging tests passed")
sys.exit(1 if FAIL else 0)
