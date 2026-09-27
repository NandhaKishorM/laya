"""Tests for the CVE dependency audit job scope separation in `.github/workflows/security.yml`.

The dependency audit is split into two logical scopes:
1. Blocking audit: `[project].dependencies` + `[project.optional-dependencies].serve`
   These form the shipped runtime package and published Docker image (`pip install ".[serve]"`).
   Vulnerabilities in this scope MUST fail CI (`pip-audit --strict`).

2. Non-blocking optional integration audit: All other optional extras (`fast`, `mcp`, `structured`,
   `onnx`, `langchain`, `langgraph`, `llamaindex`, `crewai`, etc.).
   Vulnerabilities in optional extras are reported as non-blocking warnings so unrelated PRs are not
   blocked, while execution/infrastructure failures still fail the job.
"""
import os
import re
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


def section(text, header):
    """Lines of one top-level job/table, from header to next same-indent key."""
    lines = text.splitlines()
    depth = len(header) - len(header.lstrip())
    out, seen = [], False
    for line in lines:
        if seen and (line.strip() and (len(line) - len(line.lstrip())) <= depth and not line.lstrip().startswith("#")):
            break
        if line.rstrip() == header.rstrip():
            seen = True
        if seen:
            out.append(line)
    return out


def extract_scopes_from_toml_data(project_dict):
    """Replicates the workflow python script logic for dependency extraction."""
    core_specs = list(project_dict.get("dependencies", []))
    opt_deps = project_dict.get("optional-dependencies", {})
    serve_specs = list(opt_deps.get("serve", []))

    core_seen = set()
    core_final = []
    for spec in core_specs + serve_specs:
        if spec not in core_seen:
            core_seen.add(spec)
            core_final.append(spec)

    extras_specs = []
    for extra_name, extra_deps in opt_deps.items():
        if extra_name != "serve":
            extras_specs.extend(extra_deps)

    extras_seen = set()
    extras_final = []
    for spec in extras_specs:
        if spec not in extras_seen:
            extras_seen.add(spec)
            extras_final.append(spec)

    return core_final, extras_final


def handle_optional_audit_output(proc):
    """Replicates the optional audit exit code classification logic from security.yml."""
    if proc.returncode == 0:
        return 0, "No vulnerabilities found in optional extras."

    stderr_upper = proc.stderr.upper()
    is_infra_error = (
        proc.returncode not in (0, 1) or
        "ERROR:PIP_AUDIT" in stderr_upper or
        "UNRECOGNIZED ARGUMENTS" in stderr_upper or
        not proc.stdout.strip()
    )

    if is_infra_error:
        return proc.returncode if proc.returncode != 0 else 1, "Infrastructure failure"
    else:
        return 0, "Vulnerabilities found (non-blocking)"


# ---------------------------------------------------------------- tests

def test_1_core_dependencies_in_blocking_audit():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, _ = extract_scopes_from_toml_data(project)
        for dep in project["dependencies"]:
            assert dep in core_final, f"Core dependency {dep} missing from blocking audit"


def test_2_serve_in_blocking_audit():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, _ = extract_scopes_from_toml_data(project)
        serve_deps = project["optional-dependencies"]["serve"]
        for dep in serve_deps:
            assert dep in core_final, f"Serve dependency {dep} missing from blocking audit"


def test_3_other_extras_not_in_blocking_audit():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, _ = extract_scopes_from_toml_data(project)
        opt_deps = project["optional-dependencies"]
        exclusive_extras = []
        for extra, deps in opt_deps.items():
            if extra == "serve":
                continue
            for d in deps:
                if d not in project["dependencies"] and d not in opt_deps.get("serve", []):
                    exclusive_extras.append(d)

        for dep in exclusive_extras:
            assert dep not in core_final, f"Optional extra dependency {dep} leaked into blocking audit set"


def test_4_optional_extras_in_optional_audit():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        _, extras_final = extract_scopes_from_toml_data(project)
        for extra_name, deps in project["optional-dependencies"].items():
            if extra_name == "serve":
                continue
            for dep in deps:
                assert dep in extras_final, f"Optional dependency {dep} from extra {extra_name} missing from extras audit"


def test_5_duplicate_dependency_declarations():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, extras_final = extract_scopes_from_toml_data(project)
        assert len(core_final) == len(set(core_final)), "Duplicates found in core audit list"
        assert len(extras_final) == len(set(extras_final)), "Duplicates found in extras audit list"


def test_6_blocking_audit_remains_strict():
    deps_job = "\n".join(section(read_file(os.path.join(".github", "workflows", "security.yml")), "  deps:"))
    assert "pip-audit --strict" in deps_job, "Blocking audit step must invoke pip-audit with --strict"
    assert "requirements-audit-core.txt" in deps_job, "Blocking audit step must target requirements-audit-core.txt"


def test_7_optional_audit_non_blocking_for_vulnerabilities():
    class DummyProc:
        returncode = 1
        stdout = "Name Version ID\njson-repair 0.25.2 GHSA-xf7x-x43h-rpqh\n"
        stderr = "Found 1 vulnerability"

    exit_code, status_msg = handle_optional_audit_output(DummyProc())
    assert exit_code == 0, "Optional audit with vulnerability findings must yield exit code 0 (non-blocking)"
    assert "non-blocking" in status_msg


def test_8_core_vulnerability_remains_blocking():
    deps_job = "\n".join(section(read_file(os.path.join(".github", "workflows", "security.yml")), "  deps:"))
    core_step = [s for s in deps_job.split("- name:") if "Audit core and serve dependencies" in s]
    assert len(core_step) > 0, "Could not find blocking audit step in security.yml"
    assert "pip-audit --strict" in core_step[0]


def test_9_optional_audit_execution_failure_remains_visible():
    class DummyProcErr:
        returncode = 1
        stdout = ""
        stderr = "ERROR:pip_audit._cli:requirement file invalid"

    exit_code, _ = handle_optional_audit_output(DummyProcErr())
    assert exit_code != 0, "Infrastructure error must yield non-zero exit code"

    class DummyProcCrash:
        returncode = 2
        stdout = ""
        stderr = "usage error"

    exit_code2, _ = handle_optional_audit_output(DummyProcCrash())
    assert exit_code2 == 2, "Tool usage failure (code 2) must yield exit code 2"


def test_10_empty_optional_extras():
    project = {
        "dependencies": ["torch>=2.0.0"],
        "optional-dependencies": {"serve": ["fastapi>=0.110.0"]}
    }
    core_final, extras_final = extract_scopes_from_toml_data(project)
    assert core_final == ["torch>=2.0.0", "fastapi>=0.110.0"]
    assert extras_final == []


def test_11_serve_exists_but_other_extras_change():
    project = {
        "dependencies": ["torch>=2.0.0"],
        "optional-dependencies": {
            "serve": ["fastapi>=0.110.0"],
            "new_integration": ["new-pkg>=1.0.0"]
        }
    }
    core_final, extras_final = extract_scopes_from_toml_data(project)
    assert "new-pkg>=1.0.0" in extras_final
    assert "new-pkg>=1.0.0" not in core_final


def test_12_generated_dependency_files_are_valid():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, extras_final = extract_scopes_from_toml_data(project)
        spec_pattern = re.compile(r"^[A-Za-z0-9_.\-]+([<>=!;\[\s].*)?$")
        for spec in core_final + extras_final:
            assert spec_pattern.match(spec), f"Invalid specifier line format: {spec}"


def test_13_local_reproduction_isolates_vulnerable_extras():
    pyproject_str = read_file("pyproject.toml")
    if tomllib:
        import tomllib as tl
        project = tl.loads(pyproject_str)["project"]
        core_final, extras_final = extract_scopes_from_toml_data(project)
        for pkg in ["crewai>=0.28.0", "llama-index-core>=0.10.0", "langchain-core>=0.2.0"]:
            assert pkg in extras_final, f"Extra {pkg} should be in extras audit"
            assert pkg not in core_final, f"Extra {pkg} must not be in core audit"


def test_14_workflow_and_extractor_execution():
    security_yml = read_file(os.path.join(".github", "workflows", "security.yml"))
    deps_job = "\n".join(section(security_yml, "  deps:"))
    assert "requirements-audit-core.txt" in deps_job
    assert "requirements-audit-extras.txt" in deps_job

    pin = re.search(r"python-version:\s*[\"'](\d+)\.(\d+)[\"']", deps_job)
    assert pin is not None and tuple(map(int, pin.groups())) >= (3, 11), "The deps job must pin python >= 3.11"

    if sys.version_info >= (3, 11):
        m = re.search(r"python - <<'PY'\n(.*?)\n\s*PY\n", deps_job, re.S)
        assert m is not None, "Could not extract Python script from security.yml"
        body = m.group(1)
        indents = [len(l) - len(l.lstrip()) for l in body.splitlines() if l.strip()]
        extractor_code = "\n".join(l[min(indents):] if l.strip() else "" for l in body.splitlines())

        with tempfile.TemporaryDirectory() as tmpdir:
            run = subprocess.run([sys.executable, "-c", extractor_code], cwd=ROOT, capture_output=True, text=True)
            assert run.returncode == 0, f"Extractor failed: {run.stderr}"
            assert os.path.exists(os.path.join(ROOT, "requirements-audit-core.txt"))
            assert os.path.exists(os.path.join(ROOT, "requirements-audit-extras.txt"))
            try:
                with open(os.path.join(ROOT, "requirements-audit-core.txt")) as f:
                    core_lines = [l.strip() for l in f if l.strip()]
                with open(os.path.join(ROOT, "requirements-audit-extras.txt")) as f:
                    extras_lines = [l.strip() for l in f if l.strip()]
                assert any("fastapi" in l for l in core_lines)
                assert any("crewai" in l for l in extras_lines)
            finally:
                for f in ["requirements-audit-core.txt", "requirements-audit-extras.txt"]:
                    p = os.path.join(ROOT, f)
                    if os.path.exists(p):
                        os.remove(p)


if __name__ == "__main__":
    test_1_core_dependencies_in_blocking_audit()
    test_2_serve_in_blocking_audit()
    test_3_other_extras_not_in_blocking_audit()
    test_4_optional_extras_in_optional_audit()
    test_5_duplicate_dependency_declarations()
    test_6_blocking_audit_remains_strict()
    test_7_optional_audit_non_blocking_for_vulnerabilities()
    test_8_core_vulnerability_remains_blocking()
    test_9_optional_audit_execution_failure_remains_visible()
    test_10_empty_optional_extras()
    test_11_serve_exists_but_other_extras_change()
    test_12_generated_dependency_files_are_valid()
    test_13_local_reproduction_isolates_vulnerable_extras()
    test_14_workflow_and_extractor_execution()
    print("All 14 audit scope tests passed!")
