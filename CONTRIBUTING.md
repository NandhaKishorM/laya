# Contributing

Thanks for helping improve Laya. This guide keeps reviews fast and the history clean.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Scope

Laya is a fast, local, on-device decision engine. Keep additions in that spirit: they should run
in the user's own process or on their own hardware, with no dependency on a hosted service or an
external API. A feature that only works against a hosted backend is out of scope for the core
package; it belongs in a separate integration or a community project.

## Ways to contribute

- Report a bug with the bug report template.
- Request or discuss a feature with the feature request template.
- Improve the docs under [`docs/`](docs/index.md), including the [hooks guide](docs/hooks/index.md).
- Fix a bug or add a feature with a focused pull request.
- Share benchmarks, evaluations, or integration reports, which the project treats as first class.

## Development setup

Laya supports Python 3.10 to 3.13. Use a virtual environment so nothing leaks into your system.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"
```

Optional extras are declared in `pyproject.toml`: `serve`, `fast`, `mcp`, `structured`, `onnx`,
`langchain`, `langgraph`, `llamaindex`, `crewai`.
Install the ones a change needs, for example `pip install -e ".[serve]"`.

If you use [uv](https://docs.astral.sh/uv/), the same setup is below. uv environments come without
`pip`, so use `uv pip install` wherever this guide says `pip install`.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[mcp]"
```

The server and ONNX paths are exercised by `tests/test_serve.py` and `tests/test_onnx.py`, which
skip when their extras are not installed -- `tests/test_serve.py` skips every test it has on one
line of output, so the install decides whether the server checks ran or quietly did not.
`tests/test_audit_regressions.py` reaches the same server path from five tests that do not skip:
without the `serve` extra they fail with `ModuleNotFoundError: No module named 'fastapi'`.

## Running the tests

Most suites are plain scripts, so no test runner is required:

```bash
python tests/test_router.py
python tests/test_criteria.py
python tests/test_hooks.py
python tests/test_hooks_api.py
```

A handful are pytest based and run under one pytest invocation:

```bash
python -m pytest tests/test_serve.py tests/test_truncation_direction.py -q
```

The one list that both the CI `test` job and the release gate run is
[`scripts/test_suites.py`](scripts/test_suites.py). It keeps two lists because `python tests/<name>.py`
does not execute a pytest suite -- it defines the test functions and exits 0, having asserted nothing
(#374): `SCRIPT_SUITES` print their own PASS/FAIL summary and exit non-zero on failure, and
`PYTEST_SUITES` are collected and run by pytest in a single invocation. Some lanes still name their own
suites inline under `.github/workflows/`, and `tests/test_packaging.py` fails if a suite ends up in
neither the shared list nor any workflow.

That list takes the extras the CI `test` job installs, not `mcp` alone: `serve` for `laya.serve`
and `langchain` for the real `Runnable` base class. Two more are not in any extra -- `pytest`, for
the `PYTEST_SUITES` half, and `httpx`, which `fastapi.testclient` imports and `serve` does not
declare (it arrives with `langchain-core` here, which is not a dependency to lean on). Installing
what that job installs is what makes a local run mean the same thing as the gate:

```bash
pip install -e ".[mcp,serve,langchain]" pytest httpx
python scripts/test_suites.py
```

With `mcp` alone the list is not green, and what it loses is not all visible in the failures:
`tests/test_langchain_real_mode.py` fails by design, which is how a lane that runs this list is
told to install the extra, the five server tests in `tests/test_audit_regressions.py` fail with
`ModuleNotFoundError`, and `tests/test_serve.py` skips its whole file without failing anything.

The suites that need hardware, a compiler or local weights are not part of the PR `test` job:

- `tests/test_compile_cuda.py` -- needs CUDA; CI installs CPU-only torch.
- `tests/test_fast.py` -- needs CUDA and the `tilelang` extra.
- `tests/test_fast_cpu.py` -- needs the `tilelang` extra and a C++ compiler; its CUDA parity checks skip.
- `tests/test_local_e2e.py` -- expects checkpoints under `~/laya_models`.
- `tests/test_mcp_local_e2e.py` -- expects checkpoints under `~/laya_models` and the `mcp` extra.
- `tests/test_onnx.py` -- needs the `onnx` extra; it runs in the ONNX lane of `evals.yml`, not the PR job.
- `tests/test_server_example.py` -- needs the weights `examples/server.py` loads; it runs in `evals.yml`.

Before opening a pull request, run the lint and compile checks the CI runs:

```bash
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
```

## Documentation

The site at [nandhakishorm.github.io/laya](https://nandhakishorm.github.io/laya/) builds from
`docs/` with Zensical, and its API reference builds from the docstrings in `laya/`. A new page
appears under Guides without a config change; to put it somewhere else, add it to
[`docs/.nav.yml`](docs/.nav.yml). For a docs or docstring change, build the site the way the CI
does:

```bash
pip install -r requirements-docs.txt
zensical build --strict --clean
```

The output must have no `griffe:` lines. Those are docstring problems, such as a parameter the
signature does not have, and the CI fails on them.

## Style

- Keep the public API stable. If a change must move it, update `tests/test_hooks_api.py` in the same
  pull request, because that suite is the API contract.
- Prefer the standard library and what is already a dependency over a new one.
- Leave one runnable check behind for non-trivial logic: an assert based script is enough.
- Comment only where the code cannot say it, usually a non-obvious reason or a hardware caveat.
- Match the surrounding style rather than a personal preference.

## Commits

Use conventional commit prefixes, matching the history:

```
feat(agent): ...
fix(router): ...
perf(common): ...
docs(hooks): ...
test(batch): ...
```

Keep one logical change per commit. A focused pull request is much easier to review and merge than a
large one, and it is fine to split a big change into several PRs.

## Pull requests

The template asks for four things; filling them in is what makes review quick:

1. **What** changed.
2. **Why**, ideally the concrete use case, not only the mechanism.
3. **How it was verified**: the exact commands you ran.
4. Any follow-ups you deliberately left out.

Before you open one:

- Rebase onto the latest `main`, so the diff is only your change.
- Keep it focused; split unrelated work into another PR.
- Update docs or examples when the public API changes.
- Do not commit secrets, tokens, or large binary files.

If your change moves numbers, report the before and after: the maintainer verifies decisions against
real checkpoints, and measured deltas (probability changes, latency, memory) are what gets a change
merged.

## Reporting issues

The project is maintained in English, so please write issues in English when you can; it is the
common language for everyone reading and triaging.

Use the issue templates and include:

- What you expected and what you got.
- A minimal reproduction, ideally a short script.
- Your Laya version (`python -c "import laya; print(laya.__version__)"`), Python version, OS, and
  device or backend (CPU, CUDA, MPS, ONNX).

## Reviews

Reviewers may ask you to rebase, to split a change, or to add a regression test. Those are the
normal asks in this repository, not a rejection. If you disagree with a request, say so on the PR
and explain the tradeoff.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
