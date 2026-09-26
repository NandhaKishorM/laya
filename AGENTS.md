# AGENTS.md

Context for AI coding assistants (Claude Code, Codex, Cursor, Copilot, Gemini CLI) working in this repository.

**Laya** is a fast, local, on-device decision engine. These rules mirror [CONTRIBUTING.md](CONTRIBUTING.md); the assistant-facing summary lives here because coding agents read this file automatically.

## Do NOT

- Introduce a dependency on a hosted service or external API. Features must run in the user's own process or on their own hardware; a feature that only works against a hosted backend belongs in a separate integration, not here.
- Change the public API without updating `tests/test_hooks_api.py` in the same pull request — that suite is the API contract.
- Skip the CI gates before declaring a change done:

  ```bash
  ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
  python -m compileall -q laya/ tests/
  ```

- Reformat files wholesale, reorder imports, or "modernize" surrounding code. Match the style of the file being edited; a diff full of formatting noise gets a change rejected.
- Commit secrets, tokens, or large binary files.
- Open pull requests in a language other than English. The project is triaged in English.
- Invent conventions. Use conventional commit prefixes matching the history (`feat(agent):`, `fix(router):`, `perf(common):`, `docs(hooks):`, `test(batch):`), and keep one logical change per commit.

## Where to look

| Editing | Read | Check |
|---|---|---|
| `laya/` core | — | plain script suites: `python tests/test_router.py`, `tests/test_criteria.py`, `tests/test_hooks.py`, `tests/test_hooks_api.py` |
| server path | `tests/test_serve.py` | `python -m pytest tests/test_serve.py` (skips when the `serve` extra is missing) |
| ONNX path | `tests/test_onnx.py` | `python -m pytest tests/test_onnx.py` (skips when the `onnx` extra is missing) |
| fast/local paths | `tests/test_fast.py`, `tests/test_local_e2e.py`, `tests/test_mcp_local_e2e.py` | need CUDA or checkpoints under `~/laya_models`; skipped otherwise |
| docs, docstrings in `laya/` | `docs/`, `docs/.nav.yml` for page order | `pip install -r requirements-docs.txt`, then `zensical build --strict --clean` with no `griffe:` lines in the output |

Optional extras are declared in `pyproject.toml` (`serve`, `fast`, `onnx`, `langchain`, `langgraph`); install only what the change needs.

## Pull requests

- Rebase onto the latest `main` so the diff is only your change.
- Keep it focused; split unrelated work into another PR.
- If a change moves numbers, report the before and after. The maintainer verifies decisions against real checkpoints, and measured deltas (probability changes, latency, memory) are what gets a change merged.
- Leave one runnable check behind for non-trivial logic; an assert-based script is enough.
