# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Laya** is a multilingual, non-autoregressive System 1 decision engine. It evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket, or JSON document) in a single forward pass — ~33ms on a T4 GPU. Three checkpoints are available with a `Router` that picks the right one per request:

| Checkpoint | Encoder | Params | Context | Use Case |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 | English |
| `laya-multilingual` | mmBERT-base | 322M | 1024 | 100+ languages, 2x faster |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | Fine-tuned for typed-decision workflows |

## Development Commands

### Running Tests

```bash
# Run all unit tests (no model weights loaded)
python tests/test_router.py
python tests/test_criteria.py
python tests/test_email.py
python tests/test_download.py
python tests/test_shortlist.py
python tests/test_decision_model.py
python tests/test_packaging.py
python tests/test_lang_guess.py
python tests/test_calibration_persistence.py
python tests/test_context_manager.py
python tests/test_criteria_normalization.py
python tests/test_empty_questions.py
python tests/test_local_e2e.py
python tests/test_router_memory.py
python tests/test_serve.py
python tests/test_shortlist_cosine.py
python tests/test_temperature_loading.py

# Or run via pytest
pytest tests/ -v
```

### Linting

```bash
# Ruff check (CI uses specific rules)
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120

# Byte-compile to catch syntax errors
python -m compileall -q laya/ tests/
```

### Building the Package

```bash
python -m build
python -m twine check dist/*
```

### Development Environment

```bash
# Using Nix (recommended)
nix develop  # dev shell with torch-bin, transformers, fastapi, pytest

# Or manually with virtual environment
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install "laya[serve]"  # for HTTP server
```

### Running the HTTP Server

```bash
# Via pip install
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve

# Or via Nix
nix run .#laya-serve
```

### Benchmarking

```bash
# Chinese workplace decisions benchmark (no models loaded - audit only)
python research/benchmarks/feishu_zh/audit.py
python -m unittest discover -s research/benchmarks/feishu_zh/tests -v

# Full benchmarks (requires GPU)
# See research/scripts/laya_benchmark_colab.ipynb
```

## Code Architecture

### Core Modules (`laya/`)

| Module | Responsibility |
|---|---|
| `agent.py` | High-level inference runtime (`Agent`, `load()`). Loads checkpoints from HF Hub or local paths, runs `system_one()` / `predict()` for batched decisions. |
| `router.py` | `Router` class: lazy-loads checkpoints, routes requests by script/language detection. Default `max_loaded=2` keeps English + multilingual hot. Supports `preload=True` for production. |
| `common.py` | Core architecture: `DecisionModel` (encoder + decision heads), sequence building (`build_sequence`), proper scoring rule rewards (`proper_reward`), TD(λ) targets, calibration (`ece_score`, `confidence_from_probs`), temperature clamping. |
| `lang.py` | Dependency-free script detection (`detect_script`, `analyse`, `is_english`, `guess_latin_language`). Uses stopword heuristics — not a full LID model. |
| `router.py` | Model routing logic with LRU eviction, explicit overrides (`model=`, `task=`, `lang=`), and caller-supplied `lang_guess` for external LID models. |
| `shortlist.py` | `predict_shortlist()` for high-cardinality choice questions: embeds labels, keeps top-k, runs one forward pass on shortlist. |
| `presets.py` | Built-in question schemas: `router_questions()`, `guard_questions()`, `moderation_questions()`, `triage_questions()`, `email_questions()`. |
| `email.py` | Email body cleaning utilities (`clean_email_body`, `email_state`). |
| `serve.py` | FastAPI HTTP server (`laya.serve` / `laya-serve` CLI) exposing `POST /v1/systemone` (Jev-compatible wire protocol). |

### Key Design Patterns

1. **Single Forward Pass**: All questions in a call are batched into one forward pass via `collate_items()`.
2. **Temperature Calibration**: Checkpoints ship per-type and per-bucket temperatures. `clamp_temperature()` enforces `[0.5, 5.0]` range — values outside are rejected with warning, falling back to 1.0.
3. **Router Precedence**: `explicit model` > `explicit task` > `auto_task_detection` > `explicit lang` > `lang_guess` > `built-in detection` > `default`.
3. **Lazy Loading**: `Router` downloads/builds checkpoints on first use. `max_loaded` controls LRU eviction (default 2).
4. **Context Manager**: Both `Agent` and `Router` implement `__enter__`/`__exit__` for GPU memory cleanup.

### Question Types

| Type | Output | Use Case |
|---|---|---|
| `choice` | Top label, probabilities per option, confidence | Department routing, intent classification |
| `score` | Expected level on ordinal rubric, distribution, confidence | Frustration level, urgency, severity |
| `noul` | Calibrated probability P(true) ∈ [0,1] | Phishing, spam, jailbreak, churn risk |

### Testing Philosophy

- Tests in `tests/` are **pure Python** — no model weights loaded, no network, no GPU.
- `test_router.py` and `test_lang_guess.py` verify routing logic and language detection.
- `test_calibration_persistence.py` validates temperature config persistence (synthetic configs, tiny fixtures).
- `test_local_e2e.py` is the only test that may load models (skipped in CI).

### CI Configuration

The CI (`.github/workflows/ci.yml`) runs:
- Chinese benchmark audit (no models)
- Tests across Python 3.10–3.13 with CPU-only torch
- Ruff linting + byte-compilation
- Package build + metadata validation

### Nix Integration

- Flake at `flake.nix` with overlay exposing `laya` and `laya-serve`.
- NixOS module at `nix/laya-serve.nix` for systemd service with CUDA access.
- Dev shell includes `torch-bin` (prebuilt CUDA wheels) to avoid source builds.

## Common Development Tasks

### Adding a New Test

Create a file in `tests/` following the existing pattern — use `PASS`/`FAIL` lists with `check()` helpers. Run with `python tests/test_new.py`.

### Modifying Routing Logic

Edit `laya/router.py`. The `route()` method contains the precedence chain. `analyse()` in `laya/lang.py` provides the script/language detection.

### Changing Model Architecture

Edit `laya/common.py` — `DecisionModel`, `build_model()`, `build_sequence()`. Also update `_verify_compatibility()` in `agent.py` if weight shapes change.

### Updating Checkpoint Configuration

Temperature and calibration config lives in `rl_agent_config.json` inside each checkpoint (not in the source repo). Fine-tuning notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`) handles training and export.

### Adding a Preset

Add a function to `laya/presets.py` and export it in `laya/__init__.py`.

## Important Notes

- **Python ≥ 3.10** required (enforced by `torch≥2.0`, `transformers≥4.48`).
- **No TensorFlow** — `USE_TF=0` in CI to prevent abseil deadlocks.
- **Tokenizers parallelism disabled** (`TOKENIZERS_PARALLELISM=false`) to avoid fork warnings.
- The `laya` package is **not** the model weights — those live on Hugging Face Hub (`convaiinnovations/laya*`).
- `model.safetensors` + `rl_agent_config.json` + tokenizer/encoder dirs = a valid checkpoint.
- Empty questions dict returns empty answers + zero token usage without any forward pass.