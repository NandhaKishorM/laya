#!/usr/bin/env bash
# Reproducible local setup for Laya on Linux (CPU or CUDA).
#
# Creates an isolated virtualenv, installs PyTorch from the default PyPI index (CUDA-enabled
# wheels on Linux), downloads the three Laya checkpoints, and verifies the result with a real
# forward pass.
#
# Usage:   ./setup_laya.sh [--skip-models] [--skip-verify]   (or ./setup_laya_linux.sh)
#          TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu ./setup_laya.sh
# Idempotent: re-running only fixes what is missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # repository root
VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON:-python3}"

# Keep every byte this setup writes inside the project directory.
export PIP_CACHE_DIR="$ROOT/.pip-cache"
export HF_HOME="$ROOT/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export USE_TF=0          # Laya is torch-only; a stray TensorFlow install can deadlock loading
export USE_TORCH=1

SKIP_MODELS=0
SKIP_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found; install Python 3.10+ or set PYTHON=/path/to/python" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "error: $("$PYTHON_BIN" -V 2>&1) is too old; Laya requires Python 3.10+" >&2
  exit 1
fi

say "Host"
uname -srm
"$PYTHON_BIN" -c 'import platform,sys;print("python", sys.version.split()[0], platform.machine())'

if [ "$(uname -s)" != "Linux" ]; then
  echo "warning: this script targets Linux; continuing anyway" >&2
fi

say "Virtualenv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip

say "PyTorch"
# Installed before the project so the editable install reuses this build instead of pulling
# its own. Without an NVIDIA driver, the CPU wheel avoids the much larger CUDA build.
# An explicit TORCH_INDEX_URL always wins.
if [ -z "${TORCH_INDEX_URL:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
  echo "no NVIDIA driver found: using the CPU-only torch wheel"
fi
TORCH_ARGS=()
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  TORCH_ARGS=(--index-url "$TORCH_INDEX_URL")
fi
"$VENV/bin/python" -m pip install --quiet "${TORCH_ARGS[@]}" torch

say "Installing the laya package (editable, with test extras)"
# tests/test_serve.py imports httpx, which no declared extra provides.
"$VENV/bin/python" -m pip install --quiet -e "$ROOT[serve,structured,mcp]" httpx

if [ "$SKIP_MODELS" -eq 0 ]; then
  say "Checkpoints -> $ROOT/models"
  if [ -f "$ROOT/models/laya/model.safetensors" ]; then
    echo "already present, skipping download"
  else
    # One repo bundles all three checkpoints; ~2.3 GB of safetensors.
    ( cd "$ROOT" && "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "convaiinnovations/laya",
    local_dir="models/laya",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer.json"],
)
print("downloaded ->", p)
PY
    )
  fi
  # tests/test_local_e2e.py expects <root>/{laya,laya-multilingual,laya-typed-decisions};
  # the bundle ships the two extra checkpoints as subfolders of the first.
  ln -sfn laya/multilingual   "$ROOT/models/laya-multilingual"
  ln -sfn laya/typed-decisions "$ROOT/models/laya-typed-decisions"
  ls -l "$ROOT/models"
fi

# A truncated or silently-updated download turns into quietly wrong answers, so check the weights
# against the hashes recorded in verify/checkpoints.json. Runs whenever the weights are there,
# including on a --skip-models re-run.
if [ -f "$ROOT/models/laya/model.safetensors" ]; then
  say "Checking checkpoint integrity"
  "$VENV/bin/python" "$ROOT/verify/checkpoints.py" --models "$ROOT/models"
elif [ "$SKIP_MODELS" -eq 0 ]; then
  echo "warning: models/laya/model.safetensors is still missing" >&2
fi

if [ "$SKIP_VERIFY" -eq 0 ]; then
  say "Verifying with real weights"
  "$VENV/bin/python" "$ROOT/verify/laya_smoke_test.py" --models "$ROOT/models"
fi

say "Done"
echo "Activate with:  source \"$VENV/bin/activate\""
echo "Run the suite:  python tests/test_local_e2e.py \"$ROOT/models\""
