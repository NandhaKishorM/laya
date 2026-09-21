#!/usr/bin/env bash
# Reproducible local setup for Laya on this machine (macOS on Intel x86_64).
#
# Creates an isolated virtualenv, installs the only PyTorch release that still ships a
# macOS Intel wheel, installs the newest transformers that runs on it, downloads the three
# Laya checkpoints, and verifies the result with a real forward pass.
#
# Usage:   ./setup_laya.sh [--skip-models] [--skip-verify]
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

say "Host"
uname -srm
"$PYTHON_BIN" -c 'import platform,sys;print("python", sys.version.split()[0], platform.machine())'

if [ "$(uname -s)" != "Darwin" ]; then
  echo "warning: this script targets macOS; continuing anyway" >&2
fi

say "Virtualenv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip

say "Dependencies"
# torch: PyPI and download.pytorch.org both stopped publishing macOS x86_64 wheels after
# 2.2.2, so this is the newest PyTorch this CPU can run from a wheel.
# numpy: torch 2.2.2 was compiled against NumPy 1.x and fails to initialise under NumPy 2.
# transformers: 4.57.x is the last 4.x line (5.x requires torch>=2.4, unavailable here) and
# the oldest line that reads both the ModernBERT and mmBERT checkpoints.
# huggingface_hub is capped <1.0 by transformers 4.x.
"$VENV/bin/python" -m pip install \
  "numpy<2" \
  "torch==2.2.2" \
  "transformers==4.57.6" \
  "safetensors>=0.4.0" \
  "huggingface_hub>=0.34.0,<1.0"

say "Installing the laya package (editable)"
"$VENV/bin/python" -m pip install --quiet -e "$ROOT"

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

if [ "$SKIP_VERIFY" -eq 0 ]; then
  say "Verifying with real weights"
  "$VENV/bin/python" "$ROOT/laya_smoke_test.py" --models "$ROOT/models"
fi

say "Done"
echo "Activate with:  source \"$VENV/bin/activate\""
echo "Run the suite:  python tests/test_local_e2e.py \"$ROOT/models\""
