#!/usr/bin/env bash
# One entry point for every supported platform: detects the OS and runs its setup script.
#
# Usage:   ./setup_laya.sh [--skip-models] [--skip-verify]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
  Darwin) exec "$DIR/setup_laya_macos.sh" "$@" ;;
  Linux)  exec "$DIR/setup_laya_linux.sh" "$@" ;;
  *)
    echo "error: unsupported OS '$(uname -s)'. Use Linux or macOS; on Windows, use WSL." >&2
    exit 1 ;;
esac
