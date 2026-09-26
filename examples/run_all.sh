#!/usr/bin/env bash
# Run every example in order and report which ones work.
#
#   ./examples/run_all.sh              # all of them, oldest first
#   ./examples/run_all.sh 1[0-9]       # only examples 10-19 (basename glob)
#
# Each example is a separate process, so each one loads its own checkpoints; the full sweep takes
# several minutes on CPU and a bit less on MPS. Output is captured and shown only for failures.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Prefer a virtualenv at the repository root when there is one; otherwise any Python on PATH.
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
PATTERN="${1:-[0-9][0-9]_*.py}"
LOG="$(mktemp -t laya_examples.XXXXXX)"

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "no python found -- set PYTHON=/path/to/python" >&2
  exit 2
fi

pass=0
fail=0
failed=()

for f in "$ROOT"/examples/$PATTERN; do
  [ -e "$f" ] || { echo "no examples match '$PATTERN'" >&2; exit 2; }
  name="$(basename "$f")"
  start=$(date +%s)
  if "$PY" "$f" >"$LOG" 2>&1; then
    printf '  PASS  %-42s %3ds\n' "$name" "$(( $(date +%s) - start ))"
    pass=$((pass + 1))
  else
    printf '  FAIL  %-42s %3ds\n' "$name" "$(( $(date +%s) - start ))"
    fail=$((fail + 1))
    failed+=("$name")
    sed -n '1,40p' "$LOG" | sed 's/^/        /'
  fi
done

echo
echo "  $pass passed, $fail failed"
for f in "${failed[@]:-}"; do
  [ -n "$f" ] && echo "    FAIL $f"
done
rm -f "$LOG"
[ "$fail" -eq 0 ]
