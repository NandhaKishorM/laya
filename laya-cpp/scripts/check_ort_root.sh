#!/bin/sh
# Configure-only check of cmake/onnxruntime.cmake's refusals. usage: check_ort_root.sh <good-ort-root>
# A root with the wrong API version, a wrong VERSION_NUMBER, no header, or no library must each
# fail at configure time with the named message, and the good root must configure.
set -eu
good=$1
src=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
try() {  # name root want-substring ("" = must succeed)
  if out=$(cmake -S "$src" -B "$tmp/b-$1" -DLAYA_ONNX=ON -DLAYA_ONNXRUNTIME_ROOT="$2" 2>&1); then rc=0; else rc=1; fi
  if [ -z "$3" ]; then
    [ $rc -eq 0 ] || { echo "FAIL $1: good root refused"; fail=1; }
  else
    [ $rc -ne 0 ] || { echo "FAIL $1: accepted"; fail=1; return; }
    echo "$out" | tr -s ' \n' '  ' | grep -q "$3" || { echo "FAIL $1: wrong message"; echo "$out" | tail -4; fail=1; }
  fi
}
copy() { mkdir -p "$tmp/$1"; cp -R "$good/include" "$tmp/$1/include"; mkdir -p "$tmp/$1/lib"; cp -R "$good/lib/." "$tmp/$1/lib/"; cp "$good/VERSION_NUMBER" "$tmp/$1/" 2>/dev/null || true; }
try good "$good" ""
copy api; sed -i.bak 's/^#define ORT_API_VERSION [0-9]*/#define ORT_API_VERSION 99/' "$tmp/api/include/onnxruntime_c_api.h"
try wrong-api "$tmp/api" "API version '99'"
copy vn; echo 1.29.0 > "$tmp/vn/VERSION_NUMBER"
try wrong-version-number "$tmp/vn" "VERSION_NUMBER says"
copy nohdr; rm "$tmp/nohdr/include/onnxruntime_cxx_api.h"
try no-header "$tmp/nohdr" "not an ONNX Runtime release directory"
copy nolib; find "$tmp/nolib/lib" -maxdepth 1 -type f -name 'libonnxruntime*' -delete; find "$tmp/nolib/lib" -maxdepth 1 -type l -name 'libonnxruntime*' -delete
try no-library "$tmp/nolib" "library not found\|no libonnxruntime"
[ $fail -eq 0 ] && echo "check_ort_root: 5 cases ok"
exit $fail
