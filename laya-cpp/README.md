# laya-cpp

A C++17 port of [laya](../README.md), following the same plan as [laya-go](../laya-go/README.md):
the Python package is the reference and the port is checked against goldens the reference records.

**Status: skeleton only.** There is no library code yet beyond `laya::version()`. This directory
holds the build, the presets, the CI lane and the test-count checker, so the toolchain can be
reviewed before anything is ported. ONNX Runtime support (`-DLAYA_ONNX=ON`) is not implemented
and is refused at configure time.

## Build and test

Needs CMake 3.25+ and a C++17 compiler (gcc 13, clang 18 and Apple clang are exercised). The first
configure fetches GoogleTest, pinned by commit.

```sh
cd laya-cpp
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
# also: asan-ubsan, tsan, release
```

Options: `LAYA_BUILD_TESTS` (default ON), `LAYA_ONNX` (default OFF).

Floating point is compiled with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC) and
warnings are errors. The `compile_flags` test fails if either flag is missing from a laya-cpp
translation unit.

## Test-count gate

`scripts/check_test_xml.py` reads JUnit XML and fails on any skipped or failed test and below
`--min-tests`; `scripts/test_check_test_xml.py` proves it can fail:

```sh
ctest --preset debug --output-junit "$PWD/build/debug/junit.xml"
python3 scripts/check_test_xml.py build/debug/junit.xml --min-tests 2
python3 scripts/test_check_test_xml.py
```
