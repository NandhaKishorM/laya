# Compile defaults measurements (2026-10-03)

Source: PR bodies #718 and #576 (`gh pr view 718 --repo NandhaKishorM/laya --json body`
and the same command for 576). The requested `docs/fast-backends.md` is now
`docs/compile-and-fast-path.md`. Base: fa9a2a7, branch feat/compile-defaults.

Hardware: RTX 4070 Ti SUPER 16 GiB, torch 2.11.0+cu130, driver 615.71.09.
Interpreter: `/home/ckl/projects/S/laya/.venv/bin/python`.
Checkpoint: locally cached English `convaiinnovations/laya` snapshot
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; no network downloads.

Each `.command` contains the exact benchmark invocation; the matching `.log` contains
stdout/stderr, all 30 request timings, Dynamo/Inductor cache counters, allocator memory,
and full `nvidia-smi` snapshots before load and after requests. No caches were cleared:
each cold experiment starts with a new directory, and its restart uses the same directory
in a separate process. Three passes over the same ten varying request shapes assert
identical decoded answers within each process. Request 8 is the first single-row request.
Memory columns are PyTorch peak allocated/reserved MiB, not total device usage.

This is a shared desktop GPU. Baseline began at 755 MiB occupied. Other worktrees started
AOTInductor and GPU parity jobs during the automatic-warmup experiments; their processes
are visible in the snapshots. Contended timings are observations, not isolated speedup
estimates. The checkpoint's existing out-of-range temperature warning is retained in logs.
Disk-cache reuse is verified by FX cache counters, independently of timing.

## Item 1: automatic warm-up

`compile=True` warms after runtime/device initialization. `compile_warmup=False` preserves
lazy startup; `agent.warmup()` remains callable. Eager and TileLang paths do not auto-warm.
The constructor regression uses a tiny CPU model with mocked checkpoint I/O; the existing
Dynamo eager-backend check verifies the actual warm-up builds two graphs and later shapes
add none. API defaults are pinned in `tests/test_hooks_api.py`.

Validation commands and full output are in `warmup-validation.log` (all exit codes recorded).
Documentation dependencies were checked with:

```
uv pip install --python /home/ckl/projects/S/laya/.venv/bin/python -r requirements-docs.txt
```

Output: `Using Python 3.12.13 environment at: /home/ckl/projects/S/laya/.venv`;
`Checked 3 packages in 2ms`. Strict docs builds must have no `griffe:` lines.

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc/reserved MiB | FX hits/misses |
|---|---:|---:|---:|---:|---:|
| baseline-cold | 3.078 | 50154.96 | 40468.15 | 1680.0 / 1820.0 | 0 / 2 |
| baseline-restart | 3.124 | 23281.17 | 18366.73 | 1671.0 / 1794.0 | 2 / 0 |
| warmup-cold | 108.498 | 26.61 | 10.91 | 1685.6 / 1820.0 | 0 / 2 |
| warmup-restart | 61.314 | 27.42 | 10.69 | 1671.0 / 1796.0 | 2 / 0 |

Baseline-cold ran before code edits; use `--no-warmup` to reproduce its lazy behavior on this tree.
Warm-up shifts cost into load; it does not eliminate compilation or all possible guard specializations.
The two warm-up runs were contended. Cross-process reuse yielded two FX hits and zero misses.

## Item 2: opt-in persistent Laya cache

`compile_cache=True` selects `$XDG_CACHE_HOME/laya/torchinductor` or
`~/.cache/laya/torchinductor` only if `TORCHINDUCTOR_CACHE_DIR` is absent.
The default is false, existing settings win, and non-compiled/TileLang loads do nothing.
The choice is explicitly process-wide; PyTorch owns cache compatibility/invalidation.

The `cache-verified-*` experiments unset `TORCHINDUCTOR_CACHE_DIR`, set an isolated persistent
XDG root, assert the selected path, and test reuse in separate processes (see `.command` files).
The earlier `cache-cold`, `cache-restart`, and `cache-quiet-*` logs are diagnostic attempts,
NOT measurements of the Laya cache: an early Dynamo counters import populated the default
Inductor environment variable before the opt-in ran. This was caught by auditing the actual
printed path. The benchmark now imports counters after load, asserts the path, and the agent
configures the cache before model construction can import Dynamo. A regression pins that order. Full GPU state is attached to each run, not inferred from
allocator counters. Tests cover opt-in behavior, XDG/fallback paths, explicit overrides,
repeated setup, and constructor forwarding. `cache-verified-validation.log` records every gate and core suite after the correction.
The final ordering assertion is rerun with `python tests/test_compile.py`
(`cache-order-tests.log`, exit 0).

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc/reserved MiB | FX hits/misses |
|---|---:|---:|---:|---:|---:|
| cache-verified-cold | 111.643 | 22.86 | 8.69 | 1685.6 / 1820.0 | 0 / 2 |
| cache-verified-restart | 51.292 | 24.43 | 10.84 | 1671.0 / 1796.0 | 2 / 0 |

Both verified processes selected `/home/ckl/.cache/laya-compile-defaults-verified-20261003/laya/torchinductor`.
The second process hit both FX graphs, reducing load from 111.64 s to 51.29 s.
The cache location adds no meaningful GPU memory requirement; both peaks match the earlier
warm-up experiments. These final runs had no other compute jobs in their GPU snapshots.
Reboot/container persistence was not exercised; that depends on retaining the directory.

## Item 3: opt-in reduce-overhead mode

`compile_mode="reduce-overhead"` forwards that mode to `torch.compile`. The default remains
`"default"`; the active compiled path rejects other values. Laya serializes its CUDA graph
forwards, marks a new step, and clones both outputs before unlocking. Eager, CPU, and
TileLang forwards do not enter that CUDA context. Builds lacking the step-marker API
fail explicitly when this CUDA mode is invoked.

The benchmark now reports CUDA graph node count and allocator memory after each of three
passes, and stores decoded answers so they can be compared across modes. The default-mode
reference reuses the verified cache. Reduced-overhead uses a separate fresh XDG directory
for its cold run and reuses it for its restart. GPU snapshots accompany all runs.

The weight-free CUDA check actually records a graph, changes input values across replays,
retains all returned tensors, and verifies two-thread calls against eager results:

```
TORCHINDUCTOR_CACHE_DIR=/tmp/laya-compile-defaults-cuda-test-20261003 /home/ckl/projects/S/laya/.venv/bin/python tests/test_compile_cuda.py
```

Output: `CUDA graph capture, replay, retained outputs, and two-thread parity passed`.
`mode-cuda-tests.log` records the initial check; the final check is in `mode-cuda-final.log`.
`mode-final-validation.log` records all required lint/compile gates, the compile regression,
API contract, router/criteria/hooks suites, and strict docs build. `mode-regressions.log`
records exact commands and output for batch, predict_batch, runtime_fixes, load_errors,
doc_tables, and portability checks. All exit codes are recorded.

Cross-mode decoded parity is checked with:

```
/home/ckl/projects/S/laya/.venv/bin/python benchmarks/compare_compile_defaults.py benchmarks/results/compile-defaults/mode-default-reference.log benchmarks/results/compile-defaults/mode-cold.log benchmarks/results/compile-defaults/mode-restart.log
```

See `mode-parity.log` for output. Categorical outputs must match, and numerical differences
must be at most 1e-5. The benchmark also asserts exact answer equality on repeated requests
inside each process.

| Run | Load s | Request 1 ms | Request 8 ms | Pass 2 median ms | Pass 3 median ms | Peak alloc/reserved MiB | CUDA graphs | FX hits/misses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mode-default-reference | 46.659 | 22.27 | 8.67 | 12.64 | 12.65 | 1671.0 / 1796.0 | 0 | 2 / 0 |
| mode-cold | 99.089 | 25.73 | 14.69 | 26.94 | 12.41 | 1687.5 / 1892.0 | 10 | 0 / 2 |
| mode-restart | 48.100 | 25.00 | 13.86 | 28.72 | 13.83 | 1669.5 / 1818.0 | 10 | 2 / 0 |

Both reduced-overhead runs exactly match the default-mode decoded answers (max delta 0).
They each record ten CUDA graphs despite having only two Dynamo graphs. The second pass pays
recording costs; graph recordings are recreated after restart even when both FX graphs hit disk.
On the cold run, median replay improves only from 12.65 to 12.41 ms, while recording costs
26.94 ms median. This is not a general latency win, which is why the mode stays opt-in.
Peak reserved memory is 1,892 MiB on the reduced cold run versus 1,820 MiB for the default
cold run; end-of-run reserved memory is 1,842 versus 1,796 MiB. Memory depends on recording
history; the reduced restart peaks at 1,818 MiB. Longer shape sweeps can retain more pools.

## Verification limits

- One English checkpoint, one GPU, torch 2.11.0+cu130, and ten concrete request shapes.
  No multilingual checkpoint, other GPU/OS/torch versions, long-batch OOM stress, or unbounded
  shape sweep was measured. CPU logic is covered by weight-free tests, not a CPU performance run.
- The initial warm-up/cache attempts overlapped other GPU jobs and are identified above.
  Final verified cache and mode GPU snapshots contain only this benchmark plus desktop processes.
  CPU regression suites also ran during some compilation intervals. These are single trials,
  not confidence intervals or an isolated machine throughput study.
- Persistence was checked across fresh processes, not a reboot or container recreation.
  Filesystem persistence remains the deployment's responsibility; PyTorch may invalidate graphs.
- The threaded retained-output regression uses a tiny CUDA model. The real checkpoint is
  checked serially for decoded parity. Application-owned compiled graphs/custom streams are
  outside Laya's lock; callers must coordinate them. OOM fallback under graph-pool pressure was
  not forced.
- No pushes or pull requests were performed. All three changes are local conventional commits.

## Changed source files versus fa9a2a7

Generated benchmark/test output is kept separately in this directory so the implementation
remains reviewable. The nine source, test, benchmark, and documentation files total +420/-25:

| File | Added | Removed |
|---|---:|---:|
| `laya/agent.py` | 37 | 10 |
| `laya/_compile.py` | 26 | 2 |
| `tests/test_compile.py` | 111 | 0 |
| `tests/test_compile_cuda.py` | 62 | 0 |
| `tests/test_hooks_api.py` | 5 | 0 |
| `benchmarks/bench_compile_defaults.py` | 88 | 0 |
| `benchmarks/compare_compile_defaults.py` | 33 | 0 |
| `README.md` | 17 | 11 |
| `docs/compile-and-fast-path.md` | 41 | 2 |

Final required validation (full output in `mode-final-validation.log`):

```
PATH=/home/ckl/projects/S/laya/.venv/bin:$PATH
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
python tests/test_compile.py
python tests/test_hooks_api.py
python tests/test_router.py
python tests/test_criteria.py
python tests/test_hooks.py
zensical build --strict --clean
```

Outputs: `All checks passed!`; compileall silent, exit 0; all 9 compile tests pass;
API 403 passed / 0 failed; router 703 / 0; criteria 198 / 0; hooks 240 / 0;
docs `No issues found`, with no `griffe:` lines. Every command exits 0.
Raw GPU logs preserve `nvidia-smi` trailing spaces; source diffs pass whitespace checks.
