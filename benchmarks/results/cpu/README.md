# CPU optimization evidence from existing upstream benchmarks

Acceptance claims in this report use benchmark scripts and inputs that already
exist in upstream Laya. Both variants run those scripts unchanged: clean upstream
`3293116c89425c3d04b22aa4d9464fef82e2c858` and PR source
`b4e7c9a434671e6b08daadf30c572270d015cd48`. No private fine-tuned model or private
input is used.

The earlier contributor-added harness mixed 60 upstream parity fixtures with nine
new stress cases and added a separate 33-option suite. Those measurements remain
in [CONTRIBUTOR_DIAGNOSTICS.md](CONTRIBUTOR_DIAGNOSTICS.md) for transparency. They
are not the upstream benchmark evidence used for acceptance claims here.

## Protocol

- `verify/bench_devices.py`: all three official checkpoints, CPU and MPS, and the
  script's complete CPU thread sweep.
- `research/scripts/bench_long_context.py`: all 20 upstream requests in eight
  languages, all five default padding lengths, and both the default and 8192-token
  limits. Every prediction and timing is retained by the existing script.
- `research/scripts/bench_latency.py`: all three checkpoints at 1/5/10/50
  questions, hot/cold routing, and all existing mixed-language workloads.

Script contents, fixtures, warmups and repetition counts are unchanged. Each
variant runs in a fresh process with the same Python environment and official
weights. Scripts run sequentially without competing benchmarks. Device and routing
benchmarks run upstream first; long-context runs the PR first. Initial OMP/MKL
thread counts are four, and the device script performs its own unchanged thread
sweep.

All three checkpoints are pinned to `convaiinnovations/laya` revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`, as specified by upstream
[`verify/checkpoints.json`](../../../verify/checkpoints.json). Their weight hashes
are verified before running. A dedicated offline Hub cache exposes those same
verified files for scripts that load the public model ID directly; model-loading
code is not patched.

## Measured results

Apple M5 Max, macOS 26.5.2, 36 GiB RAM; Python 3.11.12, Torch 2.15.0,
Transformers 5.22.1, Laya 0.4.26. These are sequential workstation measurements
using each upstream script's own statistics. They are descriptive, not confidence
intervals or a production traffic model. Every configuration is retained, including
slower results. Percentage columns are derived as `100 * (1 - PR / upstream)`.

### Existing long-context benchmark

Unchanged upstream script: [`research/scripts/bench_long_context.py`](https://github.com/NandhaKishorM/laya/blob/3293116c89425c3d04b22aa4d9464fef82e2c858/research/scripts/bench_long_context.py).
The script uses the official multilingual checkpoint, 20 requests in eight languages,
five padding lengths and two token limits. It warms each configuration once and
reports the median across its 20 requests.
The predictions and source-computed statistics are retained in
[upstream JSON](upstream/bench_long_context-baseline.json) and [PR JSON](upstream/bench_long_context-pr.json).

| Filler tokens | Limit | Median tokens read | Upstream (s) | PR (s) | Reduction | Correct (both) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1024 | 61 | 0.035 | 0.035 | 0.00% | 19/20 |
| 0 | 8192 | 61 | 0.035 | 0.036 | -2.86% | 19/20 |
| 1000 | 1024 | 1024 | 0.454 | 0.370 | 18.50% | 13/20 |
| 1000 | 8192 | 1032 | 0.470 | 0.401 | 14.68% | 16/20 |
| 2000 | 1024 | 1024 | 0.453 | 0.376 | 17.00% | 7/20 |
| 2000 | 8192 | 2040 | 1.218 | 0.897 | 26.35% | 17/20 |
| 4000 | 1024 | 1024 | 0.476 | 0.364 | 23.53% | 7/20 |
| 4000 | 8192 | 3972 | 4.053 | 2.146 | 47.05% | 18/20 |
| 7000 | 1024 | 1024 | 0.558 | 0.394 | 29.39% | 7/20 |
| 7000 | 8192 | 6912 | 10.787 | 4.836 | 55.17% | 8/20 |

All 200 case configurations preserve the choice, correctness and tokens read.
The maximum difference in a returned probability is 0.0001 (one public
rounding unit). This script records rounded public probabilities, not raw logits.
The accuracy decreases at some lengths are present in both variants; this execution
change does not improve model accuracy or remove the original truncation behavior.

### Existing device and thread benchmark

Unchanged upstream script: [`verify/bench_devices.py`](https://github.com/NandhaKishorM/laya/blob/3293116c89425c3d04b22aa4d9464fef82e2c858/verify/bench_devices.py).
The device comparison uses one warmup and ten timed calls. The thread sweep uses
five timed calls per setting, as implemented by the script (its printed heading
still says ten). The native output rounds milliseconds to integers; the full
[upstream](upstream/bench_devices-baseline.log) and [PR](upstream/bench_devices-pr.log)
outputs are retained.

| Checkpoint | Device | Upstream median (ms) | PR median (ms) |
|---|---|---:|---:|
| english | cpu | 243 | 245 |
| multilingual | cpu | 98 | 96 |
| typed-decisions | cpu | 243 | 252 |
| english | mps | 33 | 72 |
| multilingual | mps | 18 | 22 |
| typed-decisions | mps | 38 | 70 |

These short CPU requests show no consistent latency improvement. The CPU
optimization is inactive on MPS. The MPS timings are nevertheless slower in this
run; these measurements alone do not establish the cause.

| English CPU threads | Upstream median (ms) | PR median (ms) |
|---:|---:|---:|
| 4 | 243 | 242 |
| 8 | 253 | 251 |
| 16 | 275 | 296 |
| 28 | 289 | 324 |
| 56 | 382 | 350 |

#### Device repeat diagnostic

The original slower MPS readings prompted two fixed repeat pairs, declared in
[device-repeat-plan.json](upstream/device-repeat-plan.json) before execution.
They run the complete unchanged device script after the original six runs finish,
first PR/upstream, then upstream/PR. These are additional measurements; the
original results above are retained. All device and thread outputs are archived.

| Repeat | Order | Checkpoint | Device | Upstream median (ms) | PR median (ms) |
|---:|---|---|---|---:|---:|
| 1 | PR / upstream | english | cpu | 253 | 273 |
| 1 | PR / upstream | multilingual | cpu | 99 | 97 |
| 1 | PR / upstream | typed-decisions | cpu | 247 | 249 |
| 1 | PR / upstream | english | mps | 50 | 35 |
| 1 | PR / upstream | multilingual | mps | 21 | 19 |
| 1 | PR / upstream | typed-decisions | mps | 68 | 67 |
| 2 | upstream / PR | english | cpu | 246 | 240 |
| 2 | upstream / PR | multilingual | cpu | 96 | 93 |
| 2 | upstream / PR | typed-decisions | cpu | 245 | 240 |
| 2 | upstream / PR | english | mps | 35 | 34 |
| 2 | upstream / PR | multilingual | mps | 21 | 17 |
| 2 | upstream / PR | typed-decisions | mps | 70 | 40 |

Neither repeat reproduces the initial MPS slowdown. Device timings vary
substantially between runs even for identical code; no MPS performance change
is claimed. The short CPU requests still show no consistent improvement.

### Existing inference and routing benchmark

Unchanged upstream script: [`research/scripts/bench_latency.py`](https://github.com/NandhaKishorM/laya/blob/3293116c89425c3d04b22aa4d9464fef82e2c858/research/scripts/bench_latency.py).
Per-model and hot-router comparisons use two warmups and ten timed calls.
The cold-router section records four swaps; each mixed-language configuration
runs the original 100-call workload. The script's complete summaries, including
p95/mean/min, detection overhead and model-load times, are retained in
[upstream JSON](upstream/bench_latency-baseline.json) and [PR JSON](upstream/bench_latency-pr.json).

| Checkpoint | Questions | Upstream p50 (ms) | PR p50 (ms) | Reduction |
|---|---:|---:|---:|---:|
| english | 1 | 207.56 | 196.36 | 5.40% |
| english | 5 | 814.50 | 756.05 | 7.18% |
| english | 10 | 1628.42 | 1504.30 | 7.62% |
| english | 50 | 8162.42 | 7145.27 | 12.46% |
| multilingual | 1 | 80.90 | 68.84 | 14.91% |
| multilingual | 5 | 349.01 | 269.91 | 22.66% |
| multilingual | 10 | 582.26 | 520.79 | 10.56% |
| multilingual | 50 | 2709.19 | 2357.87 | 12.97% |
| typed-decisions | 1 | 205.34 | 195.10 | 4.99% |
| typed-decisions | 5 | 808.58 | 746.38 | 7.69% |
| typed-decisions | 10 | 1591.99 | 1459.85 | 8.30% |
| typed-decisions | 50 | 8230.84 | 7306.57 | 11.23% |

| Hot router | Upstream p50 (ms) | PR p50 (ms) | Reduction |
|---|---:|---:|---:|
| english_1q | 238.44 | 201.37 | 15.55% |
| english_10q | 1897.57 | 1530.01 | 19.37% |
| hindi_1q | 99.07 | 90.59 | 8.56% |
| hindi_10q | 923.90 | 749.16 | 18.91% |

Cold-swap median: 1646.6 ms upstream, 1317.0 ms PR.

| Mixed workload | Upstream mean (ms/call) | PR mean (ms/call) | Reduction |
|---|---:|---:|---:|
| 0%_non_english/max_loaded=1 | 834.7 | 792.6 | 5.04% |
| 0%_non_english/max_loaded=3 | 804.0 | 809.5 | -0.68% |
| 10%_non_english/max_loaded=1 | 789.3 | 764.8 | 3.10% |
| 10%_non_english/max_loaded=3 | 743.4 | 728.1 | 2.06% |
| 30%_non_english/max_loaded=1 | 768.1 | 740.0 | 3.66% |
| 30%_non_english/max_loaded=3 | 684.1 | 668.2 | 2.32% |
| 50%_non_english/max_loaded=1 | 673.3 | 636.2 | 5.51% |
| 50%_non_english/max_loaded=3 | 566.6 | 542.9 | 4.18% |

Loading, eviction and language detection are included where the upstream script
includes them. This CPU forward change does not optimize those operations.
The routing benchmark does not save predictions; decision-preservation evidence
comes from the upstream long-context outputs and existing correctness checks.

## Reproduce

Use clean checkouts of upstream `3293116c89425c3d04b22aa4d9464fef82e2c858`
and PR source `b4e7c9a434671e6b08daadf30c572270d015cd48`, with the same Python
environment. The PR's later evidence-only commits do not change inference code.
Run the following from each checkout in a separate process, sequentially. Preserve
each checkout's output before running the other variant.

```bash
export USE_TF=0 USE_TORCH=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LAYA_DEVICE=cpu
export LAYA_MODELS="$PWD/models"
export LAYA_REVISION=1c5edc17a7acd8701df6fc341c0d179f1c62c982
python verify/checkpoints.py --fetch
```

The device script uses `models/` in its checkout; the routing script uses
`LAYA_MODELS`. Both can share the same verified model directory through a symlink.
The long-context script loads the public Hub ID. Populate the Hub cache with the
same pinned files before timing, then disable network access for model loading:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

files = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
snapshot_download(
    "convaiinnovations/laya",
    revision=os.environ["LAYA_REVISION"],
    allow_patterns=[prefix + name for prefix in ("", "multilingual/", "typed-decisions/")
                    for name in files],
)
PY
export HF_HUB_OFFLINE=1

python -c 'import laya; from pathlib import Path; assert Path(laya.__file__).resolve().parent.parent == Path.cwd()'
python verify/checkpoints.py
python verify/bench_devices.py > bench_devices.log 2>&1
python research/scripts/bench_long_context.py --device cpu --out bench_long_context.json > bench_long_context.log 2>&1
python research/scripts/bench_latency.py > bench_latency.log 2>&1
# Native routing output: research/latency_benchmark_results.json
```

For the recorded run, a dedicated offline Hub cache linked to the already
downloaded, verified checkpoint files; this avoided a second download. The
benchmark code and model-loading functions were unchanged. The device script
also runs MPS on this Mac, while its CPU thread sweep overrides the initial
four-thread setting as upstream specifies.

[plan.json](upstream/plan.json) records both source commits and the SHA-256 hashes
of the unchanged benchmark scripts and checkpoint manifest. Native JSON reports
are copied byte for byte. Console logs replace only local checkout/output paths
with placeholders; their metrics and warnings are preserved. The
[artifact manifest](upstream/manifest.json) records original and archived hashes,
and [completed.json](upstream/completed.json) records each exit status.

## Correctness and project checks

The existing upstream suites remain the correctness gate. After rebasing, all
97 shared suites pass (828 pytest checks passed, 13 skipped); this count includes
the two new CPU regression suites. The new suites exercise optimization-specific
cases in addition to the upstream tests. The inference implementation has not
changed during this benchmark-provenance correction.

- Existing real-checkpoint end-to-end tests: 37 passed; existing ONNX tests:
  2 passed, with their model ID redirected to the verified local checkpoint.
- Existing smoke, raw numerical and edge sweeps pass on all three checkpoints.
- Existing soak checks pass 200 calls, four reload cycles and 48 concurrent calls.
- The existing 100-example MASSIVE English regression gate matches its committed
  baseline: accuracy 0.8200, ECE 0.1382, mean confidence 0.9582.
- Required lint/compilation checks, 173 packaging checks and strict docs build
  pass; no `griffe:` warnings.

[validation.json](validation.json) records the validation runs and log hashes;
the [MASSIVE summary](massive-en-summary.json) records that evaluation's metrics.
The measured PR source `b4e7c9a4` passes all 20 jobs in [fork CI](https://github.com/bahram-po/laya/actions/runs/42121685219),
including Python 3.10–3.13, Windows, ONNX export, packaging, lint and the SDK checks.
The same CPU implementation also passed the earlier [Linux compatibility matrix](https://github.com/bahram-po/laya/actions/runs/42090048166).
[Current fork CI](https://github.com/bahram-po/laya/actions/workflows/ci.yml?query=branch%3Aperf%2Fcpu-selective-head)
shows any later evidence-only commit's status. Upstream Actions require maintainer approval
for this first contribution. Fork results are independent evidence, not an
upstream CI pass. CUDA-specific suites were not run on CPU hosts.

The separate fork [Security workflow](https://github.com/bahram-po/laya/actions/runs/42089928457)
flags pre-existing `subprocess.run` calls in unchanged `laya/video.py`.

```bash
export LAYA_LAB_MODELS="$PWD/models"
python scripts/test_suites.py
python tests/test_local_e2e.py models
python verify/laya_smoke_test.py --models models --device cpu
python verify/numerics_check.py
python verify/edge_sweep.py --models models --device cpu
python verify/soak_check.py --models models --device cpu --calls 200 --cycles 4 --threads 4
python research/eval/laya_eval.py --model models/laya --device cpu --langs en --out fresh.json
python research/evals/check_regression.py fresh.json research/results/eval_english_51_languages.json research/evals/thresholds.json
ruff check laya/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya/ tests/
zensical build --strict --clean
```

The contributor diagnostics retain additional numerical, memory and ablation
measurements, plus the exact ONNX checkpoint-redirection command. They are
supplementary and are excluded from the upstream performance claims above.
