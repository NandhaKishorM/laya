# ARM64 and DGX Spark containers

The Dockerfile builds for Linux AMD64 and ARM64. CPU remains the default on both;
an ARM64 CPU does not imply an NVIDIA GPU.

| Host | Configuration | Validation |
| --- | --- | --- |
| Linux AMD64 CPU | `compose.yaml` | CI build and checks; CPU inference |
| Linux ARM64 CPU | `compose.yaml`, built on the ARM64 host | CI build and checks on a native ARM64 runner; CPU inference |
| Linux AMD64 NVIDIA | add `compose.cuda.yaml` (CUDA 12.8) | CUDA inference on an RTX 4070 Ti with the base quickstart |
| DGX Spark | add `compose.spark.yaml` (ARM64, CUDA 13.0), with or without `compose.http.yaml` | CI builds and loads the CUDA libraries without a GPU; Spark inference not yet reported |
| Apple Silicon | Linux ARM64 container on CPU | See [Apple Silicon](#apple-silicon) |

## ARM64 CPU

Build on the target host. Docker selects its native architecture:

```bash
docker compose run --build --rm laya
```

Cross-builds with `docker buildx build --platform linux/arm64 --load -t laya:arm64 .`
need an ARM64 builder or configured emulation. An emulated build does not show
native inference performance.

## DGX Spark

Use the Spark's Linux host with its supported NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html).
The override selects ARM64, CUDA 13.0 wheels and GPU 0. Use it instead of
`compose.cuda.yaml`, not with it:

```bash
docker compose -f compose.yaml -f compose.spark.yaml run --build --rm laya
```

To serve the HTTP API on the Spark, add `compose.http.yaml`. The
[HTTP serving](docker.md#http-serving) settings apply unchanged:

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.spark.yaml up --build laya-serve
```

Set `LAYA_GPU_ID` to select another device.

`TORCH_VERSION` pins PyTorch for every build, CPU and CUDA alike. Compose reads
`LAYA_TORCH_VERSION`; direct builds take
`--build-arg TORCH_VERSION=2.14.0 --build-arg TORCH_INDEX=cu130`. Changing either
needs a rebuild, because a runtime environment variable cannot replace the
installed wheel.

PyTorch's CUDA 13.0 builds for ARM64 depend on cuSPARSELt 0.8.0 (PyTorch 2.11)
or 0.8.1 (PyTorch 2.14). NVIDIA's AArch64 wheels for those two versions declare
`manylinux2014_sbsa` inside their `WHEEL` file, which `pip check` rejects;
0.9.0 corrects it. The build checks that the library is ELF64 AArch64 and loads,
then corrects that tag and its `RECORD` hash. Any other version with the same
defect fails the build instead of receiving the repair, and `pip check` still
runs. This follows @TheIrritainer's compatibility finding in
[FastLaya](https://github.com/emtay-com/fastlaya).

### Reporting Spark results

CI has no GPU, so Spark inference needs a report from real hardware. Run this on
the Spark and include its output with `nvidia-smi`, the OS and driver versions,
and the image revision:

```bash
docker compose -f compose.yaml -f compose.spark.yaml run --build --rm laya python -c '
import json, platform, torch
from pathlib import Path
from laya import load
assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
agent = load("convaiinnovations/laya", device="cuda")
request = json.loads(Path("/opt/laya/examples/request.json").read_text())
result = agent.predict(request["state"], request["questions"])
assert next(agent.model.parameters()).device.type == "cuda", "fell back to CPU"
assert set(result["answers"]) == set(request["questions"])
print("CUDA inference passed", torch.cuda.max_memory_allocated())
'
```

Repeat with each checkpoint you intend to run. A native ARM64 CPU test does not
establish Blackwell kernel or GPU inference support. Jetson's platform-specific
CUDA stack is not covered by the Spark override.

## Apple Silicon

Apple GPU acceleration needs native macOS PyTorch with MPS. Docker Desktop runs
a Linux container, which has no MPS backend, so the container uses CPU. Laya
already has an MPS device path, and
[PR #51](https://github.com/NandhaKishorM/laya/pull/51) and
[PR #109](https://github.com/NandhaKishorM/laya/pull/109) address MPS
compatibility and performance outside Docker.
