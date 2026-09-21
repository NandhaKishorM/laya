# Docker quickstart

Run the SDK without installing Python or PyTorch on your host. For the CPU
quickstart, allow 8 GB of RAM and 10 GB of free disk, with Docker Engine or
Docker Desktop and Compose v2 or newer.

From the repository root:

```bash
docker compose run --build --rm laya
```

This builds the checkout, runs the [sample request](../examples/docker/request.json)
on CPU and prints JSON covering `choice`, `score` and `noul`. The first request
downloads the selected public Hugging Face checkpoint; no account is needed.
Allow several minutes for its first download.
Weights stay in a named volume. Subsequent runs use `docker compose run --rm laya`.

Predictions and confidence still need evaluation on your workload. See the
[benchmark limits](../BENCHMARKS.md).

## NVIDIA GPU / CUDA

Install a compatible NVIDIA driver and configure Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The GPU image uses PyTorch CUDA 12.8 wheels. Check your GPU's compute capability
and driver against [PyTorch's supported builds](https://pytorch.org/get-started/locally/);
older cards may require a different build. Allow additional disk space for CUDA
layers. VRAM needs depend on the checkpoint, batch size and input length.

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm laya
```

The override selects GPU `0` and defaults to `LAYA_DEVICE=cuda`. Set
`LAYA_GPU_ID` to another host index or UUID. That GPU appears as device `0`
inside the container. Check access without downloading weights:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --rm laya python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu())'
```

The sample rejects unavailable CUDA before loading a checkpoint. Laya can still
fall back to CPU after a memory or inference error, so inspect its warnings.
Rebuild when switching between CPU and CUDA configurations.

This uses [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows requires Docker Desktop's supported WSL2 GPU setup. Apple MPS,
AMD/ROCm and Intel GPU containers are outside this quickstart; use CPU unless
you configure and validate another backend.

## Configuration

Set Compose variables in your shell, a local `.env` file, or the service's
`environment` block. Don't commit secrets in `.env`. Runtime variables also
work with `docker run -e`; Compose-only settings are identified below.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAYA_DEVICE` | `cpu` / `cuda` | Device selected by the base / GPU configuration |
| `LAYA_MODEL` | `auto` | Router alias: `auto`, `english`, `multilingual`, `typed-decisions` |
| `LAYA_MODEL_PATH` | unset | Compatible checkpoint path inside the container |
| `LAYA_REQUEST_FILE` | bundled request | JSON request path inside the container |
| `OMP_NUM_THREADS` | `4` | CPU threads; keep within available cores |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset | Optional Hugging Face credential |
| `HF_HUB_OFFLINE` | `0` | `1` uses only cached checkpoints |
| `HF_HOME` | `/home/laya/.cache/huggingface` | Cache path; see mount requirement below |
| `LAYA_CACHE_VOLUME` | project model cache | **Compose only:** named cache volume |
| `LAYA_GPU_ID` | `0` | **Compose only:** NVIDIA device index or UUID |
| `LAYA_TORCH_INDEX` | `cpu` / `cu128` | **Compose build:** PyTorch wheel index |

Compose forwards the runtime variables except `HF_HOME`, which stays aligned
with its fixed cache mount. If overriding `HF_HOME` in `docker run` or your own
Compose file, provide a matching mount writable by UID 10001. Direct Docker
builds select PyTorch with `--build-arg TORCH_INDEX=cu128`; runtime `-e` cannot
change the installed wheel.

```bash
LAYA_MODEL=english OMP_NUM_THREADS=2 docker compose run --build --rm laya

docker build -t laya:local .
docker run --rm -e LAYA_MODEL=english -e OMP_NUM_THREADS=2 \
  -v laya-model-cache:/home/laya/.cache/huggingface laya:local
```

For your own request:

```bash
docker compose run --rm --volume "$PWD/request.json:/inputs/request.json:ro" \
  --env LAYA_REQUEST_FILE=/inputs/request.json laya
```

For a commented configuration with request, checkpoint and secret-file mounts,
see [`compose.example.yml`](../compose.example.yml):

```bash
docker compose -f compose.yaml -f compose.example.yml run --build --rm laya
```

Add `-f compose.cuda.yaml` before `run` for NVIDIA GPUs. The example is an
override of `compose.yaml`, so cache and image settings stay in one place.

## Secret files

`HF_TOKEN_FILE` reads a mounted UTF-8 file at startup, trims surrounding
whitespace and takes precedence over `HF_TOKEN`. Unreadable, empty or invalid
files stop startup without printing their contents. The file must be readable
by UID 10001. `_FILE` applies only to supported secrets, not every setting.

With `HF_TOKEN_PATH` pointing to an existing host file outside the checkout:

```bash
docker compose run --rm --volume "$HF_TOKEN_PATH:/run/secrets/hf_token:ro" \
  --env HF_TOKEN_FILE=/run/secrets/hf_token laya
```

Docker secrets or Kubernetes Secret volumes can supply the same file. Values
are loaded into the process environment at startup; restart after changing a
file. Never use tokens as build arguments or bake them into images. Public
checkpoints need no token.

## Fine-tuned checkpoints

This image runs inference. Training scripts are tracked in
[#4](https://github.com/NandhaKishorM/laya/issues/4) and
[#26](https://github.com/NandhaKishorM/laya/issues/26); training commands can follow
once that interface is available.

Point `LAYA_CHECKPOINT_PATH` to an absolute host directory containing
`rl_agent_config.json`, `model.safetensors` and matching tokenizer files:

```bash
docker compose run --rm --volume "$LAYA_CHECKPOINT_PATH:/models/custom" \
  --env LAYA_MODEL_PATH=/models/custom laya
```

Use a working copy writable by UID 10001 because the loader may update tokenizer
configuration. A LoRA adapter alone is not a complete checkpoint. Leave
`LAYA_MODEL=auto` when setting `LAYA_MODEL_PATH`; an explicit alias and local path
are mutually exclusive. The local-path response comes from the Agent and has no
Router `routing` metadata. These settings also work with the CUDA override.
Evaluate fine-tuned checkpoints on held-out examples before relying on them.

## Development and cleanup

Open a Python prompt with `docker compose run --rm laya python`. To run the
existing routing/criteria checks and secret-file tests against your checkout
without downloading weights:

```bash
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace laya \
  sh -ec 'python tests/test_router.py; python tests/test_criteria.py; python tests/test_docker_entrypoint.py'
```

Rebuild with `--build` after changing source or the bundled example. The image
runs as UID/GID 10001. New named volumes inherit the image cache directory's
ownership; host directories must be writable by that UID. Keep model caches
writable for tokenizer compatibility updates.

`--rm` removes completed containers. `docker compose down` retains the cache.
To **delete downloaded weights**, run `docker compose down --volumes` using the
same Compose files and `LAYA_CACHE_VOLUME` setting. The next request downloads
them again; don't remove a cache shared with another project.

## HTTP serving

The base quickstart runs the SDK and publishes no ports. HTTP serving is pending
[#3](https://github.com/NandhaKishorM/laya/pull/3) or
[#31](https://github.com/NandhaKishorM/laya/pull/31). The server command and health
probe can follow the interface accepted upstream.
