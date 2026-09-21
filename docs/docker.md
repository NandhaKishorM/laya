# Docker quickstart

Run Laya's Python SDK without installing Python or PyTorch on your host.
You need Docker Engine or Docker Desktop, Docker Compose v2 or newer,
at least 8 GB of available RAM and 10 GB of free disk space for the image,
model cache and loading a checkpoint.

From the repository root:

```bash
docker compose run --build --rm laya
```

This builds the local checkout, runs the request in
[`examples/docker/request.json`](../examples/docker/request.json) on CPU, prints
the result and exits. The example uses `Router.predict()` and all three question
types: `choice`, `score` and `noul`.

The first build installs CPU-only PyTorch. The first request downloads public
checkpoints from Hugging Face; allow several minutes on a cold cache. The
current English loader downloads the whole bundled model repository, including
checkpoints the example does not load into memory. No Hugging Face account is
needed. Downloads stay in the `model-cache` volume, outside the image, and are
reused by subsequent runs:

```bash
docker compose run --rm laya
```

The response includes `answers`, `usage` and `routing`. Predictions vary by
checkpoint; a completed request confirms that inference runs, not that its
answers are accurate or calibrated for your workload. See
[the benchmark limits](../BENCHMARKS.md).

## Try your own code

Open a Python prompt with the installed package and the same model cache:

```bash
docker compose run --rm laya python
```

Or run a script from your checkout:

```bash
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace \
  laya python tests/test_router.py
docker compose run --rm --volume "$PWD:/workspace:ro" --workdir /workspace \
  laya python tests/test_criteria.py
```

Those two suites do not download model weights. The read-only source mount
lets the tests exercise your current edits. After changing source or the
example request, rebuild with `docker compose run --build --rm laya` to update
the installed package and bundled example.

CPU threads default to 4. To change that, for example on a smaller machine:

```bash
OMP_NUM_THREADS=2 docker compose run --rm laya
```

Keep the thread count at or below the physical cores available to Docker, then
measure your workload. The default image uses CPU PyTorch; its latency will differ from the GPU
benchmarks. Use the CUDA override below for an NVIDIA GPU.

## NVIDIA GPU / CUDA

For a Linux NVIDIA GPU host, install a compatible NVIDIA driver and configure
Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The override uses PyTorch's CUDA 12.8 wheels; check your GPU and driver against
[PyTorch's supported builds](https://pytorch.org/get-started/locally/).
CUDA libraries are installed in the image; the host supplies the driver.
Allow additional disk space for the CUDA image and build layers. VRAM needs
vary with the selected checkpoint, batch size and input length.

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm laya
```

This rebuilds with CUDA PyTorch, requests GPU `0` and sets `LAYA_DEVICE=cuda`.
Select another GPU index or UUID with `LAYA_GPU_ID`, for example:

```bash
LAYA_GPU_ID=1 docker compose -f compose.yaml -f compose.cuda.yaml run --build --rm laya
```

Check device visibility without downloading models:

```bash
docker compose -f compose.yaml -f compose.cuda.yaml run --rm laya python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").cpu())'
```

The selected host GPU appears as device `0` inside the container. The sample
fails before downloading models if CUDA is requested but unavailable. Laya's
existing runtime can still fall back to CPU after an allocation or inference
error; inspect its warnings rather than treating a successful response as
proof of GPU execution. Rebuild when switching back to the CPU command.

The GPU request follows [Docker's Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/).
Windows users need Docker Desktop's supported WSL2 GPU setup. Apple MPS,
AMD/ROCm and Intel GPU containers are outside this quickstart; use the CPU
path unless you configure and validate a separate backend.

## Settings and secrets

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAYA_DEVICE` | `cpu` | SDK device; the CUDA override sets `cuda` |
| `LAYA_MODEL` | `auto` | Router alias: `auto`, `english`, `multilingual` or `typed-decisions` |
| `LAYA_REQUEST_FILE` | bundled request | JSON request path inside the container |
| `LAYA_MODEL_PATH` | unset | Load a mounted, compatible checkpoint directly instead of automatic routing |
| `OMP_NUM_THREADS` | `4` | CPU thread limit |
| `HF_HOME` | `/home/laya/.cache/huggingface` | Hugging Face cache; update the volume target if changed |
| `HF_TOKEN` / `HF_TOKEN_FILE` | unset | Optional token for gated or private model downloads |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to use only already-cached checkpoints |
| `LAYA_CACHE_VOLUME` | project model cache | Existing or new named Docker volume used by Compose |
| `LAYA_GPU_ID` | `0` | Host GPU selected by the CUDA Compose override |
| `LAYA_TORCH_INDEX` | `cpu` / `cu128` | PyTorch wheel index selected by the base / CUDA Compose file |

Compose forwards the settings above except `HF_HOME`, which stays aligned with
its fixed cache mount. Set them in your shell, a local `.env` file, or the
service's `environment` block. `LAYA_CACHE_VOLUME` and `LAYA_TORCH_INDEX` are
Compose/build settings; the rest are runtime settings usable with `docker run -e`.
Do not commit a `.env` file containing secrets.

```bash
LAYA_MODEL=english OMP_NUM_THREADS=2 docker compose run --build --rm laya

docker build -t laya:local .
docker run --rm -e LAYA_MODEL=english -e OMP_NUM_THREADS=2 \
  -v laya-model-cache:/home/laya/.cache/huggingface laya:local
```

To use your own request, mount it and set `LAYA_REQUEST_FILE` to its container path:

```bash
docker compose run --rm --volume "$PWD/request.json:/inputs/request.json:ro" \
  --env LAYA_REQUEST_FILE=/inputs/request.json laya
```

`LAYA_MODEL_PATH` and an explicit `LAYA_MODEL` are mutually exclusive. Use the
former for a mounted checkpoint, or the latter to choose a built-in Router alias.
If you change `HF_HOME` in `docker run` or a custom Compose file, also provide a
matching writable cache mount.
For older GPUs, choose a PyTorch build that still supports their compute
capability; a driver that supports CUDA is not sufficient by itself.

`HF_TOKEN_FILE` reads a mounted UTF-8 file at startup, strips surrounding
whitespace and takes precedence over `HF_TOKEN`. An unreadable, empty or invalid
file stops startup without printing its contents. `_FILE` applies only to the
listed secret, not every setting. The file must be readable by UID 10001.
For example, with an existing token file outside the checkout:

```bash
docker compose run --rm \
  --volume "$HF_TOKEN_PATH:/run/secrets/hf_token:ro" \
  --env HF_TOKEN_FILE=/run/secrets/hf_token laya
```

Docker secrets and Kubernetes Secret volumes can provide the same file. Tokens
are read at startup and passed to the process environment; changing a mounted
file requires a new container. Never pass tokens as build arguments or bake them
into an image. Public checkpoints need no token.

## Fine-tuned checkpoints

This is an inference image. Training scripts are tracked upstream in
[#4](https://github.com/NandhaKishorM/laya/issues/4) and
[#26](https://github.com/NandhaKishorM/laya/issues/26); this guide does not assume
a training interface or export format beyond what the existing loader accepts.

To run the sample against a compatible local checkpoint:

```bash
docker compose run --rm \
  --volume "$LAYA_CHECKPOINT_PATH:/models/custom" \
  --env LAYA_MODEL_PATH=/models/custom laya
```

Use an absolute host path containing `rl_agent_config.json`, `model.safetensors`
and the matching tokenizer files. Use a working copy writable by UID 10001:
the loader may update tokenizer configuration. A LoRA adapter alone is not a
complete checkpoint for this loader. With `LAYA_MODEL_PATH`, the response comes
directly from the Agent and has no Router `routing` metadata. The same setting
works with the CUDA override. Evaluate the resulting checkpoint on held-out
examples from your workload before relying on its decisions or confidence.

## Cache and cleanup

The image runs as UID/GID 10001. The named volume receives the cache directory's
ownership on first use. If you replace it with a host directory, make that
directory writable by UID 10001. Keep the cache writable: Laya may update
cached tokenizer configuration for compatibility.

`--rm` removes each completed container while keeping its model cache.
`docker compose down` removes the Compose network and retains the cache.
To also **delete the downloaded weights**:

```bash
docker compose down --volumes
```

The next inference run will download them again.

## HTTP serving

This quickstart runs the SDK and publishes no ports. HTTP serving is being
discussed in [#3](https://github.com/NandhaKishorM/laya/pull/3) and
[#31](https://github.com/NandhaKishorM/laya/pull/31). A server command, port
mapping and health probe can follow once that interface is merged; this image
does not install either proposed server.
