# Build and runtime share one base so the copied virtualenv matches its interpreter.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU by default on AMD64 and ARM64. The CUDA override selects cu128 and the
# DGX Spark override selects cu130. Bump TORCH_VERSION deliberately; the check
# below fails the build if the wheel does not match the requested index.
ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0
COPY docker/check_torch.py /opt/check_torch.py
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    && python /opt/check_torch.py "${TORCH_INDEX}" \
    && pip check

WORKDIR /src
COPY pyproject.toml setup.py README.md LICENSE ./
COPY laya/ ./laya/
# The `serve` extra puts `laya-serve` (POST /v1/systemone, GET /health) in the image, so
# the same image can run a one-shot request or serve the Jev-compatible API. It adds
# fastapi and uvicorn only; torch was installed above.
RUN pip install ".[serve]" && pip check

FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="Laya Docker quickstart" \
      org.opencontainers.image.source="https://github.com/NandhaKishorM/laya" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    LAYA_DEVICE=cpu \
    HF_HOME=/home/laya/.cache/huggingface

RUN groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid laya --create-home laya \
    && mkdir -p /home/laya/.cache/huggingface \
    && chown -R laya:laya /home/laya/.cache

COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/laya/LICENSE
COPY examples/docker/ /opt/laya/examples/
COPY docker/entrypoint.py /opt/laya/entrypoint.py
USER laya
WORKDIR /home/laya

ENTRYPOINT ["python", "/opt/laya/entrypoint.py"]
CMD ["python", "/opt/laya/examples/quickstart.py"]
