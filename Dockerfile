FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY README.md LICENSE pyproject.toml ./
COPY assets ./assets
COPY torchtitan ./torchtitan

RUN python -m pip install --upgrade pip build setuptools wheel \
    && python -m build --wheel --no-isolation

FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
ARG INSTALL_WANDB=0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/torchtitan

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install only the runtime dependencies needed for the training image.
RUN python -m pip install --upgrade pip \
    && python -m pip install --pre torch --index-url ${TORCH_INDEX_URL} \
    && python -m pip install \
        "torchdata>=0.8.0" \
        "datasets>=3.6.0" \
        tensorboard \
        fsspec \
        tyro \
        "tokenizers>=0.15.0" \
        safetensors \
        einops \
        pillow \
    && if [ "${INSTALL_WANDB}" = "1" ]; then python -m pip install wandb; fi

COPY --from=builder /build/dist/*.whl /tmp/
RUN python -m pip install --no-deps /tmp/*.whl \
    && rm -f /tmp/*.whl

CMD ["bash"]
