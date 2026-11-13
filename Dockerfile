# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOAD=never VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app
ARG UV_GROUPS="--group embeddings --group media --group c2pa"
# Linux wheels of torch on PyPI depend on the CUDA packages (several GB). The locked
# dependencies are therefore installed from the CPU-only wheel index instead; the CUDA and
# triton packages are filtered out of the export.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN uv venv /opt/venv
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --locked --no-dev --no-hashes --no-emit-project $UV_GROUPS \
        | grep -Ev '^(nvidia-|triton)' > /tmp/requirements.txt \
    && uv pip install --no-deps --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match -r /tmp/requirements.txt
COPY scambot ./scambot
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
# The project is installed editable: the settings find config/ and .env relative to the package.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps -e .

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --system --uid 10001 --no-create-home scambot \
    && mkdir -p /models /var/lib/scambot/media \
    && chown scambot /models /var/lib/scambot/media
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/models/huggingface
ENV MEDIA_DIR=/var/lib/scambot/media OMP_THREAD_LIMIT=1
WORKDIR /app
USER scambot
EXPOSE 8000 8001
CMD ["uvicorn", "scambot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
