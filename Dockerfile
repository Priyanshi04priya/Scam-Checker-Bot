FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group embeddings --group media --group c2pa --no-install-project
COPY scambot ./scambot
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --locked --no-dev --group embeddings --group media --group c2pa
RUN useradd --system --uid 10001 scambot
RUN mkdir -p /models /var/lib/scambot/media && chown scambot /models /var/lib/scambot/media
ENV HF_HOME=/models/huggingface MEDIA_DIR=/var/lib/scambot/media OMP_THREAD_LIMIT=1
USER scambot
