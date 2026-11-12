FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group embeddings --no-install-project
COPY scambot ./scambot
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --locked --no-dev --group embeddings
RUN useradd --system --uid 10001 scambot
RUN mkdir -p /models && chown scambot /models
ENV HF_HOME=/models/huggingface
USER scambot
