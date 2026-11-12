# Scam Checker Bot

Users forward text, images, video or audio to a Telegram bot (a WhatsApp demo runs on the same
backend). The bot runs a set of independent checks on the message — scam pattern analysis,
fact-check lookup, image provenance, caption context, AI-generated image detection and
synthetic voice detection — combines their results with fixed rules into a single verdict, and
replies in the user's language with links to the evidence.

## Status

Project skeleton: settings, structured logging, tracing, database migrations setup, local
services and CI. Bot features are added incrementally.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker with Compose (only for local Postgres and Redis)

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys you need
docker compose up -d   # Postgres (with pgvector) and Redis
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
uv run pytest            # run the tests
uv run pytest --cov      # run the tests with coverage
```

## Database migrations

```bash
uv run alembic upgrade head                                  # apply all migrations
uv run alembic revision --autogenerate -m "describe change"  # create a new migration
uv run alembic upgrade head --sql                            # print SQL without a database
```

The database URL comes from the `DATABASE_URL` setting.

## Configuration

Secrets and connection settings come from environment variables or a `.env` file (see
`.env.example`). Behaviour — routing, verdict rules, thresholds, scam patterns, sources, prompts
and locales — lives in YAML and prompt files under `config/`.

## Project layout

```
scambot/
├── core/             settings, logging, tracing, shared contracts
├── channels/         Telegram and WhatsApp adapters
├── api/              webhooks, intake, admin
├── worker/           job pipeline, scheduled jobs
├── preprocess/       validation, hashing, OCR, speech-to-text, keyframes, entities
├── agent/            analysis, planning, execution, verdict, reply
├── mcp_server/       checker tools and vendor clients
└── factcheck_index/  fact-check ingest and search
config/               routing, rules, thresholds, patterns, sources, prompts, locales
migrations/           Alembic migrations
tests/                unit and integration tests
eval/                 evaluation datasets and runner
```

## Logging

Logs are written as JSON lines, one event per line, each carrying the `trace_id` of the request
it belongs to. Secrets and message content are redacted before anything is written.
