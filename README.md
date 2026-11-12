# Scam Checker Bot

Users forward text, images, video or audio to a Telegram bot (a WhatsApp demo runs on the same
backend). The bot runs a set of independent checks on the message — scam pattern analysis,
fact-check lookup, image provenance, caption context, AI-generated image detection and
synthetic voice detection — combines their results with fixed rules into a single verdict, and
replies in the user's language with links to the evidence.

## Status

The Telegram bot answers `/start`, `/help` and `/privacy` in English and Hindi, acknowledges
each message with "Checking…" and then edits that message into the verdict. Text messages are
checked for scams: known scam templates (KYC update, lottery, job offer, "digital arrest",
electricity disconnection and more), pressure and threats, requests for payment or an OTP,
phone numbers, UPI IDs, shortened links and look-alike brand domains. Every check is stored in
the database. Fact-checks and the image, video and audio checks are added in later steps.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker with Compose (only for local Postgres and Redis)

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys you need
docker compose up -d postgres redis   # Postgres (with pgvector) and Redis
uv run alembic upgrade head            # create the tables
```

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
uv run pytest            # run the tests
uv run pytest --cov      # run the tests with coverage
```

## Running the bot locally

Create a bot with @BotFather, put its token in `.env` as `TELEGRAM_BOT_TOKEN`, start Postgres
and Redis (`docker compose up -d postgres redis`), apply the migrations
(`uv run alembic upgrade head`), then run these in three terminals:

```bash
uv run python -m scambot.mcp_server.server           # checker server (or: docker compose up -d mcp)
uv run python -m scambot.channels.telegram --poll    # receive messages by long polling
uv run arq scambot.worker.settings.WorkerSettings    # process queued checks
```

The worker reaches the checker server at `MCP_URL` (default `http://localhost:8001/mcp`).
`ANTHROPIC_API_KEY` is optional: without it the scam check uses its rules only and replies use
fixed explanations. `GOOGLE_API_KEY` turns on Safe Browsing checks for links.

Deployments receive updates by webhook instead: `uv run uvicorn scambot.api.main:app` serves
`POST /webhook/telegram` (requests must carry `TELEGRAM_WEBHOOK_SECRET` in the
`X-Telegram-Bot-Api-Secret-Token` header) and `GET /health`.

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
