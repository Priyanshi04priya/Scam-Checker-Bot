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
the database. Claims are also matched against published fact-checks: the Google Fact Check
Tools search plus a local index of articles from Indian and international fact-checkers
(`config/sources.yaml`), searched by meaning and by words, so a Hindi forward can match an
English fact-check. The image, video and audio checks are added in later steps.

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

### Media tools

Photos, videos and audio need a few extra tools. Without one of them that step is skipped and
the check continues on the caption and the other text.

```bash
sudo apt install ffmpeg tesseract-ocr tesseract-ocr-hin   # video frames, audio, OCR (English + Hindi)
uv sync --group media                                     # speech-to-text, language detection, scene detection
```

The Whisper speech model (about 0.5 GB) is downloaded on first use. Without Tesseract, text in
images is not read; without `ffmpeg` there is no speech-to-text or video frames; without the
`media` group, speech-to-text is off and the language is guessed from the script.

Image checks (original-source search, caption comparison, AI-image detection) read the
downloaded file from the media directory, so the worker and the checker server must share
`MEDIA_DIR`: in Docker the `media` volume is mounted into the checker server, and a worker on the
host needs the checker server on the host with the same `MEDIA_DIR`. `GOOGLE_API_KEY` also turns
on the image search and `SIGHTENGINE_USER` / `SIGHTENGINE_SECRET` turn on AI-image detection;
without them those checks are reported as unavailable. `uv sync --group c2pa` optionally adds
reading of embedded content credentials.

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
fixed explanations. `GOOGLE_API_KEY` turns on Safe Browsing checks for links and the Fact Check
Tools search.

## Fact-check index

Searching the local index needs the optional embedding dependencies (PyTorch and
sentence-transformers, a large download) and the `intfloat/multilingual-e5-small` model, which
is downloaded on first start:

```bash
uv sync --group embeddings
```

Without them the checker server and worker still run; claims are then matched through the Fact
Check Tools search only. The worker indexes new fact-checks every day at 02:00 IST. To fill the
index with older articles, run a backfill (it uses the same settings):

```bash
uv run python -m scambot.worker.backfill --lookback-days 730
uv run python -m scambot.worker.backfill --source altnews --lookback-days 7   # spot-check one publisher
```

Only pages on each publisher's own site are fetched, robots.txt is honoured and every site gets
a pause between requests. The embeddings group currently installs PyTorch from PyPI, which
pulls in CUDA libraries; switch it to the PyTorch CPU wheel index before building deployment
images.

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
