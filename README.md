# Scam Checker Bot

Users forward text, images, video or audio to a Telegram bot (a WhatsApp demo runs on the same
backend). The bot runs a set of independent checks on the message — scam pattern analysis,
fact-check lookup, image provenance, caption context, AI-generated image detection and
synthetic voice detection — combines their results with fixed rules into a single verdict, and
replies in the user's language with links to the evidence.

## Status

The Telegram bot answers `/start`, `/help` and `/privacy` in English and Hindi, acknowledges
each message with "Checking…" and then edits that message into the verdict. The optional
WhatsApp demo (see below) does the same with two messages and reply buttons. Text messages are
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

### Voice check

Voice notes and videos with sound are also rated for a synthetic or cloned voice. The rating is
an uncertain signal: it can raise the risk, it never says a voice is genuine, and the reply only
adds a cautious note.

```bash
uv sync --group voice --group media     # torch and transformers (large downloads)
```

For Docker, build with `UV_GROUPS="--group embeddings --group media --group c2pa --group voice"`;
the image grows by the torch wheels, so consider a CPU-only torch index on the VM. Without the
`voice` group the check is reported as unavailable and everything else still runs. Settings:
`VOICE_BACKEND` (`huggingface`, `reality_defender` or `none`), `VOICE_MODEL_ID`,
`VOICE_MODEL_REVISION` (pin a commit hash once the model is chosen), `VOICE_CPU_THREADS`.
The hosted `reality_defender` backend is only a stub today; using one would send the audio to a
third party, so the privacy notice must name it before it is enabled. The local model sends
nothing out.

#### Voice detector evaluation

The default model id (`MelodyMachine/Deepfake-audio-detection-V2`) is provisional. The
comparison of two or three open anti-spoofing models has NOT been run yet: it needs model weights
and labelled audio that are not part of the automated tests. Run it on a machine with network
access:

```bash
uv sync --group voice --group media
python -m eval.voice_compare --data DIR --candidate a=MODEL_ID_1 --candidate b=MODEL_ID_2 \
    --out results.csv --scores-out scores.csv
```

`DIR` holds `real/` and `fake/` folders (or pass `--manifest labels.csv`). The output has
accuracy, equal error rate and latency per model; use it to choose the model and set
`voice.signal_min` and the verdict rule. Measuring latency: read the `mean_latency_s` and
`p95_latency_s` columns, and the `duration_ms` of the `job_replied` log line for whole jobs.

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

## WhatsApp demo (optional)

The same backend can answer on WhatsApp through the Cloud API. It is on when
`WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` are set; `WHATSAPP_APP_SECRET` is then required,
because every webhook call is verified with it.

1. Create a Meta developer account, then an app of type Business with the WhatsApp product.
   API Setup gives a free test number and its Phone number ID; add up to 5 recipient numbers
   and verify each with the code WhatsApp sends.
2. App settings, Basic: copy the App secret into `WHATSAPP_APP_SECRET`.
3. Business settings, Users, System users: add a system user (admin), assign the app with full
   control and generate a token that never expires with `whatsapp_business_messaging` and
   `whatsapp_business_management`. Put it in `WHATSAPP_TOKEN` (the temporary token on the
   dashboard expires after about 24 hours).
4. Pick a random `WHATSAPP_VERIFY_TOKEN`, and set `WHATSAPP_PHONE_NUMBER_ID` and
   `WHATSAPP_ALLOWLIST` (digits only, for example `919800000001`). Optionally set
   `WHATSAPP_API_VERSION` to the newest version shown in the Meta app dashboard.
5. Expose the API: `cloudflared tunnel --url http://localhost:8000` (or ngrok). The webhook URL
   is `https://<host>/webhook/whatsapp`.
6. WhatsApp, Configuration, Webhook: enter that URL and the verify token, choose Verify and
   save, then subscribe to the `messages` field (Manage).
7. Run the API, worker and checker server as above, and send "hi" from an allowed phone.

Demo script: (a) send "help", then "language hindi"; (b) forward the electricity-disconnection
scam text: an acknowledgement arrives first, then the verdict with Details, Helpful and Wrong
buttons; tap Details; (c) send the same image on WhatsApp and on Telegram to see the same
verdict; (d) send a voice note; (e) message the bot from a number that is not on the allowlist:
nothing happens, and the log shows only `whatsapp_sender_not_allowed` with a hashed id; (f) send
"delete".

Limits: a test number serves about 5 recipients. Free-form replies work only within 24 hours of
the user's last message; outside that window the reply is dropped and logged (no template
messages are sent). A sent WhatsApp message cannot be edited, so the acknowledgement and the
verdict are two messages. Commands are plain words (`help`, `history`, `language hindi`,
`delete`), and short chit-chat is answered with the help text. The bot only checks content;
WhatsApp's terms do not allow general-purpose AI assistants.

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

## Operations

- **Reused verdicts.** A finished verdict is shown again (with a one-line note saying when it was
  made) when the same content arrives within 7 days: an identical file or text, or, for images, a
  near-identical picture (perceptual hash within 8 bits) with the same caption. Verdicts made while
  a service was down, timed-out ones and reused ones are never served. Video and audio match on the
  exact file only; near-duplicate matching on key frames is not implemented.
- **Checker cache.** Each checker's result is also kept in Redis for hours to days, keyed by the
  tool and its input, so the same picture with a different caption only re-runs the caption check.
- **Limits.** 20 checks per user per hour and a block list. Change them at runtime by writing the
  `settings` table (`rate_limit_per_hour`, `cache_window_days`) and, for a block, `users.blocked`;
  the values are cached for 60 seconds. The environment (`RATE_LIMIT_PER_HOUR`,
  `CACHE_WINDOW_DAYS`) and `config/thresholds.yaml` supply the defaults, in that order.
- **Admin pages.** Set `ADMIN_PASSWORD_HASH` (see `.env.example`) to serve `/admin` and `/metrics`
  behind HTTP Basic login; without it they are not mounted. Ten wrong passwords from one address
  lock it out for five minutes. The page changes the limits above and the block list instantly.
- **Placeholder prices.** `costs.vendor_usd_per_call` (per-call vendor prices) and
  `costs.budget_per_check_usd` (the per-check budget the dashboard compares against) in
  `config/thresholds.yaml` are placeholder figures, as are the LLM prices. The team must confirm
  them against the plans actually used before trusting the cost numbers.
- **/delete.** A user can erase everything stored about them. A nightly job deletes records older
  than 90 days.
- **Retries.** A job that fails is retried up to twice, 5 seconds apart, and a job interrupted by
  a stopped or killed worker is run again by the queue; a finished request is never answered twice.

Manual kill test (laptop): start the stack and the bot, send a 40-60 second video and, when the
worker logs `job_started`, run `kill -9` on the worker. Restart it: after about 110 seconds the job
is picked up again (`job_started` with `attempt=2`, then `job_replied`) and the "Checking..."
message is replaced by exactly one verdict. With `kill -TERM` instead the job is re-queued at once.
`select status, count(*) from requests where job_id = '...' group by 1;` should show one `replied` row.

## Project layout

```
scambot/
├── core/             settings, logging, tracing, shared contracts
├── channels/         Telegram and WhatsApp (Cloud API) adapters
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

## Evaluation

The `eval/` folder measures accuracy, grounding, fallbacks and latency; see
[eval/README.md](eval/README.md) for how to run it (an offline run needs no keys) and how the team
collects the real dataset. [DEMO.md](DEMO.md) is the script for a live demonstration.
