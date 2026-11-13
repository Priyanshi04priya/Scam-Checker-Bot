# Evaluation

This folder measures how well the bot works: scam precision and recall, matching of debunked
claims, old-media detection, "could not verify" behaviour, prompt-injection resistance, vendor
outage fallbacks, caching, media retention, reply grounding, latency, and the AI-image and
synthetic-voice checks.

**The seed data measures plumbing, not accuracy.** Everything in `data/seed/` was written for
this project to check that the harness runs end to end. Do not quote its numbers as results.
Real numbers need the real dataset described below.

## Quick start

All commands run from the repository root (the folder with `pyproject.toml`).

```bash
# Offline run on the seed data: no network, no keys, deterministic.
uv run python -m eval.run --mode offline --suite all --out eval/results/seed
sed -n 1,60 eval/results/seed/report.md

# Threshold tuning on recorded results (dev split picks, test split reports; never edits config).
uv run python -m eval.tune --results eval/results/seed/results.jsonl --out eval/results/tune

# Per-checker accuracy on labelled folders (real/ and fake/ subfolders).
uv run python -m eval.checker_accuracy deepfake --data PATH --out eval/results/deepfake
uv run python -m eval.checker_accuracy voice --data PATH --out eval/results/voice

# Load test summary from a Locust stats file.
uv run python -m eval.loadtest.summarise --csv eval/results/load_stats.csv
```

### Live run (laptop only, real keys)

Live mode untested in CI: it needs real keys, so it cannot run in the automated tests. It reuses
the production checker wiring (`create_deps`). Put the keys in `.env`, then:

```bash
# First run: calls the vendors and records every answer.
uv run python -m eval.run --mode live --data eval/data/real \
    --replay-file eval/results/full.replay.json --out eval/results/full
# Scoring again is free: nothing is sent to a vendor, a missing answer is an error.
uv run python -m eval.run --mode live --data eval/data/real \
    --replay-file eval/results/full.replay.json --replay-only --out eval/results/full2
```

Notes:

- The `factcheck` suite needs the real fact-check index: pass `--database-url postgresql+asyncpg://...`
  (the index must be filled with the backfill command first). Without it the suite is skipped
  and the report says so.
- Live mode imports `FakeAdapter` and `InMemoryStore` from `tests/` (it does not talk to a chat
  platform and needs no database for the request records). The `tests/` folder is not in the
  Docker image, so run live mode from a full checkout.
- `--disable anthropic,google_vision,google_factcheck,google_safe_browsing,sightengine,voice`
  switches vendors off to test the fallback. In live mode the matching key is blanked, which
  exercises the same "missing key" path production uses.
- Cut-offs that act after the checkers (the verdict rules) can be re-tuned from the recorded
  results for free. Cut-offs inside a checker (`scam.signal_min`, `provenance.old_media_days`,
  `deepfake.signal_min`, `voice.signal_min`) change what the checker returns, so re-running them
  needs `eval.tune --rerun`.
- `--today YYYY-MM-DD` is the date the bot treats as "today" (for date logic). Offline mode uses
  a fixed date so its recorded answers match; changing it makes the recorded answers miss and the
  bot falls back to rule-only results.

### Load test

```bash
# Key-less: an in-process stack with fake vendors and simulated vendor latency.
uv run --group loadtest uvicorn eval.loadtest.fake_stack:app --port 8099 &
LOADTEST_FAKE_VENDORS=1 LOADTEST_TOTAL=100 uv run --group loadtest locust \
    -f eval/loadtest/locustfile.py --headless -u 10 -r 10 -t 5m --stop-timeout 60 \
    --host http://127.0.0.1:8099 --csv eval/results/load
uv run python -m eval.loadtest.summarise --csv eval/results/load_stats.csv
```

Fake-mode numbers show that the harness, the queueing and the concurrency behave; they say
nothing about real vendor speed. For a real instance leave out `LOADTEST_FAKE_VENDORS`, set
`LOADTEST_WEBHOOK_SECRET`, point `--host` at the instance (it needs a real bot token and file
access), and read verdict latencies from the database:
`python -m eval.loadtest.summarise --csv ... --database-url ... --since 2026-01-01T00:00:00+00:00`.
The run stops after `LOADTEST_TOTAL` requests (default 100) and `--users` / `-u` sets the
concurrency (10 for the target run).

## What a run writes

`results.jsonl` (one result per item, including the checker results), `results.csv`,
`metrics.json`, `confusion_<suite>.csv`, `errors.csv` (items that missed their expected
outcome), `audit.csv` and `manual_review.csv` (the grounding audit), `report.md`, and
`run_meta.json` (time, mode, hashes of the config and dataset files, how many answers were live,
replayed or offline, token cost, vendor call counts). A report shows each criterion as PASS, FAIL
or INCOMPLETE; INCOMPLETE means the item quota was not reached with real (non-synthetic) items.

## Dataset format

One JSON object per line (JSONL), or a CSV with the same column names. Media paths are relative
to the dataset file's folder.

| field | meaning |
| --- | --- |
| `id` | unique, `a-z0-9_-`, 3 to 40 characters |
| `kind` | `text`, `image`, `video`, `audio` or `voice` |
| `text` | message text or caption (may be empty for media) |
| `media_path` / `media_url` | the file (media items need one); the loader never downloads |
| `suite` | `scam`, `factcheck`, `old_media`, `unverifiable`, `injection`, `genuine` (control images for old-media), `deepfake`, `voice` |
| `expected_label` | one of the six verdict labels |
| `expected_source` | fact-check article URL that must be matched (factcheck suite) |
| `category` | free tag such as `kyc`, `lottery`, `digital_arrest`, `legit_bank_alert` |
| `language` | `en`, `hi` (Devanagari) or `hinglish` (Latin script) |
| `split` | `dev` or `test` |
| `source_note` | required provenance: where the item came from |
| `synthetic` | true for invented items; required when `source_note` starts with `SYNTHETIC` |
| `canary` | injection suite only: a token the injected text asks the bot to print |

Folders: `data/seed/` (committed, synthetic), `data/templates/` (one example row per suite,
marked `SYNTHETIC-PLACEHOLDER`; a dataset under a `real/` folder refuses such rows),
`data/real/` (your real set and media; ignored by git).

## Collection guide for the team

Targets for the real set:

| suite | quota |
| --- | --- |
| debunked claims | 30, at least 10 in Hindi, each with the URL of a real published fact-check in `expected_source` and `source_note` |
| old media | 20 images that have a known earlier publication, plus about 20 genuine recent control images (`suite: genuine`) |
| scam / legitimate | 50 scam and 50 legitimate messages (mix of English, Hindi, Hinglish; include scam-adjacent legitimate ones such as a real bank alert) |
| unverifiable | 30 claims with no verifiable source (rumours, predictions, anecdotes) |
| injection | 20 forwards containing instructions aimed at the bot, each with a unique canary token |
| deepfake / voice | folders `real/` and `fake/` of images and of voice notes (include compressed Telegram voice notes) |

Rules:

- Use only content you are allowed to store. Strip phone numbers and other personal data. Do not
  scrape in ways a site's terms forbid. Keep all media under `eval/data/real/media/`, which git
  ignores.
- Labelling: two people label each item; settle disagreements and note the outcome in
  `source_note`.
- Splits: assign `dev` or `test` by hashing the id (about 50/50) before looking at any result,
  and never move an item afterwards. Tune only on `dev`; report on `test`.
- Cost: run live once with `--replay-file`, then iterate with `--replay-only`.
- The seed set measures plumbing, not accuracy, and says so in every report.

## Files

`schema.py` (dataset model and loaders), `metrics.py` (pure maths), `replay.py` (record and
replay, outage simulation), `backends.py` (offline and live stacks), `tasks.py` (running items
and scoring suites), `audit.py` (grounding audit), `run.py` (the runner), `tune.py` (threshold
tuning), `report.py` (CSV and Markdown), `checker_accuracy.py` (image and voice folders),
`report_template.md` (final report skeleton), `loadtest/` (Locust file, fake stack, summary),
`voice_compare.py` (compare voice detectors).

To rebuild the small recorded-answer file offline mode uses (after changing the seed data):
`uv run python -m eval.replay build-seed`.
