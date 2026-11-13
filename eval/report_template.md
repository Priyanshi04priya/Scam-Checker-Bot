# Evaluation report: {{project}}

## Summary

{{summary}}

## System under test

- Commit: {{commit}}
- Date: {{date}}
- Configuration file hashes: see `run_meta.json` ({{config_hashes}})
- Models used: {{models}}

## Datasets

| Suite | Items | dev | test | Provenance | Limitations |
| --- | --- | --- | --- | --- | --- |
| Scam / legitimate | {{n_scam}} | | | | |
| Debunked claims | {{n_factcheck}} | | | | |
| Old media (and genuine controls) | {{n_old_media}} | | | | |
| Unverifiable | {{n_unverifiable}} | | | | |
| Injection | {{n_injection}} | | | | |
| Deepfake / voice folders | {{n_media_checkers}} | | | | |

## Results

| Metric | Target | Observed | n | 95% CI | Status |
| --- | --- | --- | --- | --- | --- |
| Scam precision | >= 0.90 | | | | |
| Scam recall | >= 0.80 | | | | |
| Debunked-claim matching | >= 27/30 | | | | |
| Debunked-claim matching precision | >= 0.90 | | | | |
| Old-media accuracy | >= 17/20 | | | | |
| Old-media precision against genuine controls | >= 0.85 | | | | |
| Unverifiable-never-genuine | 0 of 30 | | | | |
| Prompt-injection resistance | 20/20 | | | | |
| Reply grounding (automated, plus manual sample of 100) | 0 unsupported | | | | |
| Vendor-outage fallback | reply for every item, missing check named | | | | |
| Cache latency | 10/10 under 3 s | | | | |
| Latency, text (load test) | p95 <= 15 s | | | | |
| Latency, image (load test) | p95 <= 30 s | | | | |
| Latency, audio and video (load test) | p95 <= 60 s | | | | |
| Media retention | no file left | | | | |
| Deepfake check (per-checker) | reported only | | | | |
| Voice check (per-checker) | reported only | | | | |

The deepfake and voice checks are shown to users as a low-confidence signal only, so they are
reported, not gated.

## Threshold tuning log

For each tuned cut-off: the dev-split results, the value chosen, and the held-out test results
for the chosen and the current values (from `tune_report.md`). Cut-offs were chosen on the dev
split only. {{tuning}}

## Cost per check

{{cost}}

## Error analysis

The most common misclassifications (from `errors.csv`), grouped by cause, with an example for
each. {{errors}}

## Threats to validity

- Dataset size and how the items were collected; language and category coverage.
- Items that may have appeared in a model's training data.
- Vendor behaviour changes over time; results are a snapshot.
- Cut-offs were tuned on a small dev split.
- {{other_threats}}

## Reproduction

```bash
uv run python -m eval.run --mode live --data eval/data/real --replay-file eval/results/full.replay.json --out eval/results/full
uv run python -m eval.run --mode live --data eval/data/real --replay-file eval/results/full.replay.json --replay-only --out eval/results/rescored
uv run python -m eval.tune --results eval/results/full/results.jsonl --out eval/results/tune
uv run python -m eval.checker_accuracy deepfake --data PATH --out eval/results/deepfake
uv run python -m eval.checker_accuracy voice --data PATH --out eval/results/voice
uv run python -m eval.loadtest.summarise --csv eval/results/load_stats.csv
```
