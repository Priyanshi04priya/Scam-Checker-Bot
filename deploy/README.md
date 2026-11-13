# Deploying to one VM

This runbook takes a fresh VM to a running bot. Everything runs from `docker-compose.yml` in the
repository root: Caddy (HTTPS), the API, the worker, the checker server (`mcp`), Postgres, Redis
and a daily backup job. Only Caddy publishes ports (80 and 443).

What could not be tested without a server, and so needs your eyes on the first run: the image
build and its size, ARM wheel availability, certificate issuance, the real webhook, the port scan,
healthcheck timings, and backup/restore against a real Postgres. The memory limits and first-boot
time are estimates to tune.

## 1. The VM

Preferred: Oracle Cloud Always Free Ampere A1 (up to 4 OCPU / 24 GB, Ubuntu 24.04 aarch64).
Fallback: any VM with at least 2 vCPU / 8 GB (x86_64), for example one paid with student credit.

- "Out of host capacity" on Oracle is common: retry another availability domain or another time,
  or use the fallback VM.
- Idle Always Free instances can be reclaimed: keep the CPU in use or upgrade the account to pay
  as you go (free shapes stay free).

## 2. Domain

Point an A record at the VM's public IP: a free DuckDNS name, or `<ip-with-dashes>.sslip.io`.
Check with `dig +short $DOMAIN`.

## 3. Firewall

Cloud security list or network security group: allow ingress TCP 22 (your IP only if it is
static), TCP 80 (certificate challenge and redirect) and TCP 443 (UDP 443 is optional, for
HTTP/3). Nothing else.

Oracle's Ubuntu images also ship iptables rules that block everything except SSH:

```bash
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

(or `ufw allow 22,80,443/tcp`). Docker-published ports bypass ufw; only Caddy publishes any.

## 4. Docker

Install Docker Engine and the compose plugin from Docker's official apt repository, then:

```bash
sudo usermod -aG docker $USER      # log out and in again
sudo systemctl enable --now docker # the stack returns after a reboot
docker compose version
```

## 5. Configure

```bash
git clone <repository-url> /opt/scambot && cd /opt/scambot
cp deploy/prod.env.example .env && chmod 600 .env
openssl rand -hex 32    # USER_ID_PEPPER
openssl rand -hex 32    # TELEGRAM_WEBHOOK_SECRET
openssl rand -hex 24    # POSTGRES_PASSWORD (hex only: it is embedded in a URL, other characters are rejected at start-up)
openssl rand -hex 24    # REDIS_PASSWORD
```

Fill in `DOMAIN`, `ACME_EMAIL`, the four secrets above, `TELEGRAM_BOT_TOKEN` and the vendor
keys you have. The dashboard is off unless `ADMIN_PASSWORD_HASH` is set:

```bash
docker compose build api
docker compose run --rm api python -m scambot.api.admin.hashpw
```

Paste the hash into `.env` between single quotes (`ADMIN_PASSWORD_HASH='$argon2id$...'`);
inside `.env` that keeps the `$` characters as they are (do not double them there). The
dashboard is only reachable from `ADMIN_ALLOW_CIDRS` (default: nobody remote); add your own
address as `203.0.113.7/32`.

Production refuses to start with a missing bot token, a weak webhook secret or pepper, a
localhost or default-password database, or a malformed admin hash. The `migrate` service checks
this first, so a bad `.env` stops the whole stack before anything else starts.

`docker compose config` prints interpolated secrets: do not paste its output anywhere.

## 6. First boot

```bash
docker compose build api      # 10-25 minutes on ARM; installs CPU-only torch
docker compose up -d
docker compose ps             # migrate exits 0, the others become healthy
curl -I https://$DOMAIN/health
```

Build one service first (`build api`) so five services do not build the same image in parallel.
The first start downloads the embedding model into the `models` volume, so `mcp` has a
five-minute start period and the worker waits for it. Optional fact-check backfill:

```bash
docker compose exec worker python -m scambot.worker.backfill --lookback-days 730
```

The default build has no local voice model: `.env` sets `VOICE_BACKEND=none`. To enable it, set
`UV_GROUPS="--group embeddings --group media --group c2pa --group voice"` in `.env`, set
`VOICE_BACKEND=huggingface`, and rebuild (the image grows by the model's dependencies).

## 7. Register the webhook

After `https://$DOMAIN/health` answers:

```bash
docker compose run --rm api python -m scambot.channels.telegram \
    --set-webhook https://$DOMAIN/webhook/telegram
docker compose run --rm api python -m scambot.channels.telegram --webhook-info
```

`--webhook-info` should show no `last_error_message`. Then send the bot a text, an image and a
voice note from a phone. Registering again is safe and also rotates the secret. Roll back with
`--delete-webhook` (add `--drop-pending` to discard queued updates). Never run `--poll` on this
bot from a laptop: polling removes the webhook. Use a separate development bot.

WhatsApp (optional): callback URL `https://$DOMAIN/webhook/whatsapp`, verify token equal to
`WHATSAPP_VERIFY_TOKEN`, subscribe to the `messages` field, set `WHATSAPP_APP_SECRET`,
`WHATSAPP_ALLOWLIST` and a System User permanent token as `WHATSAPP_TOKEN` (a dashboard token
expires after about 24 hours). Restart after editing `.env`: `docker compose up -d`. Meta needs a
publicly trusted certificate (Let's Encrypt via Caddy is fine) and the handshake only succeeds
once the stack is up.

## 8. Verify from outside

From another machine (not the VM):

```bash
nmap -Pn -p- --min-rate 2000 <ip>          # only 22, 80, 443 open
nmap -Pn -p 5432,6379,8000,8001 <ip>       # closed or filtered
nmap -Pn -sU -p 443 <ip>                   # optional
curl -sI https://$DOMAIN/metrics           # 404
curl -sI https://$DOMAIN/admin/            # 404 from an address not in ADMIN_ALLOW_CIDRS
curl -sI https://$DOMAIN/docs              # 404 (also /openapi.json)
curl -si -X POST https://$DOMAIN/webhook/telegram   # 403 without the secret header
```

On the VM: `ss -tlnp` shows only Caddy's proxy on 80 and 443 (plus sshd), and
`docker compose port postgres 5432` prints nothing.

The checker server is internal-only by network design: it has no published port, sits on the
internal `backend` network plus the outbound `egress` network, Caddy cannot reach it, and it
accepts only the Host values in `MCP_ALLOWED_HOSTS`.

## 9. Updating and rollback

```bash
docker tag scambot-app:latest scambot-app:prev   # keep the running image for rollback
git pull
docker compose build api
docker compose up -d      # migrate runs again; expect ~10 s of webhook errors that Telegram retries
```

Rollback: `git checkout <tag>`, then `APP_TAG=prev docker compose up -d`. Undo a schema change
with `docker compose run --rm migrate alembic downgrade -1` only if that migration supports it;
otherwise restore a backup. More workers: `docker compose up -d --scale worker=3` (mind the RAM).

## 10. Backups and restore

The `backup` service writes `scambot-<UTC time>.dump` files to the `backups` volume every day at
`BACKUP_AT_UTC` (default 21:30 UTC, 03:00 IST) and deletes those older than `BACKUP_KEEP_DAYS`
(default 14) after each successful backup. Redis is not backed up: it holds the queue and cache
only, and its append-only file survives restarts.

Run one now: `docker compose run --rm --entrypoint /bin/bash backup /deploy/backup.sh`.
List: `docker compose run --rm --entrypoint ls backup -l /backups`.

Restore (replaces the current data):

```bash
docker compose cp backup:/backups/scambot-<time>.dump ./restore.dump
deploy/restore.sh ./restore.dump --yes
```

The volume lives on the same disk as the database, so also copy backups off the machine, by hand
or from cron:

```bash
docker run --rm -v scambot_backups:/b -v $PWD:/out alpine tar czf /out/backups.tgz -C /b .
scp backups.tgz you@other-host:   # or the Oracle Object Storage CLI
```

Do a restore drill (restore a dump into a scratch stack) once a quarter.

## 11. Monitoring, logs and disk

- `deploy/healthcheck.sh $DOMAIN` checks `/health` and every compose service; run it from cron:
  `*/5 * * * * cd /opt/scambot && deploy/healthcheck.sh "$DOMAIN" >>/var/log/scambot-health.log 2>&1`.
  Set `HEALTHCHECK_PING_URL` to a dead-man-switch URL (for example healthchecks.io) to be told
  when the checks stop.
- Also add an external monitor (UptimeRobot or Better Stack free tier, 5-minute HTTPS check on
  `/health`): a monitor on the VM cannot see a dead VM.
- Logs: `docker compose logs -f --tail=100 api`; `docker compose ps`; `docker stats --no-stream`.
  Logs rotate (10 MB x 5 per service) and are JSON without message text.
- Disk: `docker system df`, `docker image prune`. No Prometheus runs here, so `/metrics` is
  blocked at the edge. Checker errors already alert the `ADMIN_CHAT_ID` chat.

## 12. ARM notes

The base images (python:3.12-slim-bookworm, caddy:2, redis:7, pgvector/pgvector:pg17) are
multi-arch. Python wheels used: torch (the CPU index has aarch64), ctranslate2 and faster-whisper,
onnxruntime, opencv-python-headless and c2pa-python (manylinux_2_28 aarch64 is in `uv.lock`).
Tesseract eng and hin come from Debian arm64. Build natively on the VM; do not cross-build. If a
package has no arm64 wheel the build fails at the dependency install step: use the x86 VM.

For an 8 GB VM set in `.env`: `WORKER_MEM_LIMIT=2g`, `MCP_MEM_LIMIT=2g`,
`POSTGRES_MEM_LIMIT=768m`, `WHISPER_MODEL=base`. The default limits add up to about 9.7 GB.

## 13. Cost and safety

- Anthropic spend is capped by the per-check budget and rate limit; also set a hard limit in the
  vendor console.
- Secrets live only in `.env` (mode 600), never in compose files or git.
- Keep the VM patched (`unattended-upgrades` is listed below).

## 14. Hardening ideas (not done)

Read-only root filesystem with tmpfs, a shared bearer token for the checker server, a Redis
password, a Caddy rate-limit plugin, fail2ban on SSH, unattended-upgrades.

## 15. Image size

Torch comes from the CPU wheel index and the CUDA packages are excluded when the image is built
(build argument `TORCH_INDEX`, default `https://download.pytorch.org/whl/cpu`). A single image
serves the API, worker and checker server; consider splitting it if `docker images` shows more
than 3.5 GB. Record the measured size here after the first build: _not measured yet_.
