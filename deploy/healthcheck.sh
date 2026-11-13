#!/usr/bin/env bash
# Checks the public /health endpoint and the compose services. Meant for cron every 5 minutes.
# usage: deploy/healthcheck.sh <domain>
# Optional: HEALTHCHECK_PING_URL (dead-man-switch URL; "/fail" is appended on failure) and
# COMPOSE_DIR (directory with docker-compose.yml, default: the current directory).
set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "usage: healthcheck.sh <domain>" >&2
    exit 2
fi
domain="$1"
compose_dir="${COMPOSE_DIR:-.}"
ping_url="${HEALTHCHECK_PING_URL:-}"

fail() {
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) healthcheck FAILED: $*" >&2
    if [[ -n "$ping_url" ]]; then
        curl -fsS --max-time 10 "${ping_url%/}/fail" >/dev/null 2>&1 || true
    fi
    exit 1
}

body="$(curl -fsS --max-time 10 "https://$domain/health")" || fail "no answer from https://$domain/health"
[[ "$body" == *'"status":"ok"'* || "$body" == *'"status": "ok"'* ]] || fail "unexpected health response"

if command -v docker >/dev/null 2>&1 && [[ -f "$compose_dir/docker-compose.yml" ]]; then
    while IFS='|' read -r service health state; do
        [[ -z "$service" ]] && continue
        if [[ "$service" == "migrate" ]]; then
            continue
        fi
        if [[ "$state" != "running" ]]; then
            fail "service $service is $state"
        fi
        if [[ -n "$health" && "$health" != "healthy" ]]; then
            fail "service $service is $health"
        fi
    done < <(cd "$compose_dir" && docker compose ps --all --format '{{.Service}}|{{.Health}}|{{.State}}')
fi

if [[ -n "$ping_url" ]]; then
    curl -fsS --max-time 10 "$ping_url" >/dev/null 2>&1 || true
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) healthcheck ok"
