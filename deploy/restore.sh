#!/usr/bin/env bash
# Restore the database from a dump made by backup.sh. Run from the repository root on the host.
# usage: deploy/restore.sh <dump-file> --yes
set -euo pipefail

if [[ $# -ne 2 || "$2" != "--yes" ]]; then
    echo "usage: deploy/restore.sh <dump-file> --yes" >&2
    echo "This replaces the current database contents with the dump." >&2
    exit 2
fi
dump="$1"
if [[ ! -f "$dump" ]]; then
    echo "no such file: $dump" >&2
    exit 1
fi

echo "1/4 stopping api, worker and mcp"
docker compose stop api worker mcp
echo "2/4 restoring $dump"
# pg_restore exits 1 on harmless warnings; carry on to the schema step either way and report it.
restore_status=0
docker compose exec -T postgres sh -c \
    'pg_restore --clean --if-exists --no-owner --no-acl -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    <"$dump" || restore_status=$?
if [[ "$restore_status" -ne 0 ]]; then
    echo "pg_restore exited with status $restore_status; check the messages above" >&2
fi
echo "3/4 aligning the schema"
docker compose run --rm migrate
echo "4/4 starting mcp, api and worker"
docker compose start mcp api worker
echo "done (Redis is not backed up: queued jobs and cached verdicts start empty if it was lost)"
