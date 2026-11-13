#!/usr/bin/env bash
# Database backup. No argument: one backup now. --loop: one backup every day at BACKUP_AT_UTC.
# Connection settings come from the libpq variables PGHOST, PGUSER, PGPASSWORD and PGDATABASE.
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/backups}"
BACKUP_KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"
BACKUP_AT_UTC="${BACKUP_AT_UTC:-21:30}"

log() { printf '%s backup: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

check_keep_days() {
    if ! [[ "$BACKUP_KEEP_DAYS" =~ ^[1-9][0-9]*$ ]]; then
        log "BACKUP_KEEP_DAYS must be a positive integer"
        return 1
    fi
}

backup_once() {
    # errexit is off inside a function called from `||`, so the check must return explicitly.
    check_keep_days || return 1
    mkdir -p "$BACKUP_DIR"
    local stamp tmp final
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    tmp="$BACKUP_DIR/.tmp-$stamp.dump"
    final="$BACKUP_DIR/scambot-$stamp.dump"
    if ! pg_dump --format=custom --no-owner --no-acl --file "$tmp"; then
        rm -f "$tmp"
        log "pg_dump failed"
        return 1
    fi
    if ! pg_restore --list "$tmp" >/dev/null; then
        rm -f "$tmp"
        log "the dump did not pass the integrity check"
        return 1
    fi
    chmod 600 "$tmp"
    mv "$tmp" "$final"
    log "wrote $(basename "$final") ($(wc -c <"$final") bytes)"
    find "$BACKUP_DIR" -maxdepth 1 -name 'scambot-*.dump' -mtime +"$BACKUP_KEEP_DAYS" -delete
    find "$BACKUP_DIR" -maxdepth 1 -name '.tmp-*.dump' -mmin +60 -delete
}

seconds_until_next_run() {
    local now target
    now="$(date -u +%s)"
    target="$(date -u -d "today $BACKUP_AT_UTC" +%s)"
    if ((target <= now)); then
        target="$(date -u -d "tomorrow $BACKUP_AT_UTC" +%s)"
    fi
    echo $((target - now))
}

case "${1:-}" in
    "")
        backup_once
        ;;
    --loop)
        while true; do
            wait_s="$(seconds_until_next_run)" || wait_s=3600
            log "next backup in ${wait_s}s"
            sleep "$wait_s"
            backup_once || log "backup failed, will retry at the next scheduled time"
        done
        ;;
    *)
        echo "usage: backup.sh [--loop]" >&2
        exit 2
        ;;
esac
