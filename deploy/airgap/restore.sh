#!/usr/bin/env bash
#
# Sainik Sahayak — air-gap restore.
#
# Loads a backup produced by backup.sh into Postgres. DESTRUCTIVE: it drops and
# recreates every object present in the dump, replacing current data. Stop the
# app first so nothing writes mid-restore:
#
#   docker compose ... stop web rag        # keep db up (we restore into it)
#
# Usage:
#   deploy/airgap/restore.sh ./backups/sainik-<ts>.dump
#   DB_CONTAINER=supabase-db POSTGRES_DB=postgres deploy/airgap/restore.sh <file>
#   FORCE=1 deploy/airgap/restore.sh <file>    # skip the confirmation prompt
#
set -euo pipefail

DB_CONTAINER="${DB_CONTAINER:-supabase-db}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-postgres}"

dump="${1:-}"
if [ -z "$dump" ] || [ ! -f "$dump" ]; then
  echo "usage: restore.sh <path-to-.dump>" >&2
  exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
  echo "ERROR: database container '$DB_CONTAINER' is not running." >&2
  exit 1
fi

if [ "${FORCE:-0}" != "1" ]; then
  echo "About to restore:"
  echo "    ${dump}"
  echo "into database '${POSTGRES_DB}' on container '${DB_CONTAINER}'."
  echo "This OVERWRITES current data. Type 'yes' to continue:"
  read -r confirm
  [ "$confirm" = "yes" ] || { echo "aborted."; exit 1; }
fi

# --clean --if-exists: drop existing objects first so the restore is idempotent.
# --no-owner: don't try to reassign ownership to roles that may differ post-rebuild.
docker exec -i "$DB_CONTAINER" \
  pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    --clean --if-exists --no-owner < "$dump"

echo "restore complete — bring the app back up (docker compose ... up -d web rag)."
