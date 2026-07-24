#!/usr/bin/env bash
#
# Sainik Sahayak — air-gap backup.
#
# Dumps the ENTIRE Postgres database. Everything that makes the deployment what
# it is lives there: the knowledge base (documents, chunks, and their 1024-dim
# embeddings), user accounts and profiles, conversations, and the append-only
# audit log. A single custom-format dump restores the whole system with
# restore.sh.
#
# The original PDFs in KB_DIR and the Supabase Storage blobs are NOT included —
# they are re-ingestable inputs, whereas the embeddings in Postgres are the
# expensive, non-reproducible artefact. Back up KB_DIR separately if you want to
# avoid re-ingesting after a bare-metal rebuild.
#
# Secrets (.env, JWT secret, service keys) are intentionally NOT dumped — keep
# them in your own secret store. Without them a restored database cannot be
# served, so protect both.
#
# Usage:
#   deploy/airgap/backup.sh                    # → ./backups/sainik-<ts>.dump
#   OUT_DIR=/secure/backups deploy/airgap/backup.sh
#   DB_CONTAINER=supabase-db POSTGRES_DB=postgres deploy/airgap/backup.sh
#
set -euo pipefail

DB_CONTAINER="${DB_CONTAINER:-supabase-db}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-postgres}"
OUT_DIR="${OUT_DIR:-./backups}"

if ! docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
  echo "ERROR: database container '$DB_CONTAINER' is not running." >&2
  echo "       set DB_CONTAINER to your Postgres container name and retry." >&2
  exit 1
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="${OUT_DIR}/sainik-${ts}.dump"
mkdir -p "$OUT_DIR"

echo "Dumping '${POSTGRES_DB}' from '${DB_CONTAINER}' → ${out}"
# -Fc: custom format (compressed, restorable selectively/in parallel by pg_restore).
docker exec -i "$DB_CONTAINER" \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "$out"

if [ ! -s "$out" ]; then
  echo "ERROR: backup file is empty — dump failed." >&2
  rm -f "$out"
  exit 1
fi

# Integrity: a valid custom-format dump lists its table of contents cleanly.
# Verify with the container's own pg_restore so no host client tools are needed.
if docker exec -i "$DB_CONTAINER" pg_restore --list - < "$out" >/dev/null 2>&1; then
  echo "verified: dump table-of-contents is readable"
else
  echo "ERROR: dump failed integrity check (pg_restore --list)." >&2
  exit 1
fi

bytes="$(wc -c < "$out")"
echo "OK: wrote ${out} (${bytes} bytes)"
echo "Store this file AND your .env / secret keys somewhere safe and offline —"
echo "you need both to restore a working deployment."
