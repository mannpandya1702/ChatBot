#!/usr/bin/env bash
# Apply the Sainik Sahayak migrations to the self-hosted Supabase Postgres.
#
# Run this AFTER the Supabase stack is up and healthy (GoTrue must have created
# auth.users — our schema has foreign keys to it). Idempotent-ish: the migrations
# use `if not exists` / `create or replace` where practical, but treat this as a
# one-time bootstrap on a fresh database.
#
#   DB_CONTAINER=supabase-db POSTGRES_DB=postgres ./apply-migrations.sh
#
# Requires: docker access to the running db container. No external psql client
# and no network needed — everything runs inside the container.
set -euo pipefail

DB_CONTAINER="${DB_CONTAINER:-supabase-db}"
POSTGRES_DB="${POSTGRES_DB:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-$HERE/../../supabase/migrations}"

if ! docker exec "$DB_CONTAINER" true 2>/dev/null; then
  echo "ERROR: cannot reach db container '$DB_CONTAINER'." >&2
  echo "Set DB_CONTAINER to the running Postgres container name (docker ps)." >&2
  exit 1
fi

# Sanity: auth.users must exist, or the FKs in migration 2 will fail.
if ! docker exec -i "$DB_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "select to_regclass('auth.users') is not null" | grep -q t; then
  echo "ERROR: auth.users not found. Bring up the Supabase stack and let GoTrue" >&2
  echo "       finish its migrations before applying these." >&2
  exit 1
fi

shopt -s nullglob
files=("$MIGRATIONS_DIR"/*.sql)
if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: no .sql files in $MIGRATIONS_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  echo ">>> applying $(basename "$f")"
  docker exec -i "$DB_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -v ON_ERROR_STOP=1 -q < "$f"
done

echo
echo "All migrations applied. Verifying RLS is enabled on every public table..."
docker exec -i "$DB_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "
  select count(*) filter (where not rowsecurity) as rls_disabled,
         count(*) as total
  from pg_tables where schemaname = 'public';" \
  | awk -F'|' '{ printf "public tables: %s, RLS-disabled: %s\n", $2, $1;
                 if ($1+0 > 0) { print "WARNING: some public tables have RLS disabled!"; exit 1 } }'
echo "Done."
