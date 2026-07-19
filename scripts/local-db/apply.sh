#!/usr/bin/env bash
# Local stand-in for `supabase db reset`: recreate the test database, apply the
# Supabase-auth shim, then every migration in supabase/migrations in order.
# DoD check: this script exiting 0 == "fresh db reset applies clean".
set -euo pipefail

DB_NAME="${SAINIK_TEST_DB:-sainik_test}"
cd "$(dirname "$0")/../.."

dropdb --if-exists "$DB_NAME"
createdb "$DB_NAME"

psql -v ON_ERROR_STOP=1 -q -d "$DB_NAME" -f scripts/local-db/shim-auth.sql

for f in supabase/migrations/*.sql; do
  echo "applying $f"
  psql -v ON_ERROR_STOP=1 -q -d "$DB_NAME" -f "$f"
done

echo "OK: $DB_NAME rebuilt with $(ls supabase/migrations/*.sql | wc -l) migrations"
