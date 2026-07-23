#!/usr/bin/env bash
# Create the first super_admin on the offline host — no Node/npm required.
# Creates the auth user via the GoTrue admin API and the profile row via psql,
# rolling back the auth user if the profile insert fails (no orphans).
#
#   SUPABASE_DOCKER_DIR=/path/to/supabase/docker \
#     ./seed-admin.sh --service-number SUP001 --name "Full Name" [--rank Col] [--unit HQ]
#
# Reads SERVICE_ROLE_KEY / SUPABASE_PUBLIC_URL / POSTGRES_DB from the bundle .env.
set -euo pipefail

arg() { local n="$1"; shift; while [ $# -gt 0 ]; do [ "$1" = "--$n" ] && { echo "${2:-}"; return; }; shift; done; }
SERVICE_NUMBER="$(arg service-number "$@")"
FULL_NAME="$(arg name "$@")"
RANK="$(arg rank "$@")"
UNIT="$(arg unit "$@")"
[ -n "$SERVICE_NUMBER" ] && [ -n "$FULL_NAME" ] || { echo "need --service-number and --name" >&2; exit 1; }

SUPABASE_DOCKER_DIR="${SUPABASE_DOCKER_DIR:-.}"
ENV_FILE="$SUPABASE_DOCKER_DIR/.env"
getenv() { grep -E "^$1=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-; }
SERVICE_ROLE_KEY="${SERVICE_ROLE_KEY:-$(getenv SERVICE_ROLE_KEY)}"
SUPABASE_URL="${SUPABASE_URL:-$(getenv SUPABASE_PUBLIC_URL)}"
POSTGRES_DB="${POSTGRES_DB:-$(getenv POSTGRES_DB)}"; POSTGRES_DB="${POSTGRES_DB:-postgres}"
DB_CONTAINER="${DB_CONTAINER:-supabase-db}"
[ -n "$SERVICE_ROLE_KEY" ] && [ -n "$SUPABASE_URL" ] || { echo "missing SERVICE_ROLE_KEY / SUPABASE_PUBLIC_URL (set them or SUPABASE_DOCKER_DIR)" >&2; exit 1; }

EMAIL="$(printf '%s' "$SERVICE_NUMBER" | tr '[:upper:]' '[:lower:]')@sainik.internal"
TEMP_PW="$(openssl rand -base64 15 | tr '+/' '-_' | tr -d '=')"

# 1. create the auth user (GoTrue admin API)
resp="$(curl -sS -X POST "$SUPABASE_URL/auth/v1/admin/users" \
  -H "apikey: $SERVICE_ROLE_KEY" -H "Authorization: Bearer $SERVICE_ROLE_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$TEMP_PW\",\"email_confirm\":true}")"
UID_VAL="$(printf '%s' "$resp" | grep -o '"id":"[0-9a-f-]\{36\}"' | head -1 | cut -d'"' -f4)"
[ -n "$UID_VAL" ] || { echo "auth user creation failed:" >&2; echo "$resp" >&2; exit 1; }

# 2. insert the profile (psql). Roll the auth user back on failure.
sql_lit() { printf "%s" "$1" | sed "s/'/''/g"; }
rank_sql="null"; [ -n "$RANK" ] && rank_sql="'$(sql_lit "$RANK")'"
unit_sql="null"; [ -n "$UNIT" ] && unit_sql="'$(sql_lit "$UNIT")'"

if ! docker exec -i "$DB_CONTAINER" psql -U postgres -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -q <<SQL
insert into public.profiles
  (id, service_number, full_name, rank, unit, role, access_tier, is_active, must_change_password)
values
  ('$UID_VAL', '$(sql_lit "$SERVICE_NUMBER")', '$(sql_lit "$FULL_NAME")', $rank_sql, $unit_sql,
   'super_admin', 3, true, true);
SQL
then
  echo "profile insert failed — rolling back auth user $UID_VAL" >&2
  curl -sS -X DELETE "$SUPABASE_URL/auth/v1/admin/users/$UID_VAL" \
    -H "apikey: $SERVICE_ROLE_KEY" -H "Authorization: Bearer $SERVICE_ROLE_KEY" >/dev/null || true
  exit 1
fi

echo "super_admin created."
echo "  service number : $SERVICE_NUMBER"
echo "  login email    : $EMAIL"
echo "  temp password  : $TEMP_PW"
echo "Shown once — first login forces a password change + TOTP enrollment."
