#!/usr/bin/env bash
# One command to stand up the whole offline stack on the air-gapped host:
# load images → restore models → Supabase core → migrations → seed admin →
# app services. Idempotent: safe to re-run (won't re-seed, won't reload).
#
#   SUPABASE_DOCKER_DIR=/path/to/supabase/docker \
#   BUNDLE_DIR=/path/to/airgap-bundle \
#     ./bringup.sh [--slim] [--no-load]
#
# Assumes bootstrap-env.sh has already written $SUPABASE_DOCKER_DIR/.env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../.." && pwd)"
SUPABASE_DOCKER_DIR="${SUPABASE_DOCKER_DIR:?set SUPABASE_DOCKER_DIR to your supabase/docker dir}"
BUNDLE_DIR="${BUNDLE_DIR:-$HERE/airgap-bundle}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-supabase}"
ADMIN_SERVICE_NUMBER="${ADMIN_SERVICE_NUMBER:-SUP001}"
ADMIN_NAME="${ADMIN_NAME:-Super Admin}"

SLIM=0; LOAD=1
for a in "$@"; do case "$a" in --slim) SLIM=1;; --no-load) LOAD=0;; esac; done

ENV_FILE="$SUPABASE_DOCKER_DIR/.env"
[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE — run bootstrap-env.sh first" >&2; exit 1; }
getenv() { grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2-; }
DB_CONTAINER="${DB_CONTAINER:-supabase-db}"
POSTGRES_DB="$(getenv POSTGRES_DB)"; POSTGRES_DB="${POSTGRES_DB:-postgres}"
WEB_PORT="$(getenv WEB_PORT)"; WEB_PORT="${WEB_PORT:-3000}"
PUBLIC_URL="$(getenv SUPABASE_PUBLIC_URL)"

COMPOSE=(docker compose --env-file "$ENV_FILE"
         -f "$SUPABASE_DOCKER_DIR/docker-compose.yml"
         -f "$REPO_DIR/deploy/airgap/docker-compose.airgap.yml")
[ "$SLIM" -eq 1 ] && COMPOSE+=(-f "$REPO_DIR/deploy/airgap/docker-compose.slim.yml")

step() { echo; echo "==> $*"; }

# 1. load images
if [ "$LOAD" -eq 1 ]; then
  for tar in supabase-images.tar.gz app-images.tar.gz; do
    if [ -f "$BUNDLE_DIR/$tar" ]; then
      step "loading $tar"; gunzip -c "$BUNDLE_DIR/$tar" | docker load
    fi
  done
fi

# 2. restore ollama models into the (project-scoped) volume if empty
VOL="${COMPOSE_PROJECT_NAME}_ollama_models"
docker volume create "$VOL" >/dev/null
if [ -f "$BUNDLE_DIR/ollama_models.tgz" ] && \
   ! docker run --rm -v "$VOL":/m alpine sh -c 'ls -A /m | grep -q .' 2>/dev/null; then
  step "restoring ollama models into $VOL"
  docker run --rm -v "$VOL":/root/.ollama -v "$BUNDLE_DIR":/in alpine \
    sh -c 'tar xzf /in/ollama_models.tgz -C /root/.ollama'
fi

# 3. Supabase core first (GoTrue must create auth.users before our migrations)
step "starting Supabase core"
"${COMPOSE[@]}" up -d db kong auth rest storage

step "waiting for Postgres + auth.users"
for i in $(seq 1 60); do
  if docker exec "$DB_CONTAINER" pg_isready -U postgres >/dev/null 2>&1 && \
     docker exec -i "$DB_CONTAINER" psql -U postgres -d "$POSTGRES_DB" -tAc \
       "select to_regclass('auth.users') is not null" 2>/dev/null | grep -q t; then
    echo "   ready."; break
  fi
  [ "$i" -eq 60 ] && { echo "timed out waiting for auth.users" >&2; exit 1; }
  sleep 3
done

# 4. schema / RLS / functions
step "applying migrations"
DB_CONTAINER="$DB_CONTAINER" POSTGRES_DB="$POSTGRES_DB" \
  "$REPO_DIR/deploy/airgap/apply-migrations.sh"

# 5. seed the first super_admin (only if none exists)
existing="$(docker exec -i "$DB_CONTAINER" psql -U postgres -d "$POSTGRES_DB" -tAc \
  "select count(*) from public.profiles where role='super_admin'" 2>/dev/null || echo 0)"
if [ "${existing:-0}" -eq 0 ]; then
  step "seeding super_admin ($ADMIN_SERVICE_NUMBER)"
  SUPABASE_DOCKER_DIR="$SUPABASE_DOCKER_DIR" DB_CONTAINER="$DB_CONTAINER" POSTGRES_DB="$POSTGRES_DB" \
    "$REPO_DIR/deploy/airgap/seed-admin.sh" --service-number "$ADMIN_SERVICE_NUMBER" --name "$ADMIN_NAME"
else
  step "super_admin already exists — skipping seed"
fi

# 6. app services
step "starting ollama + rag + web"
"${COMPOSE[@]}" up -d ollama rag web

# 6b. ensure the LLM model(s) are present. From a staged bundle they were
#     restored into the volume; otherwise pull them now (needs internet — do
#     this during your setup window, before you disconnect the host).
GEN_MODEL="$(getenv GENERATION_MODEL)"
CLASSIFY_MODEL="$(getenv OLLAMA_CLASSIFY_MODEL)"
step "waiting for ollama"
for i in $(seq 1 30); do "${COMPOSE[@]}" exec -T ollama ollama list >/dev/null 2>&1 && break; sleep 2; done
ensure_model() {
  local m="$1"; [ -z "$m" ] && return 0
  if "${COMPOSE[@]}" exec -T ollama ollama list 2>/dev/null | grep -q "$m"; then
    echo "   model present: $m"
  else
    echo "   pulling model: $m  (needs internet; skip by staging it in the bundle)"
    "${COMPOSE[@]}" exec -T ollama ollama pull "$m" \
      || echo "   WARN: could not pull $m (offline?). Load it before using chat."
  fi
}
ensure_model "$GEN_MODEL"
ensure_model "$CLASSIFY_MODEL"

echo
echo "=========================================================="
echo " Sainik Sahayak is up."
echo "   web         : http://<this-host>:$WEB_PORT   (public URL: $PUBLIC_URL)"
echo "   rag/ollama  : internal only"
echo " Next: put PDFs in \$KB_DIR and run"
echo "   ${COMPOSE[*]} exec rag python cli.py ingest /kb --tier 1"
echo "=========================================================="
