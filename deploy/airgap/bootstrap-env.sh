#!/usr/bin/env bash
# One-shot config generator. Fills the Supabase bundle's .env with strong random
# secrets, mints the anon/service_role API keys, wires in our overlay vars, and
# locks down auth (no signup, no anon). Pure bash + openssl — no Node needed, so
# it runs on the staging machine or the offline host.
#
#   SUPABASE_DOCKER_DIR=/path/to/supabase/docker \
#   SUPABASE_PUBLIC_URL=http://192.168.10.5:8000 \
#   GEN_MODEL=qwen2.5:7b-instruct \
#   ./bootstrap-env.sh
#
# Re-running is safe: existing secrets are kept unless you pass --force.
set -euo pipefail

SUPABASE_DOCKER_DIR="${SUPABASE_DOCKER_DIR:?set SUPABASE_DOCKER_DIR to your supabase/docker dir}"
SUPABASE_PUBLIC_URL="${SUPABASE_PUBLIC_URL:?set SUPABASE_PUBLIC_URL, e.g. http://<host-lan-ip>:8000}"
GEN_MODEL="${GEN_MODEL:-qwen2.5:7b-instruct}"
CLASSIFY_MODEL="${CLASSIFY_MODEL:-}"
OLLAMA_IMAGE="${OLLAMA_IMAGE:-ollama/ollama:0.5.4}"
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

ENV_FILE="$SUPABASE_DOCKER_DIR/.env"
EXAMPLE="$SUPABASE_DOCKER_DIR/.env.example"

if [ ! -f "$ENV_FILE" ]; then
  [ -f "$EXAMPLE" ] || { echo "no .env or .env.example in $SUPABASE_DOCKER_DIR" >&2; exit 1; }
  cp "$EXAMPLE" "$ENV_FILE"
  echo "created $ENV_FILE from .env.example"
fi

# setkv KEY VALUE — replace an existing KEY=... line or append it.
setkv() {
  local k="$1" v="$2"
  if grep -qE "^${k}=" "$ENV_FILE"; then
    # value may contain / and & — use a non-/ delimiter and escape &,\
    local esc=${v//\\/\\\\}; esc=${esc//&/\\&}
    sed -i "s|^${k}=.*|${k}=${esc}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$k" "$v" >> "$ENV_FILE"
  fi
}

# keep an existing secret unless --force; otherwise generate a fresh one.
keep_or_gen() {
  local k="$1" gen="$2" cur
  cur=$(grep -E "^${k}=" "$ENV_FILE" | head -1 | cut -d= -f2- || true)
  if [ "$FORCE" -eq 0 ] && [ -n "$cur" ] && ! printf '%s' "$cur" | grep -qiE 'your-|change|example|super-secret|placeholder'; then
    printf '%s' "$cur"
  else
    printf '%s' "$gen"
  fi
}

b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }
mint_jwt() {  # $1 = role
  local role="$1" now exp header payload data sig
  now=$(date +%s); exp=$((now + 60*60*24*365*10))
  header='{"alg":"HS256","typ":"JWT"}'
  payload="{\"role\":\"${role}\",\"iss\":\"supabase\",\"iat\":${now},\"exp\":${exp}}"
  data="$(printf '%s' "$header" | b64url).$(printf '%s' "$payload" | b64url)"
  sig=$(printf '%s' "$data" | openssl dgst -sha256 -hmac "$JWT_SECRET" -binary | b64url)
  printf '%s.%s' "$data" "$sig"
}

# --- secrets (kept across re-runs unless --force) ---
POSTGRES_PASSWORD=$(keep_or_gen POSTGRES_PASSWORD "$(openssl rand -hex 24)")
JWT_SECRET=$(keep_or_gen JWT_SECRET "$(openssl rand -hex 32)")
RAG_SERVICE_SECRET=$(keep_or_gen RAG_SERVICE_SECRET "$(openssl rand -hex 32)")
DASHBOARD_PASSWORD=$(keep_or_gen DASHBOARD_PASSWORD "$(openssl rand -hex 12)")
export JWT_SECRET

ANON_KEY=$(mint_jwt anon)
SERVICE_ROLE_KEY=$(mint_jwt service_role)

# --- write everything ---
setkv POSTGRES_PASSWORD  "$POSTGRES_PASSWORD"
setkv JWT_SECRET         "$JWT_SECRET"
setkv ANON_KEY           "$ANON_KEY"
setkv SERVICE_ROLE_KEY   "$SERVICE_ROLE_KEY"
setkv DASHBOARD_PASSWORD "$DASHBOARD_PASSWORD"
setkv SUPABASE_PUBLIC_URL "$SUPABASE_PUBLIC_URL"
setkv API_EXTERNAL_URL   "$SUPABASE_PUBLIC_URL"
setkv SITE_URL           "$SUPABASE_PUBLIC_URL"
# invite-only, MFA-enforced posture
setkv DISABLE_SIGNUP        "true"
setkv ENABLE_ANONYMOUS_USERS "false"
# our overlay vars
setkv RAG_SERVICE_SECRET       "$RAG_SERVICE_SECRET"
setkv GENERATION_MODEL         "$GEN_MODEL"
setkv OLLAMA_CLASSIFY_MODEL    "$CLASSIFY_MODEL"
setkv RERANK_REFUSAL_THRESHOLD "0.35"
setkv IP_ALLOWLIST             ""
setkv WEB_PORT                 "3000"
setkv OLLAMA_IMAGE             "$OLLAMA_IMAGE"
setkv KB_DIR                   "${KB_DIR:-./kb}"

echo "wrote $ENV_FILE"
echo "  SUPABASE_PUBLIC_URL = $SUPABASE_PUBLIC_URL"
echo "  GENERATION_MODEL    = $GEN_MODEL${CLASSIFY_MODEL:+  (classify: $CLASSIFY_MODEL)}"
echo "  secrets + anon/service_role keys generated (10-year keys)."
echo
echo "Next: build the web image with these keys (stage-models.sh reads NEXT_PUBLIC_*),"
echo "or just run bringup.sh on the offline host."
