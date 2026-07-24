# Sainik Sahayak — operations runbook

Day-2 operations for the air-gapped deployment. First-time install is in
[`deploy/airgap/README.md`](deploy/airgap/README.md) (or
[`deploy/airgap/WINDOWS.md`](deploy/airgap/WINDOWS.md) for a Windows host); this
file is what you keep next to the running system.

Throughout, `COMPOSE` is shorthand for the merged compose invocation you run from
the Supabase docker directory:

```bash
cd ~/supabase/docker            # wherever the Supabase self-hosting bundle lives
export COMPOSE_PROJECT_NAME=supabase
COMPOSE="docker compose --env-file .env \
  -f docker-compose.yml \
  -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml"
```

---

## Quick reference

| Task | Command |
|------|---------|
| Status of every service | `$COMPOSE ps` |
| Start everything | `$COMPOSE up -d` |
| Stop everything (keep data) | `$COMPOSE down` |
| Restart one service | `$COMPOSE restart web` |
| Tail logs | `$COMPOSE logs -f web` (or `rag`, `ollama`, `db`) |
| Web liveness | `curl -s http://localhost:3000/api/health` → `{"status":"ok"}` |

`down` stops containers but **keeps the named volumes** (your database and
models). Never add `-v` unless you intend to erase all data.

---

## Users

The web **Admin** console (top-right link, admin/super-admin only) is the normal
path: invite jawans, set their access tier, activate/deactivate, and reset
passwords. New users get a one-time password and are forced through a password
change + authenticator (TOTP) enrollment on first login before any access.

Seed the very first super-admin from the CLI (one time):

```bash
SUPABASE_URL="$SUPABASE_PUBLIC_URL" SUPABASE_SERVICE_ROLE_KEY="<SERVICE_ROLE_KEY>" \
  npm run seed:admin -- --service-number SUP001 --name "Full Name" --rank Col --unit HQ
```

---

## Documents

Add or update the knowledge base by dropping PDFs in `KB_DIR` and ingesting at
the right access tier (`1` = every jawan, higher = more restricted):

```bash
$COMPOSE exec rag python cli.py ingest /kb --tier 2
# re-ingest a changed file (dedup is by SHA-256):
$COMPOSE exec rag python cli.py ingest /kb --tier 2 --reindex
```

You can also upload/re-ingest/delete from the Admin → Documents page. Scanned
pages are OCR'd (Hindi + English) automatically.

---

## Maintenance jobs

Both connect straight to Postgres inside the `rag` container and are safe to run
on a schedule (cron) or by hand.

**Sweep stale uploads.** A document is stuck in `processing` only if the
ingester was killed mid-run; this marks any such orphan `failed` so it stops
showing an endless spinner in the console:

```bash
$COMPOSE exec rag python cli.py sweep-stale --older-than-minutes 60
```

**Purge old data (retention).** Opt-in per table — nothing is deleted unless you
pass a window. Messages cascade with their conversation. `login_attempts` only
feed the short lockout window, so they can be purged aggressively:

```bash
$COMPOSE exec rag python cli.py purge \
  --login-attempts-days 7 \
  --analytics-days 365 \
  --audit-days 730 \
  --conversations-days 365
```

Suggested cron (daily at 02:30): sweep, then purge. Adjust windows to your
unit's retention policy.

---

## Backup & restore

Everything that is expensive to rebuild — the chunked, embedded knowledge base,
users, conversations, and the append-only audit log — lives in Postgres. Back it
up with one command:

```bash
OUT_DIR=/secure/backups ~/chatbot/deploy/airgap/backup.sh
# → /secure/backups/sainik-<timestamp>.dump   (verified custom-format dump)
```

Restore (destructive — stop the app first, then load a dump):

```bash
$COMPOSE stop web rag
FORCE=0 ~/chatbot/deploy/airgap/restore.sh /secure/backups/sainik-<timestamp>.dump
$COMPOSE up -d web rag
```

Back up your **`.env` and key material separately and securely** — a restored
database cannot be served without them, and they must never sit next to the
dump. See [`SECURITY.md`](SECURITY.md).

---

## Secrets

Rotation (`JWT_SECRET`, `ANON_KEY`, `SERVICE_ROLE_KEY`, `RAG_SERVICE_SECRET`) is
documented step-by-step in [`SECURITY.md`](SECURITY.md#rotating-secrets). Rotate
on personnel changes and on any suspected exposure.

---

## Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| A valid in-KB question returns "not found" | The rerank gate (`RERANK_REFUSAL_THRESHOLD`, default 0.35) fired. Check the document answers the question in words a user would type — acronyms matter. Tune only with care; lowering trades safety for coverage. |
| `rag` never becomes healthy | Models still loading on first boot (`start_period` 120s). Watch `$COMPOSE logs -f rag`. If it stays down, the shared secret may be unset — see below. |
| `rag` exits immediately | `RAG_SERVICE_SECRET` is empty or the built-in default. Set a strong secret in `.env` and `$COMPOSE up -d rag`. |
| A document is stuck on "processing" | The ingester was interrupted. Run the **sweep-stale** job, then re-ingest. |
| 6-digit login code always rejected | Host clock drift. TOTP needs correct time — fix the machine's clock. |
| Login blocked after failed attempts | Lockout is per service number for a short window; wait it out or reset via Admin. |
| "no space left on device" | Free disk (old dumps, `docker system prune`), then retry. Deletes still succeed when writes fail. |
| Everything 403s | The `IP_ALLOWLIST` is set and the client IP isn't on it. Adjust the allowlist or connect from an allowed network. |

---

## Gotcha: Docker volume names

Compose names volumes `<project>_<volume>` and the project defaults to the
directory you run compose from. If you staged models or data under one project
name and bring the stack up under another, the containers get **fresh empty
volumes** and your data appears to vanish (it's still in the old volume). Always
`export COMPOSE_PROJECT_NAME=supabase` (as at the top of this file) so the names
are stable across every command. Verify with `docker volume ls`.

---

## Upgrades

Rebuild images on the staging (online) machine, re-run `stage-models.sh`,
transfer the image tars to the air-gapped host, `docker load`, then
`$COMPOSE up -d`. Take a fresh backup first. Database schema changes ship as new
migrations — apply them with `deploy/airgap/apply-migrations.sh` (idempotent).
