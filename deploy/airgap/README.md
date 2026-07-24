# Sainik Sahayak — Air-Gapped Deployment

This is the deployment for **RESTRICTED / classified / access-controlled**
content. Everything — the database, the embedding and OCR models, the reranker,
and the language model — runs on **your own offline hardware**. No query text
and no document content ever leaves the machine. There is no call to any hosted
API and no data-residency question to answer, because nothing is remote.

Use this path (not the cloud pilot) whenever the knowledge base contains
material that is not cleared for foreign-hosted processing. Authorization to
process such material on this deployment is the operator's information-security
authority's responsibility.

---

## Quick start (near-turnkey)

> **On Windows?** Follow **[`WINDOWS.md`](./WINDOWS.md)** — a beginner-friendly,
> click-by-click version of this (WSL2 + Docker Desktop, provision-then-
> disconnect). The steps below are the OS-agnostic reference.

Three commands, given Docker + the Supabase self-hosting bundle
(§2). The scripts generate every secret and key for you, so you edit almost
nothing. Detailed manual steps and troubleshooting follow in §2 onward.

**A — on a machine with internet:** generate config, then stage images + models.

```bash
# writes all secrets, mints anon/service keys, locks down auth (no signup/anon)
SUPABASE_DOCKER_DIR=/path/to/supabase/docker \
SUPABASE_PUBLIC_URL=http://<offline-host-lan-ip>:8000 \
GEN_MODEL=qwen2.5:7b-instruct \
  deploy/airgap/bootstrap-env.sh

# builds our images + pulls the model(s) into a transferable bundle
SUPABASE_DOCKER_DIR=/path/to/supabase/docker GEN_MODEL=qwen2.5:7b-instruct \
  deploy/airgap/stage-models.sh
```

Copy three things to the offline host: this repo, the Supabase bundle (with its
now-filled `.env`), and `deploy/airgap/airgap-bundle/`.

**B — on the offline host:** one command does the rest.

```bash
SUPABASE_DOCKER_DIR=/path/to/supabase/docker \
BUNDLE_DIR=/path/to/airgap-bundle \
  deploy/airgap/bringup.sh --slim
```

It loads images, restores the models, starts Supabase, applies the
migrations (and asserts RLS on every table), seeds the first super_admin
(**prints a one-time password**), and starts the app. `--slim` skips the
non-essential Supabase services (dashboard, edge functions).

**C — ingest your document:**

```bash
# put your PDF(s) in the folder you set as KB_DIR, then:
cd /path/to/supabase/docker
docker compose -f docker-compose.yml \
  -f /path/to/repo/deploy/airgap/docker-compose.airgap.yml \
  exec rag python cli.py ingest /kb --tier 1
```

That's it — open `http://<offline-host>:<WEB_PORT>` and sign in with the seeded
admin (first login forces a password change + TOTP). Everything below is the
same flow, unpacked, for when you want the detail or hit a snag.

---

## 1. What you get

```
        ┌──────────────────────── offline host (no internet) ────────────────────────┐
        │                                                                             │
 browser│   ┌────────┐   ┌──────────────── Supabase self-hosted ─────────────────┐   │
  ──────┼──▶│  web   │──▶│  kong ──▶ auth (GoTrue, TOTP/AAL2)                     │   │
        │   │(Next15)│   │           rest (PostgREST) ──┐                         │   │
        │   └───┬────┘   │           storage            │                         │   │
        │       │        │           db (Postgres 15 + pgvector) ◀──┐             │   │
        │       │        └────────────────────────────────────────│─────────────┘   │
        │       ▼                                                   │                 │
        │   ┌────────┐   embed / rerank / OCR (BGE-M3,             │ documents+chunks │
        │   │  rag   │───bge-reranker-v2-m3, Tesseract hin+eng)    │ (service path)   │
        │   └────────┘───────────────────────────────────────────▶┘                 │
        │       │                                                                     │
        │       ▼                                                                     │
        │   ┌────────┐   generation + classify/rewrite, fully local                  │
        │   │ ollama │   (e.g. qwen2.5:7b-instruct)                                   │
        │   └────────┘                                                                │
        └─────────────────────────────────────────────────────────────────────────────┘
```

The chat pipeline, the fail-closed retrieval gate, and the deny-by-default RLS
are **identical** to the cloud pilot — only the LLM transport (`LLM_PROVIDER=
ollama`) and the hosting change. The same prompts, the same `[S#]` citation
enforcement, the same NOT_FOUND refusal.

Files in this directory:

| file | role |
|---|---|
| `docker-compose.airgap.yml` | overlay adding `ollama`, `rag`, `web` onto the Supabase stack |
| `.env.airgap.example` | config to append to the Supabase bundle's `.env` |
| `gen-jwt.mjs` | mint `anon` / `service_role` keys offline from your JWT secret |
| `stage-models.sh` | **online** step: pull models + build images → transferable bundle |
| `apply-migrations.sh` | apply `supabase/migrations/*.sql` to the self-hosted DB |
| `README.md` | this runbook |

---

## 2. Prerequisites

- **Two machines:**
  1. a **staging machine with internet** (to pull images and models once), and
  2. the **offline host** (air-gapped) that will actually run the system.
  Both need Docker + the Compose plugin. If your offline host has a one-time
  provisioning window with internet, you can use one machine — but treat the
  pull step as the only online step.
- The **Supabase self-hosting bundle** on the staging machine:
  ```
  git clone --depth 1 https://github.com/supabase/supabase
  cp -r supabase/docker /path/to/supabase-docker
  ```
  We layer onto it rather than reinventing it — it is the tested way to run
  Supabase (Postgres+pgvector, GoTrue MFA, PostgREST, Storage, Kong, Studio)
  offline.
- This repository, available on both machines (it holds the migrations, the two
  Dockerfiles, and the admin-seed script).

---

## 3. Decide configuration (do this first)

Pick these values now; several steps bake them in.

```bash
# a strong DB password and a JWT secret (>= 32 chars)
export POSTGRES_PASSWORD="$(openssl rand -hex 24)"
export JWT_SECRET="$(openssl rand -hex 32)"

# derive the two Supabase API keys from the secret, offline:
node deploy/airgap/gen-jwt.mjs "$JWT_SECRET"
#   ANON_KEY=eyJ...
#   SERVICE_ROLE_KEY=eyJ...

# a shared secret for the rag-service:
export RAG_SERVICE_SECRET="$(openssl rand -hex 32)"

# the LAN URL browsers will use to reach the offline host's Kong (port 8000):
export SUPABASE_PUBLIC_URL="http://<offline-host-LAN-ip>:8000"

# the local model(s) you will run:
export GEN_MODEL="qwen2.5:7b-instruct"       # generation
export CLASSIFY_MODEL="qwen2.5:3b-instruct"  # optional; blank = reuse GEN_MODEL
```

> **Model choice.** Any Ollama chat model works. A 7–8B instruct model
> (qwen2.5, llama3.1) is a reasonable floor for grounded, citation-following
> answers; larger is better if the hardware allows. The generation model must
> follow the "answer only from the numbered sources or reply NOT_FOUND"
> instruction well — test it (§10) before relying on it.

---

## 4. Stage on the connected machine

```bash
GEN_MODEL="$GEN_MODEL" CLASSIFY_MODEL="$CLASSIFY_MODEL" \
SUPABASE_DOCKER_DIR=/path/to/supabase-docker \
NEXT_PUBLIC_SUPABASE_URL="$SUPABASE_PUBLIC_URL" ANON_KEY="<ANON_KEY>" \
  deploy/airgap/stage-models.sh
```

This builds `sainik/rag:local` (BGE-M3 + reranker + Tesseract baked in) and
`sainik/web:local` (standalone Next server), pulls the Ollama runtime and your
model(s), and pulls the upstream Supabase images. It writes
`deploy/airgap/airgap-bundle/`:

```
app-images.tar.gz       sainik/rag, sainik/web, ollama/ollama
ollama_models.tgz       the model blobs
supabase-images.tar.gz  the upstream Supabase images
MANIFEST.txt            exact tags — copy these into your .env
```

The rag image bakes the models in and runs with `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1`, so it never reaches for a model hub at runtime.

---

## 5. Load on the offline host

Copy `airgap-bundle/` and this repo across, then:

```bash
# 1. load every image
gunzip -c airgap-bundle/supabase-images.tar.gz | docker load
gunzip -c airgap-bundle/app-images.tar.gz      | docker load

# 2. restore the Ollama models into the named volume the overlay expects.
#    (Compose prefixes volumes with the project name; create it up front.)
docker volume create supabase_ollama_models   # see note below on the exact name
docker run --rm -v supabase_ollama_models:/root/.ollama -v "$PWD/airgap-bundle":/in \
  alpine sh -c "tar xzf /in/ollama_models.tgz -C /root/.ollama"
```

> **Volume name.** Compose names volumes `<project>_<volume>`. The project name
> defaults to the directory you run compose from (`docker` → `docker_…`) unless
> you set `-p`. Pin it with `export COMPOSE_PROJECT_NAME=supabase` and the volume
> becomes `supabase_ollama_models`. Use the same project name in every command
> below. If you prefer, skip the pre-restore and instead `ollama pull` after
> bring-up **only if** the host briefly has a registry mirror.

---

## 6. Configure and bring up Supabase core

In your Supabase bundle's `docker/.env`:

1. Set `POSTGRES_PASSWORD`, `JWT_SECRET`, `ANON_KEY`, `SERVICE_ROLE_KEY`,
   `DASHBOARD_PASSWORD`, and `SUPABASE_PUBLIC_URL` to the §3 values (replace the
   shipped demo values — do not leave them).
2. Enforce invite-only auth:
   ```
   DISABLE_SIGNUP=true
   ENABLE_ANONYMOUS_USERS=false
   ```
   TOTP MFA is enabled automatically by the air-gap overlay (`docker-compose.airgap.yml`
   sets `GOTRUE_MFA_*` on the `auth` service), so you do not hand-edit it here. Your
   first admin login — which forces TOTP enrolment — is the live confirmation MFA
   works; if enrolment fails there, MFA isn't active (check the `auth` container env).
3. **Append** the contents of `deploy/airgap/.env.airgap.example` and fill in
   `RAG_SERVICE_SECRET`, `GENERATION_MODEL`, `OLLAMA_CLASSIFY_MODEL`,
   `RERANK_REFUSAL_THRESHOLD`, `IP_ALLOWLIST`, `WEB_PORT`, `KB_DIR`.

Bring up **only the Supabase core first**, so GoTrue creates `auth.users`
before our migrations run:

```bash
cd /path/to/supabase-docker
export COMPOSE_PROJECT_NAME=supabase
docker compose up -d
docker compose ps          # wait until db + auth are healthy
```

---

## 7. Apply the schema, RLS, and functions

```bash
# from the repo (adjust DB_CONTAINER if your db container is named differently)
DB_CONTAINER=supabase-db POSTGRES_DB=postgres \
  deploy/airgap/apply-migrations.sh
```

This applies migrations 1–5 in order (extensions + pgvector, schema, RLS,
functions, seed settings) and then asserts **every** `public` table has RLS
enabled. If it prints a WARNING, stop and investigate — deny-by-default is the
security boundary.

---

## 8. Create the first super admin

```bash
# from the repo root
SUPABASE_URL="$SUPABASE_PUBLIC_URL" \
SUPABASE_SERVICE_ROLE_KEY="<SERVICE_ROLE_KEY>" \
  npm run seed:admin -- \
    --service-number SUP001 --name "Full Name" --rank Col --unit HQ
```

It prints a one-time temporary password **once**. First login forces a password
change and TOTP enrollment (AAL2) before any access — enforced by the
middleware, not by convention.

---

## 9. Bring up the app (ollama + rag + web)

```bash
cd /path/to/supabase-docker
docker compose \
  -f docker-compose.yml \
  -f /path/to/repo/deploy/airgap/docker-compose.airgap.yml \
  up -d ollama rag web

docker compose ps          # rag healthy after models load (start_period 120s)
```

`rag` reaches Postgres at `db:5432`; `web` reaches Kong at `kong:8000`, the
rag-service at `rag:8000`, and Ollama at `ollama:11434` — all on the internal
network. Only `web` publishes a host port (`WEB_PORT`).

---

## 10. Ingest the knowledge base and verify

Put the PDF(s) in the folder you set as `KB_DIR`, then ingest at the right
access tier (tier gates who can retrieve it — `1` = all jawans, higher = more
restricted):

```bash
docker compose -f docker-compose.yml \
  -f /path/to/repo/deploy/airgap/docker-compose.airgap.yml \
  exec rag python cli.py ingest /kb --tier 2
#   OK  <file>.pdf: NNN pages, NN chunks, lang=...
```

Ingestion runs entirely inside the `rag` container: PyMuPDF extraction with
Tesseract `hin+eng` OCR fallback for scanned pages, structure-aware chunking,
BGE-M3 embeddings, and an upsert into Postgres. Dedup is by SHA-256; re-ingest a
changed file with `--reindex`.

**Verify end to end** (get a session token for your seeded user, then):

```bash
# grounded question that IS in the KB → expect cited [S#] answer, refused:false
curl -sX POST http://<host>:$WEB_PORT/api/chat \
  -H "Authorization: Bearer <access-token>" -H "Content-Type: application/json" \
  -d '{"conversationId":"<id>","message":"<a question your document answers>"}'

# out-of-scope question → expect the exact NOT_FOUND message, refused:true
curl -sX POST http://<host>:$WEB_PORT/api/chat \
  -H "Authorization: Bearer <access-token>" -H "Content-Type: application/json" \
  -d '{"conversationId":"<id>","message":"What is the capital of France?"}'
```

A grounded answer must carry `[S#]` citations; anything the sources don't
support must come back as NOT_FOUND. If a legitimate in-KB question refuses,
it is usually the **rerank gate** (§11), not a bug.

---

## 11. Operations & tuning

- **Refusals on valid questions.** The pipeline fails closed when the reranker's
  top score is below `RERANK_REFUSAL_THRESHOLD` (default 0.35). This is
  deliberate. If real questions refuse, first check the document actually
  contains the answer *in words a user would search for* (acronyms matter — a
  passage that only spells out a term won't match a query that uses its
  acronym). Adjust the threshold only with care and testing; **lowering it
  trades safety for coverage** and can let ungrounded answers through.
- **Adding documents.** Drop new PDFs in `KB_DIR`, rerun the ingest command.
- **Rotating secrets.** Regenerate `JWT_SECRET` → regenerate both keys with
  `gen-jwt.mjs` → update `.env` → `docker compose up -d` to restart. Rotate
  `RAG_SERVICE_SECRET` the same way.
- **Backups.** Back up the Postgres volume (the KB, users, audit log live there)
  and your `.env`. The audit log is append-only at the database level.
- **Updates.** Rebuild images on the staging machine, re-run `stage-models.sh`,
  transfer, `docker load`, `docker compose up -d`.

---

## 12. Security posture (what this deployment guarantees)

- **No egress.** Generation, classification, embeddings, reranking, and OCR are
  all local. With the host air-gapped, no query or document content can leave.
- **Deny by default.** RLS is enabled on every table and asserted at migration
  time (§7). A user's JWT reads only what their tier allows; retrieval is
  RLS-scoped inside `hybrid_search`, so a lower-tier user cannot retrieve
  higher-tier chunks even via the API directly.
- **Fail closed.** Low retrieval confidence or a citation-less generation ⇒
  NOT_FOUND. The model cannot answer from parametric knowledge.
- **AAL2 everywhere.** The middleware admits no route without a session that has
  completed TOTP MFA and belongs to an active profile.
- **Append-only audit.** `audit_logs` has UPDATE/DELETE revoked at the table
  level; metadata only, never query text.

---

## 13. What is verified vs. what you run

Honest scope, so nothing surprises you:

- **Verified in this repo:** the app code, the migrations (applied to a live
  Postgres 17 and RLS-proven by `scripts/test-rls.ts`), the rag-service
  (`pytest`), the chat pipeline (`vitest` + a live end-to-end smoke), the
  Ollama provider wiring, the `web` standalone build, and this overlay's compose
  merge + interpolation.
- **You run (standard, well-trodden):** the upstream Supabase self-hosting stack
  and the one-time image/model staging. These use Supabase's own tooling and
  Docker's normal load/run flow; this runbook sequences them. Cross-check image
  tags and GoTrue variable names against the Supabase self-hosting docs for the
  bundle version you stage.
