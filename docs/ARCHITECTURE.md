# Sainik Sahayak — Architecture

Production-grade internal chatbot for Indian Army jawans. Answers **only** from an
admin-managed PDF knowledge base with mandatory `[S#]` citations, or refuses with a
canned bilingual NOT_FOUND message. Invite-only, TOTP-MFA (AAL2) enforced, mobile-first,
Hindi/English/Hinglish.

Two non-negotiables shape everything:

1. **Fail closed** — low retrieval confidence or a citation-less generation ⇒ NOT_FOUND.
2. **Deny by default** — RLS on every table; no route without an active AAL2 session.

## Components

| Component | Tech | Where | Role |
|---|---|---|---|
| `apps/web` | Next.js 15 (App Router), React 19, Tailwind, shadcn/ui, Vercel AI SDK | Vercel `bom1` | Jawan chat UI, admin console, all API routes. All model calls server-side. |
| Supabase | Postgres 15 + pgvector, Auth (TOTP MFA), private Storage, RLS | `ap-south-1` | System of record + the actual security boundary. |
| `services/rag` | Python 3.11 FastAPI, PyMuPDF, Tesseract (hin+eng), BGE-M3, bge-reranker-v2-m3 | Fly.io Mumbai / ap-south-1 VM | `/ingest`, `/embed`, `/rerank`, `/health` behind `X-Service-Secret`. |
| LLM | `LLM_PROVIDER=anthropic` (pilot) or `ollama` (air-gapped) | provider-side / local | Generation: Sonnet · classify/rewrite: Haiku · same prompts both providers. |

**Residency note:** Supabase, Vercel and the rag-service all sit in Mumbai regions, which
covers data at rest and app compute. With `LLM_PROVIDER=anthropic`, inference (query text +
retrieved context) is processed on Anthropic infrastructure outside India. Fully in-country
processing = the Ollama path. Classified or unclear material must never use the cloud path
(spec §15 Q1); embeddings and OCR are always local to the rag-service either way.

## Chat pipeline (spec §3)

```
middleware (session + AAL2 + is_active + rate limit)
  → zod validate (≤2000 chars)
  → classify (Haiku): greeting | kb_question | out_of_scope | injection_attempt
  → rewrite to standalone query (Haiku, last 6 messages)
  → rag-service /embed
  → hybrid_search RPC (HNSW cosine + tsvector, RRF k=60, top 20, SECURITY INVOKER ⇒ RLS tier-scoped)
  → rag-service /rerank (→ top 5, sigmoid scores)
  → best < rerank_refusal_threshold ⇒ NOT_FOUND + record_query_event(refused)
  → stream Sonnet with §7 system prompt (context wrapped as untrusted data)
  → post-check: no [S#] in final text ⇒ replace with NOT_FOUND
  → persist message + citations + latency · log_event · record_query_event
```

## Data model (migrations 2–5)

`profiles`, `documents`, `chunks` (vector(1024), generated tsvector, HNSW + GIN),
`conversations`, `messages`, `audit_logs` (append-only), `app_settings`, plus:

- **`login_attempts`** — server-side lockout tracking (5 fails ⇒ 15 min), service-role only.
- **`query_analytics`** — resolves the spec §4/§10 tension: admins need "top queries" and
  the unanswered-queries gap backlog, but audit stores no query text and chat is owner-only.
  This table stores the **rewritten standalone query with no user linkage** (write path:
  `record_query_event()`, service-role only). Log minimisation holds; insights work.

## RLS matrix (enforced by `supabase/migrations/…rls.sql`, proven by `scripts/test-rls.ts`)

| table | anon | jawan | admin | super_admin | service role |
|---|---|---|---|---|---|
| profiles | — | own row (R) | all (R), `is_active` on jawans (U) | all (R/U; role+tier edits) | full |
| documents | — | ready ∧ tier ≤ own (R) | same as jawan¹ | same¹ | full |
| chunks | — | ready ∧ tier ≤ own (R) | same | same | full |
| conversations | — | owner CRUD | — | — | full |
| messages | — | owner R + feedback-only U | — | — | full |
| audit_logs | — | write via `log_event()` only | R | R | full² |
| app_settings | — | R | R | R/W | full |
| query_analytics | — | — | R | R | write via fn |
| login_attempts | — | — | — | — | full |

¹ Admin-console listings (incl. `processing`/`failed`) read via service role server-side.
² `UPDATE`/`DELETE` on `audit_logs` are revoked at the table level for client roles —
append-only is a DB guarantee, not an app convention. Trigger guards additionally pin
client profile updates (role/tier = super_admin only, immutable identity columns) and
client message updates (feedback column only). A deactivated user's JWT reads nothing
even if it reaches PostgREST directly — `is_active` is checked inside RLS, not just
middleware.

## Phase log

- **Phase 0 — scaffold + schema: DONE, and LIVE on hosted Supabase.** Migrations apply
  clean on a fresh database (`scripts/local-db/apply.sh`, stand-in for `supabase db
  reset` — the remote build environment has no Docker, so the Supabase-auth surface the
  migrations rely on is shimmed by `scripts/local-db/shim-auth.sql`; migrations
  themselves are untouched Supabase SQL). RLS matrix green via `npm run test:rls`
  (61/61). Seed script ready (`npm run seed:admin`).
  - **Applied to the hosted project** (`ap-south-1`, **PostgreSQL 17.6** — Supabase's
    current default; migrations are 15/16/17-compatible) over the Management API
    (this sandbox has HTTPS egress only; direct Postgres TCP is blocked). Verified
    live: 9 tables all RLS-enabled (0 disabled), 13 policies, 6 settings seeded,
    pgvector in `public`, HNSW + GIN indexes present, audit_logs INSERT/UPDATE/DELETE
    revoked. Deny-by-default confirmed at the real API boundary (anon → 401 on
    profiles/app_settings; service_role → 200) and the authenticated path proven via a
    create→sign-in→read→cleanup cycle (super_admin saw only its own row; test user
    deleted, DB left clean). Canonical apply path for operators is `supabase db push`.
- **Phase 1 — rag-service: code-complete, verified on synthetic fixtures.**
  FastAPI service (`/health`, `/embed`, `/rerank`, `/ingest`, all behind
  `X-Service-Secret`), PyMuPDF extraction with Tesseract `hin+eng` OCR fallback
  (<40 chars/page), structure-aware chunker (target 600 / hard-max 900 / 15%
  overlap / tables never split / heading breadcrumb), pluggable embed+rerank
  backends (`bge` real · `hashing` deterministic double), Supabase-Storage and
  local file sources, bulk `cli.py ingest`, and a Dockerfile that bakes in the
  models. `pytest` = 19 passed incl. an end-to-end ingest into real
  Postgres+pgvector (chunks + 1024-dim embeddings asserted in the DB) and the
  CLI dedup path. **Remaining for DoD sign-off:** run the operator's real PDF
  through the `bge` backend end-to-end (pending the file + a real-model smoke
  check). The `bge` backend selects via `RAG_EMBEDDING_BACKEND=bge`.
- **Phase 2 — chat API: code-complete, pipeline unit-tested.** `apps/web` (Next 15,
  React 19). The full Section 3 pipeline (`lib/chat/pipeline.ts`) with dependency
  injection: classify (Haiku) → rewrite → rag `/embed` → `hybrid_search` RPC
  (RLS-scoped via the user JWT) → rag `/rerank` → fail-closed threshold → S7-prompt
  generation → citation post-check → persist + audit + analytics. `LLM_PROVIDER`
  factory (Anthropic via Vercel AI SDK, Ollama via fetch — same prompts). Real
  clients: rag (`lib/rag`), Supabase user+service (`lib/supabase`), DB layer
  (`lib/chat/db.ts`), settings loader, zod request schema, and `app/api/chat/route.ts`
  (Bearer auth for now; Phase 3 adds the cookie+AAL2 middleware). `vitest` = 15 passed
  (branch routing, fail-closed refusals, uncited-answer replacement, log
  minimisation, Hindi NOT_FOUND); `tsc --noEmit` clean.
  - **Live DoD PASSED** (`apps/web/scripts/live-smoke.mts` — drives the real
    pipeline against live Supabase + Anthropic `claude-sonnet-4-6` + the local
    rag-service with a seeded KB and a tier-3 user, then cleans up). "How do I
    apply for annual leave?" → grounded Hinglish answer, every sentence cited
    `[S1]`/`[S2]`, `refused:false`; "capital of France?" → exact `NOT_FOUND`,
    `refused:true`. Driving the pipeline from Node surfaced three integration
    fixes the unit tests can't: undici proxy egress (`EnvHttpProxyAgent`), cold
    rag-model warmup, and normalising the AI SDK's `/v1` base URL against
    `ANTHROPIC_BASE_URL`. The operator's real PDF replaces the seed KB for the
    production sign-off; the same pipeline is curl-able at `/api/chat` once
    Phase 3 auth issues sessions.
- **Phase 3 — auth: foundation code-complete.** `middleware.ts` — deny-by-default
  gate on every route: IP allowlist (fail-closed) → session → AAL2
  (`mfa.getAuthenticatorAssuranceLevel`) → active profile, funnelling to
  `/onboarding` until password-change + TOTP are done and signing out
  deactivated users. `lib/auth/ip.ts` (CIDR allowlist, 5 tests), `lib/auth/
  lockout.ts` (5 failures / 15 min via login_attempts, service-role only),
  `lib/auth/invite.ts` (super_admin invites any role/tier; admin invites jawans
  at tier 1 only; one-time temp password), `lib/supabase/server.ts` (SSR cookie
  client). **UI complete:** `/login` (two-step password → TOTP via server
  actions, lockout + IP enforced server-side) and `/onboarding` (forced
  password change → TOTP QR enrollment → AAL2 verify), plus `POST /auth/signout`.
  `tsc` clean; `vitest` 20/20. **Remaining for DoD sign-off:** the live auth run
  (/chat unreachable without AAL2; deactivated user signed out; lockout).
- **Phase 4 — chat UI: code-complete.** `/chat` — mobile-first conversation view
  (olive/navy Tailwind theme, Hindi-capable), user/assistant bubbles, a Sources
  list from `[S#]` citations, a distinct treatment for fail-closed NOT_FOUND,
  bilingual empty state with example prompts, char-capped composer, and an Admin
  link for admins. `/api/chat` now accepts the cookie session (AAL2 middleware)
  as well as Bearer, and returns `conversationId`. UI foundation: Tailwind v3 +
  CSS-variable theme, accessible primitives (`lib/ui`), browser Supabase client.
- **Phase 5 — admin console: code-complete.** `/admin` (role-gated via
  `requireAdmin`): **Users** — service-role listing incl. inactive, invite
  (one-time password), activate/deactivate with role-correct authorization;
  **Documents** — PDF upload (dedup by sha256) → private `kb` Storage bucket
  (auto-created) → rag `/ingest`, with live status, re-ingest, and
  delete (chunks cascade); **Analytics** — top answered questions and the
  unanswered-query gap from `query_analytics` (no user linkage), refusal rate.
  Polish: `not-found`, error boundary, web manifest, chat loading state.
- Phases 6–8 (hardening, eval, cloud deploy config): pending. Air-gap deploy
  kit shipped (`deploy/airgap/`).

## Document upload path

Uploads go **browser → Supabase Storage directly**, not through the app's API.
The web tier mints a signed upload URL scoped to a single object, the browser
`PUT`s the bytes to Storage, and a second small JSON call records the row and
queues ingestion.

This is forced by serverless hosting, and the constraint is worth stating plainly
because it is invisible until it bites: a Vercel function accepts at most a
**4.5 MB** request body and is killed at 60s (300s on Pro), while the document
cap is 50 MB and OCR on a large scan runs for minutes. Posting the file to an
API route would have failed on both counts. It is also simply better everywhere
else — the bytes take one hop instead of two, and a slow upload never occupies a
server process.

Two properties keep that safe:

- **Ingestion is asynchronous and owns the status.** `/ingest` with
  `background: true` validates the document exists, returns `202`, and processes
  in a worker capped by `RAG_INGEST_CONCURRENCY` (default 1 — two concurrent
  scans would exceed the RAM of a small box). Every escape path in that worker
  ends in `mark_failed`, because there is no caller left to record a failure;
  the admin table polls `documents.status`. A process restart mid-job is the one
  case that strands a row in `processing`, which is what the stale sweeper
  (`cli.py sweep-stale`) exists for.
- **The checksum is verified against the real bytes.** Since the web tier never
  sees the file, `documents.sha256` is computed in the browser. The rag-service
  re-hashes what it downloads and fails the document on a mismatch, so a file
  swapped between signing and upload can never be indexed under another's hash.
  The browser uses WebCrypto where available and a plain-JS SHA-256 otherwise —
  `crypto.subtle` is absent outside secure contexts, which is exactly the
  air-gapped LAN case (`http://10.0.0.5:3000`).

## Deployment paths

| Path | Web | DB / Storage / Auth | rag-service | LLM | Guide |
|---|---|---|---|---|---|
| Air-gapped | own box | self-hosted Supabase | own box | Ollama, local | `deploy/airgap/` |
| Single server | own box | self-hosted Supabase | own box | Ollama, local | `deploy/hosted/` |
| Vercel split | Vercel | hosted Supabase | one small box | Anthropic API | `deploy/vercel/` |

The rag-service cannot be made serverless: BGE-M3 + the reranker + Tesseract need
~5 GB resident for minutes at a time. Any hosted path therefore keeps one
always-on box, which is also what keeps embeddings and OCR local to
infrastructure the operator controls.

## Content classification handling (governance)

Deliberate policy, enforced in this session: **RESTRICTED / classified / unclear
material is never ingested through the cloud pipeline** (Anthropic API or hosted
Supabase) and is not downloaded into the cloud build sandbox. Such content is
handled only on the operator's **air-gapped** deployment (self-hosted Supabase +
`LLM_PROVIDER=ollama` + local BGE/OCR), per spec §1.4 and §15 Q1. The cloud
pilot (Phases 0–2) is for genuinely unclassified welfare/pension/leave/SOP
documents only. Classification and handling authorization are the operator's
information-security authority's responsibility, not the build's. Authorship of a
classified document does not lift its marking — declassification is a formal act
of the issuing authority, so "the author says it's fine" is not a basis to route
RESTRICTED content through the cloud path.

## Air-gapped deployment (`deploy/airgap/`)

The offline path for classified/access-controlled KBs. Everything runs on the
operator's own hardware — Postgres+pgvector, BGE-M3/reranker/OCR, and the LLM
(Ollama) — so no query text or document content ever leaves the machine. It is a
thin overlay (`docker-compose.airgap.yml` adds `ollama` + `rag` + `web`) on the
upstream Supabase self-hosting stack, plus offline helpers. The operator path is
~3 commands: `bootstrap-env.sh` (generate every secret + mint the anon/service
keys with bash+openssl, lock down auth), `stage-models.sh` (the one online step —
pull models/images into a transferable bundle), and `bringup.sh` (load →
Supabase core → `apply-migrations.sh` → `seed-admin.sh` → app; `--slim` drops the
non-essential services). `gen-jwt.mjs` is the Node equivalent of the key minting.
The chat pipeline, fail-closed gate, `[S#]` enforcement, and RLS are identical to
the cloud pilot — only the LLM transport and hosting differ. Full runbook:
`deploy/airgap/README.md`.

The `web` service ships as a standalone Next server (`output: "standalone"`,
`apps/web/Dockerfile`); the Ollama provider uses `GENERATION_MODEL` for
generation and an optional lighter `OLLAMA_CLASSIFY_MODEL` for classify/rewrite.
