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

- **Phase 0 — scaffold + schema: DONE.** Migrations apply clean on a fresh database
  (`scripts/local-db/apply.sh`, stand-in for `supabase db reset` — the remote build
  environment has no Docker, so the Supabase-auth surface the migrations rely on is
  shimmed by `scripts/local-db/shim-auth.sql`; migrations themselves are untouched
  Supabase SQL). RLS matrix green via `npm run test:rls`. Seed script ready
  (`npm run seed:admin`, needs hosted credentials).
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
- Phase 2 — chat API: pending (needs LLM provider decision, spec §15 Q1 →
  answered *unclassified*, so Anthropic is viable for the pilot; needs creds).
- Phases 3–8: pending.
