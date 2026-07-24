# Security policy & posture

Sainik Sahayak is an internal, air-gapped assistant that answers **only** from an
admin-curated document set, with mandatory citations or a clean refusal. This
document states what the system guarantees, how to handle restricted material,
and how to manage secrets.

---

## Handling restricted material (read first)

**A classification marking is a control, not a preference.** If a document is
marked RESTRICTED (or higher), it is handled **only** on the operator's
air-gapped deployment — self-hosted Supabase + local models (Ollama, BGE, OCR),
with the host disconnected from all networks before the document is ever placed
on it.

Restricted content must **never**:

- be sent to any hosted or cloud service (managed Supabase, any external LLM API,
  any SaaS), or
- be placed on an internet-connected build/CI/developer machine, or
- be pasted into issues, chat tools, or support channels.

Authorship or local permission does **not** downgrade a marking. The cloud/hosted
configuration of this project is for **unclassified** content only. The air-gap
kit under `deploy/airgap/` exists precisely so restricted material can be served
without egress. When in doubt, treat content as restricted and keep it on the
air-gapped host.

---

## Security posture

- **No egress.** On an air-gapped host, generation, classification, embeddings,
  reranking, and OCR are all local. No query or document text leaves the machine.
- **Deny by default.** Row-Level Security is enabled on every table; only
  explicit policies open access, and the install asserts RLS is on everywhere.
- **Fail closed.** The chat pipeline refuses (returns NOT_FOUND) whenever
  retrieval confidence is below threshold or a generated answer lacks a valid
  `[S#]` citation — an ungrounded answer is never returned.
- **Least privilege.** Jawans read only documents at or below their access tier
  and only their own conversations. Admins never read chat content.

---

## Security boundaries

**Network.** Optional `IP_ALLOWLIST` restricts the app to known networks (VPN/LAN)
and is enforced before anything else. On an air-gapped host the network itself is
the outer boundary.

**Authentication.** Password sign-in plus mandatory TOTP (AAL2) — enforced by the
middleware, not by convention. First login forces a password change and
authenticator enrollment. Repeated failures are rate-limited per service number
(server-side lockout).

Enrollment is refused once a verified factor exists, so an attacker holding only a
stolen password cannot enroll their own device to reach AAL2. The consequence is
that a user who loses their phone has **no self-service recovery**: an admin must
clear the factor (Admin → Users → Manage → Reset authenticator). That path is
privileged — admins may act on jawans only, super-admins on anyone — it signs the
user out of all sessions, and every use is written to the audit log with the
acting admin and the target.

**Authorization.** RLS is the real boundary. Tier gating is denormalised onto
chunks so a jawan can never retrieve above their tier even via search. Admin
mutations are constrained by column-level guards (e.g. only a super-admin may
change a role or tier).

**Data & logs.** The audit log is append-only at the database level (direct
INSERT/UPDATE/DELETE are revoked; writes go through a single definer function
that stamps a server-computed origin marker). Audit rows store metadata only —
never the user's query text. Analytics store the rewritten query with no user
linkage.

---

## Secret management

The deployment holds these secrets in `.env` (never committed):

| Secret | Purpose |
|--------|---------|
| `JWT_SECRET` | signs all auth tokens; `ANON_KEY` and `SERVICE_ROLE_KEY` are JWTs derived from it |
| `ANON_KEY` | public Supabase key (also baked into the web image at build time) |
| `SERVICE_ROLE_KEY` | RLS-bypassing server key — treat as root; server-side only |
| `RAG_SERVICE_SECRET` | shared secret authenticating web → rag calls |
| `POSTGRES_PASSWORD` | database superuser password |

Rules:

- Keep `.env` and any backup dumps in **separate** secure locations. A database
  dump plus these keys together is a full compromise.
- Generate strong values (`openssl rand -hex 32`). The rag-service refuses to
  start on an empty or built-in-default `RAG_SERVICE_SECRET`.
- `SERVICE_ROLE_KEY` and `POSTGRES_PASSWORD` are never exposed to the browser and
  never logged.

### Rotating secrets

Rotate on personnel changes, suspected exposure, and on a periodic schedule.
Take a backup first (`deploy/airgap/backup.sh`).

**`RAG_SERVICE_SECRET`** (lowest blast radius):

```bash
# in .env set RAG_SERVICE_SECRET to a new `openssl rand -hex 32`, then:
$COMPOSE up -d rag web        # both must share the new value
```

**`JWT_SECRET` + `ANON_KEY` + `SERVICE_ROLE_KEY`** (these move together — the keys
are signed by `JWT_SECRET`, so rotating it invalidates every session and both
keys must be re-minted):

```bash
# 1. set a new JWT_SECRET in .env
# 2. re-mint both keys offline from the new secret:
node deploy/airgap/gen-jwt.mjs            # writes new ANON_KEY / SERVICE_ROLE_KEY
# 3. update ANON_KEY and SERVICE_ROLE_KEY in .env
# 4. the anon key is inlined into the web bundle at build time — rebuild web:
$COMPOSE build web
# 5. restart the stack so GoTrue and every service pick up the new secret:
$COMPOSE up -d
```

All users must sign in again after this (expected).

**`POSTGRES_PASSWORD`:** change it on the database role and in `.env`
(`RAG_DATABASE_URL` and the Supabase services derive from it), then restart the
stack. Coordinate as a maintenance window — connections drop during the change.

(`$COMPOSE` is defined at the top of [`RUNBOOK.md`](RUNBOOK.md).)

---

## Threat model

**Defended:** network exfiltration (air-gap + no egress), unauthenticated access
(AAL2), cross-tier data access and IDOR (RLS + ownership checks), ungrounded or
fabricated answers (fail-closed citations), prompt injection via document content
(context is delimited and neutralised; the classifier screens obvious attempts),
audit tampering and forgery (append-only + origin marker), password brute force
(lockout).

**Out of scope:** a compromised host or a malicious operator with `.env`/service
credentials (they hold the keys by design); the physical security of the machine
and its backups; the correctness of the source documents themselves.

---

## Reporting a vulnerability

Report suspected vulnerabilities through your unit's internal security channel to
the system operator — **not** via public issues, and never including restricted
content. Include reproduction steps and impact; do not post exploit details where
unauthorized readers could see them.
