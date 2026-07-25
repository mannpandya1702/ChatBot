# Deploy on Vercel (frontend) + Supabase (backend)

The app split across managed services: **Vercel** serves the web app and admin
console, **Supabase** holds the database, the PDFs and sign-in, and **one small
server** does the document processing.

That third piece is not optional. Extraction, Hindi/English OCR, embeddings and
reranking need about 5 GB of models resident in memory for minutes at a time —
which is precisely what a serverless function cannot do. Supabase stores the
PDFs; it does not read them.

```
   browser ──HTTPS──► Vercel (Next.js: chat UI, admin console, API routes)
      │                  │
      │                  ├──► Supabase   Postgres + pgvector, Auth, private Storage
      │                  └──► rag-service   extract · OCR · embed · rerank
      └──── PDF upload ─────► Supabase Storage (direct, see "Why uploads bypass Vercel")
```

**Time:** about an hour, most of it waiting for the model image to build.

Prefer everything on one box, no Vercel? Use [`../hosted/HOSTED.md`](../hosted/HOSTED.md)
(or [`../hosted/ORACLE.md`](../hosted/ORACLE.md) for the free tier). Offline
install? [`../airgap/README.md`](../airgap/README.md).

> **Classified material never goes this way.** This path uses hosted Supabase
> and, with `LLM_PROVIDER=anthropic`, an external model API. RESTRICTED or
> unclear content belongs on the air-gapped deployment. See "Content
> classification handling" in [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

---

## What you need

- A **Supabase** project (region `ap-south-1` / Mumbai for residency).
- A **server for the rag-service**: 4 vCPU, **8 GB RAM**, ~40 GB disk, public IP.
  Oracle's Always Free ARM shape (4 cores / 24 GB, ₹0/month) is more than
  enough — [`../hosted/ORACLE.md`](../hosted/ORACLE.md) Part 1 covers creating
  one, including the two traps that catch most people.
- A **domain** with two records you can point: one for the rag-service, one
  optional vanity name for the app.
- A **Vercel** account.

---

## 1. Supabase

Skip whatever you have already done.

1. Create the project, then apply the schema from a machine with the Supabase
   CLI:

   ```bash
   supabase link --project-ref <your-project-ref>
   supabase db push
   ```

2. Seed the first super-admin (one time):

   ```bash
   SUPABASE_URL=https://PROJECT.supabase.co \
   SUPABASE_SERVICE_ROLE_KEY=<service-role-key> \
     npm run seed:admin -- --service-number SUP001 --name "Full Name" --rank Col --unit HQ
   ```

3. **Enable MFA.** The middleware requires AAL2 on every route, so if TOTP is
   off nobody can ever finish signing in. Supabase dashboard →
   **Authentication → Providers → Multi-Factor** → enable **TOTP (App
   Authenticator)**.

The private `kb` Storage bucket is created automatically on the first upload —
nothing to do.

---

## 2. The rag-service box

DNS first: point an **A record** at the server's IP, e.g.
`rag.sahayak.example.in`. Certificates are issued by checking DNS, so this has
to resolve before you bring anything up.

```bash
dig +short rag.sahayak.example.in    # should print your server IP
```

Then, on the server:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && exec su -l $USER

git clone https://github.com/mannpandya1702/chatbot.git
cd chatbot/deploy/vercel
cp .env.rag.example .env
$EDITOR .env          # fill in every value; generate the secret with: openssl rand -hex 32

docker compose -f docker-compose.rag.yml up -d --build
```

The build downloads BGE-M3 and the reranker (~5 GB) and bakes them into the
image, so the first build is slow and every start after it is fast. Watch it
come up:

```bash
docker compose -f docker-compose.rag.yml logs -f rag
curl -s https://rag.sahayak.example.in/health     # {"status":"ok",...}
```

`/health` is the only endpoint without the shared secret, and it returns no
secrets. Everything else answers `401` without the `X-Service-Secret` header.

> **Firewall.** Ports 80 and 443 must be open. On Oracle Cloud that means
> **both** the console Security List and the instance's own iptables rules —
> see [`../hosted/ORACLE.md`](../hosted/ORACLE.md) Part 4, which is where most
> first attempts stall.

---

## 3. Vercel

Import the GitHub repository, then set:

| Setting | Value |
|---|---|
| **Root Directory** | `apps/web` ← this one matters; the repo is a monorepo |
| Framework Preset | Next.js (detected) |
| Build / Install command | leave as detected |

Region and function limits come from [`apps/web/vercel.json`](../../apps/web/vercel.json)
(Mumbai `bom1`, to sit next to Supabase) — nothing to configure by hand.

### Environment variables

Set these for **Production** (and Preview, if you use it):

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://PROJECT.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase → Settings → API → anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Settings → API → **service_role** key |
| `RAG_SERVICE_URL` | `https://rag.sahayak.example.in` (no trailing slash) |
| `RAG_SERVICE_SECRET` | the same value as on the rag box |
| `LLM_PROVIDER` | `anthropic` |
| `ANTHROPIC_API_KEY` | your key |
| `GENERATION_MODEL` | `claude-sonnet-4-6` |
| `CLASSIFIER_MODEL` | `claude-haiku-4-5` |
| `RERANK_REFUSAL_THRESHOLD` | `0.35` |
| `TRUSTED_PROXY_COUNT` | `1` |

`TRUSTED_PROXY_COUNT=1` is not optional if you use `IP_ALLOWLIST`. The client IP
is taken that many hops from the right of `X-Forwarded-For`; at `0` the app
would trust the leftmost entry, which any caller can set to whatever they like.

> `SUPABASE_SERVICE_ROLE_KEY` bypasses RLS entirely. Keep it out of anything
> named `NEXT_PUBLIC_*` — CI fails the build if a client component so much as
> imports the module that reads it.

Deploy. Then add your app domain in **Settings → Domains** if you want one.

---

## 4. Check it end to end

1. **Health** — `https://<your-app>/api/health` → `{"status":"ok"}`.
2. **Sign in** with the seeded super-admin. You will be forced through a
   password change and authenticator enrollment. If the 6-digit step never
   appears, MFA is off in Supabase (step 1.3).
3. **Upload a PDF** — Admin → Documents → choose a file → **Upload & ingest**.
   The row appears as *Processing* and flips to *Ready* with a page and chunk
   count once the rag-service finishes. It refreshes on its own.
4. **Ask something the document answers.** You should get an answer with `[S1]`
   citations and a Sources list.
5. **Ask something it doesn't** — "what is the capital of France?" — and you
   should get the bilingual refusal. That is the fail-closed behaviour working,
   not a bug.

---

## Why uploads bypass Vercel

Worth knowing before you change anything in the admin console.

A Vercel serverless function can only receive a **4.5 MB** request body, and is
killed at 60 seconds (300 on Pro). A scanned 40 MB manual that takes four
minutes to OCR fails both limits comfortably. So the upload does not go through
the API at all:

1. the browser hashes the file and asks `/api/admin/documents/upload-url` for a
   signed URL, scoped to one object in the private bucket;
2. it `PUT`s the bytes **straight to Supabase Storage** — no size ceiling, and
   none of it on Vercel's clock;
3. it calls `/api/admin/documents`, which records the row and *queues*
   ingestion — a fast call that returns while the work runs on the rag box.

The rag-service owns `documents.status` from that point, and the admin table
polls it. The checksum the browser computed is re-verified against the real
bytes during ingestion, so a file that changed in between fails instead of being
indexed under someone else's hash.

Two consequences to remember:

- **The status column is the source of truth**, not the upload response. A
  document that says *Processing* forever means the rag-service died mid-job;
  `docker compose logs rag` will say why, and **Re-ingest** retries it.
- **Ingests are serialised** (`RAG_INGEST_CONCURRENCY=1`). Uploading ten PDFs
  at once is fine — they queue. Raising it on an 8 GB box will cause swapping.

---

## Costs

| Piece | Typical |
|---|---|
| Vercel | Free (Hobby) for this traffic shape |
| Supabase | Free tier fits a pilot; Pro (~$25/mo) for backups + more storage |
| rag-service box | ₹0 on Oracle Always Free, else ~$25–40/mo for 8 GB |
| Anthropic API | per question — the only cost that scales with usage |

Wanting no per-question cost at all means running the model yourself, which
means one box running everything: [`../hosted/ORACLE.md`](../hosted/ORACLE.md).

---

## Day-2 operations

[`../../RUNBOOK.md`](../../RUNBOOK.md) — users, documents, backups, tuning the
refusal threshold, and what to do when something breaks. It applies to this
install too, with two differences:

- **Logs** for the web app are in the Vercel dashboard, not `docker logs`.
- **Backups** are Supabase's job here (Pro plan has PITR); the backup scripts in
  `deploy/airgap/` are for self-hosted Postgres and do not apply.
