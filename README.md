# Sainik Sahayak (सैनिक सहायक)

Invite-only bilingual chatbot for Indian Army jawans. Every answer is cited from an
admin-managed PDF knowledge base — or the bot refuses cleanly. No guesswork.

**For authorized personnel only / केवल अधिकृत कर्मियों हेतु**

## Layout

```
apps/web/              Next.js 15 app (chat UI + admin console)     [Phase 2/4]
services/rag/          FastAPI ingestion/embedding/rerank service   [Phase 1]
supabase/migrations/   schema, RLS, hybrid_search, log_event        [Phase 0 ✓]
scripts/               seed-admin, test-rls, local db harness       [Phase 0 ✓]
golden/                eval set (golden.jsonl)                      [Phase 7]
docs/                  ARCHITECTURE.md · SECURITY.md · RUNBOOK.md
```

## Phase 0 verification

```bash
npm install
npm run db:local:apply   # fresh DB + all migrations (local stand-in for `supabase db reset`)
npm run test:rls         # full RLS matrix: 5 personas + anon + service_role
```

Against a hosted Supabase project: `supabase db reset`, then seed the first admin:

```bash
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
  npm run seed:admin -- --service-number SUP001 --name "Your Name"
```

See `docs/ARCHITECTURE.md` for the full design and the phase log.
