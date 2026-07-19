/**
 * Sainik Sahayak — RLS matrix test (Phase 0 DoD).
 *
 * Asserts every cell of the access matrix from spec §4 with five personas
 * (super_admin, admin, jawan tier-1, jawan tier-2, deactivated jawan) plus
 * anon and service_role, against a freshly-migrated database.
 *
 * Run: npm run db:local:apply && npm run test:rls
 * (On hosted Supabase the same assertions run via the CLI-created shadow db.)
 */
import { Client } from "pg";

const DB = process.env.SAINIK_TEST_DB ?? "sainik_test";

// Fixed fixture ids keep failures readable.
const SUPER = "00000000-0000-4000-8000-000000000001";
const ADMIN = "00000000-0000-4000-8000-000000000002";
const JAWAN1 = "00000000-0000-4000-8000-000000000003"; // tier 1
const JAWAN2 = "00000000-0000-4000-8000-000000000004"; // tier 2
const INACTIVE = "00000000-0000-4000-8000-000000000005";

const DOC_T1 = "00000000-0000-4000-8000-000000000011"; // ready, tier 1
const DOC_T2 = "00000000-0000-4000-8000-000000000012"; // ready, tier 2
const DOC_PROC = "00000000-0000-4000-8000-000000000013"; // processing, tier 1

const CONV_J1 = "00000000-0000-4000-8000-000000000021";
const CONV_J2 = "00000000-0000-4000-8000-000000000022";
const MSG_J1_USER = "00000000-0000-4000-8000-000000000031";
const MSG_J1_ASST = "00000000-0000-4000-8000-000000000032";
const MSG_J2_ASST = "00000000-0000-4000-8000-000000000033";

const unitVec = (): string => `[1${",0".repeat(1023)}]`;

type Persona =
  | { kind: "user"; id: string }
  | { kind: "anon" }
  | { kind: "service" };

const db = new Client({
  host: "/var/run/postgresql",
  database: DB,
  user: process.env.SAINIK_TEST_DB_USER ?? "root",
});

/** Run fn inside a rolled-back transaction impersonating a persona. */
async function as<T>(p: Persona, fn: (q: (sql: string, params?: unknown[]) => Promise<any>) => Promise<T>): Promise<T> {
  await db.query("begin");
  try {
    if (p.kind === "user") {
      await db.query(`select set_config('request.jwt.claims', $1, true)`, [
        JSON.stringify({ sub: p.id, role: "authenticated" }),
      ]);
      await db.query("set local role authenticated");
    } else if (p.kind === "anon") {
      await db.query(`select set_config('request.jwt.claims', '{}', true)`);
      await db.query("set local role anon");
    } else {
      await db.query(`select set_config('request.jwt.claims', '{}', true)`);
      await db.query("set local role service_role");
    }
    return await fn((sql, params) => db.query(sql, params));
  } finally {
    await db.query("rollback");
  }
}

let pass = 0;
let fail = 0;
const failures: string[] = [];

async function check(name: string, fn: () => Promise<void>) {
  try {
    await fn();
    pass++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    fail++;
    const msg = e instanceof Error ? e.message : String(e);
    failures.push(`${name}: ${msg}`);
    console.log(`  ✗ ${name} — ${msg}`);
  }
}

function expect(cond: boolean, detail: string) {
  if (!cond) throw new Error(detail);
}

/** Expect the query inside fn to raise, with message matching `re`. */
async function expectError(re: RegExp, fn: () => Promise<unknown>) {
  try {
    await fn();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    expect(re.test(msg), `raised, but message ${JSON.stringify(msg)} !~ ${re}`);
    return;
  }
  throw new Error(`expected error matching ${re}, but query succeeded`);
}

async function seed() {
  const users: Array<[string, string]> = [
    [SUPER, "SUP001@sainik.internal"],
    [ADMIN, "ADM001@sainik.internal"],
    [JAWAN1, "JWN001@sainik.internal"],
    [JAWAN2, "JWN002@sainik.internal"],
    [INACTIVE, "JWN003@sainik.internal"],
  ];
  for (const [id, email] of users) {
    await db.query(`insert into auth.users (id, email) values ($1, $2)`, [id, email]);
  }
  await db.query(
    `insert into public.profiles (id, service_number, full_name, rank, unit, role, access_tier, is_active) values
     ($1, 'SUP001', 'Super Admin', 'Col',  'HQ',     'super_admin', 3, true),
     ($2, 'ADM001', 'Admin One',   'Maj',  'HQ',     'admin',       1, true),
     ($3, 'JWN001', 'Jawan One',   'Sep',  'Unit A', 'jawan',       1, true),
     ($4, 'JWN002', 'Jawan Two',   'Nk',   'Unit B', 'jawan',       2, true),
     ($5, 'JWN003', 'Jawan Three', 'Sep',  'Unit A', 'jawan',       1, false)`,
    [SUPER, ADMIN, JAWAN1, JAWAN2, INACTIVE],
  );
  await db.query(
    `insert into public.documents (id, title, original_filename, storage_path, sha256, access_tier, status) values
     ($1, 'Leave Rules 2025',   'leave.pdf',   'kb/leave.pdf',   'sha-t1',   1, 'ready'),
     ($2, 'Pension Manual',     'pension.pdf', 'kb/pension.pdf', 'sha-t2',   2, 'ready'),
     ($3, 'Pending Upload',     'pending.pdf', 'kb/pending.pdf', 'sha-proc', 1, 'processing')`,
    [DOC_T1, DOC_T2, DOC_PROC],
  );
  await db.query(
    `insert into public.chunks (document_id, chunk_index, content, embedding, page_start, page_end, access_tier) values
     ($1, 0, 'Annual leave may be applied through the unit adjutant.', $4::vector, 1, 1, 1),
     ($2, 0, 'Family pension rules and claim procedure.',              $4::vector, 3, 4, 2),
     ($3, 0, 'Draft content still processing.',                        $4::vector, 1, 1, 1)`,
    [DOC_T1, DOC_T2, DOC_PROC, unitVec()],
  );
  await db.query(
    `insert into public.conversations (id, user_id, title) values
     ($1, $2, 'leave question'), ($3, $4, 'pension question')`,
    [CONV_J1, JAWAN1, CONV_J2, JAWAN2],
  );
  await db.query(
    `insert into public.messages (id, conversation_id, role, content, citations) values
     ($1, $2, 'user',      'How do I apply for leave?', null),
     ($3, $2, 'assistant', 'Apply via the adjutant [S1].', '[{"s":1}]'::jsonb),
     ($4, $5, 'assistant', 'Pension claim steps [S1].',    '[{"s":1}]'::jsonb)`,
    [MSG_J1_USER, CONV_J1, MSG_J1_ASST, MSG_J2_ASST, CONV_J2],
  );
}

async function main() {
  await db.connect();
  await seed();
  const j1 = { kind: "user", id: JAWAN1 } as const;
  const j2 = { kind: "user", id: JAWAN2 } as const;
  const adm = { kind: "user", id: ADMIN } as const;
  const sup = { kind: "user", id: SUPER } as const;
  const off = { kind: "user", id: INACTIVE } as const;
  const anon = { kind: "anon" } as const;
  const svc = { kind: "service" } as const;

  console.log("\nprofiles");
  await check("jawan sees only own profile", () =>
    as(j1, async (q) => expect((await q(`select id from profiles`)).rowCount === 1, "expected 1 row")));
  await check("admin sees all profiles", () =>
    as(adm, async (q) => expect((await q(`select id from profiles`)).rowCount === 5, "expected 5 rows")));
  await check("super_admin sees all profiles", () =>
    as(sup, async (q) => expect((await q(`select id from profiles`)).rowCount === 5, "expected 5 rows")));
  await check("anon cannot read profiles", () =>
    as(anon, (q) => expectError(/permission denied/, () => q(`select id from profiles`))));
  await check("deactivated user reads nothing, even own row", () =>
    as(off, async (q) => expect((await q(`select id from profiles`)).rowCount === 0, "expected 0 rows")));
  await check("jawan cannot update own profile", () =>
    as(j1, async (q) =>
      expect((await q(`update profiles set full_name='X' where id=$1`, [JAWAN1])).rowCount === 0, "expected 0 updated")));
  await check("admin can deactivate a jawan", () =>
    as(adm, async (q) =>
      expect((await q(`update profiles set is_active=false where id=$1`, [JAWAN1])).rowCount === 1, "expected 1 updated")));
  await check("admin cannot change a jawan's role", () =>
    as(adm, (q) => expectError(/only super_admin/, () => q(`update profiles set role='admin' where id=$1`, [JAWAN1]))));
  await check("admin cannot change other profile fields", () =>
    as(adm, (q) => expectError(/only change is_active/, () => q(`update profiles set full_name='X' where id=$1`, [JAWAN1]))));
  await check("admin cannot touch non-jawan rows", () =>
    as(adm, async (q) =>
      expect((await q(`update profiles set is_active=false where id=$1`, [SUPER])).rowCount === 0, "expected 0 updated")));
  await check("super_admin can change access_tier", () =>
    as(sup, async (q) =>
      expect((await q(`update profiles set access_tier=3 where id=$1`, [JAWAN2])).rowCount === 1, "expected 1 updated")));
  await check("service_number is immutable even for super_admin", () =>
    as(sup, (q) => expectError(/immutable/, () => q(`update profiles set service_number='ZZ' where id=$1`, [JAWAN1]))));
  await check("no client INSERT into profiles", () =>
    as(adm, (q) =>
      expectError(/row-level security/, () =>
        q(`insert into profiles (id, service_number, full_name) values ($1,'NEW','New Guy')`, [JAWAN1]))));
  await check("no client DELETE of profiles", () =>
    as(sup, async (q) =>
      expect((await q(`delete from profiles where id=$1`, [JAWAN1])).rowCount === 0, "expected 0 deleted")));

  console.log("\ndocuments");
  await check("tier-1 jawan sees only tier-1 ready docs", () =>
    as(j1, async (q) => {
      const r = await q(`select id from documents`);
      expect(r.rowCount === 1 && r.rows[0].id === DOC_T1, `got ${JSON.stringify(r.rows)}`);
    }));
  await check("tier-2 jawan sees tier-1 and tier-2 ready docs", () =>
    as(j2, async (q) => expect((await q(`select id from documents`)).rowCount === 2, "expected 2 rows")));
  await check("processing docs are invisible to all client JWTs", () =>
    as(j2, async (q) =>
      expect((await q(`select id from documents where id=$1`, [DOC_PROC])).rowCount === 0, "expected 0 rows")));
  await check("anon cannot read documents", () =>
    as(anon, (q) => expectError(/permission denied/, () => q(`select id from documents`))));
  await check("no client INSERT into documents (admin included)", () =>
    as(adm, (q) =>
      expectError(/row-level security/, () =>
        q(`insert into documents (title, original_filename, storage_path, sha256) values ('x','x','x','sha-x')`))));
  await check("no client UPDATE of documents", () =>
    as(j2, async (q) =>
      expect((await q(`update documents set title='hax' where id=$1`, [DOC_T1])).rowCount === 0, "expected 0 updated")));
  await check("service role sees every document (console listings)", () =>
    as(svc, async (q) => expect((await q(`select id from documents`)).rowCount === 3, "expected 3 rows")));

  console.log("\nchunks & hybrid_search");
  await check("tier-1 jawan sees only tier-1 chunks of ready docs", () =>
    as(j1, async (q) => {
      const r = await q(`select document_id from chunks`);
      expect(r.rowCount === 1 && r.rows[0].document_id === DOC_T1, `got ${JSON.stringify(r.rows)}`);
    }));
  await check("tier-2 jawan sees tier-1 + tier-2 chunks", () =>
    as(j2, async (q) => expect((await q(`select id from chunks`)).rowCount === 2, "expected 2 rows")));
  await check("deactivated user retrieves no chunks", () =>
    as(off, async (q) => expect((await q(`select id from chunks`)).rowCount === 0, "expected 0 rows")));
  await check("hybrid_search is tier-scoped for tier-1 jawan", () =>
    as(j1, async (q) => {
      const r = await q(`select doc_title from hybrid_search($1::vector, 'leave', 20)`, [unitVec()]);
      expect(r.rowCount === 1 && r.rows[0].doc_title === "Leave Rules 2025", `got ${JSON.stringify(r.rows)}`);
    }));
  await check("hybrid_search returns both tiers for tier-2 jawan", () =>
    as(j2, async (q) =>
      expect((await q(`select chunk_id from hybrid_search($1::vector, 'leave', 20)`, [unitVec()])).rowCount === 2, "expected 2 rows")));
  await check("anon cannot execute hybrid_search", () =>
    as(anon, (q) => expectError(/permission denied/, () => q(`select * from hybrid_search($1::vector, 'x', 5)`, [unitVec()]))));

  console.log("\nconversations");
  await check("owner sees only own conversations", () =>
    as(j1, async (q) => {
      const r = await q(`select id from conversations`);
      expect(r.rowCount === 1 && r.rows[0].id === CONV_J1, `got ${JSON.stringify(r.rows)}`);
    }));
  await check("admin sees zero conversations (no chat snooping)", () =>
    as(adm, async (q) => expect((await q(`select id from conversations`)).rowCount === 0, "expected 0 rows")));
  await check("owner can create a conversation for self", () =>
    as(j1, async (q) =>
      expect((await q(`insert into conversations (user_id, title) values ($1,'t') returning id`, [JAWAN1])).rowCount === 1, "insert failed")));
  await check("cannot create a conversation for someone else", () =>
    as(j1, (q) =>
      expectError(/row-level security/, () => q(`insert into conversations (user_id, title) values ($1,'t')`, [JAWAN2]))));
  await check("owner can rename own conversation", () =>
    as(j1, async (q) =>
      expect((await q(`update conversations set title='renamed' where id=$1`, [CONV_J1])).rowCount === 1, "expected 1 updated")));
  await check("owner can delete own conversation", () =>
    as(j1, async (q) =>
      expect((await q(`delete from conversations where id=$1`, [CONV_J1])).rowCount === 1, "expected 1 deleted")));
  await check("deactivated user cannot create conversations", () =>
    as(off, (q) =>
      expectError(/row-level security/, () => q(`insert into conversations (user_id, title) values ($1,'t')`, [INACTIVE]))));

  console.log("\nmessages");
  await check("owner reads own messages", () =>
    as(j1, async (q) => expect((await q(`select id from messages`)).rowCount === 2, "expected 2 rows")));
  await check("other users' messages are invisible", () =>
    as(j1, async (q) =>
      expect((await q(`select id from messages where id=$1`, [MSG_J2_ASST])).rowCount === 0, "expected 0 rows")));
  await check("admin sees zero messages", () =>
    as(adm, async (q) => expect((await q(`select id from messages`)).rowCount === 0, "expected 0 rows")));
  await check("owner can set thumbs feedback", () =>
    as(j1, async (q) =>
      expect((await q(`update messages set feedback=1 where id=$1`, [MSG_J1_ASST])).rowCount === 1, "expected 1 updated")));
  await check("owner cannot rewrite message content", () =>
    as(j1, (q) => expectError(/only change message feedback/, () => q(`update messages set content='forged' where id=$1`, [MSG_J1_ASST]))));
  await check("cannot set feedback on someone else's message", () =>
    as(j1, async (q) =>
      expect((await q(`update messages set feedback=-1 where id=$1`, [MSG_J2_ASST])).rowCount === 0, "expected 0 updated")));
  await check("clients cannot INSERT messages (server pipeline only)", () =>
    as(j1, (q) =>
      expectError(/row-level security/, () =>
        q(`insert into messages (conversation_id, role, content) values ($1,'user','direct write')`, [CONV_J1]))));
  await check("clients cannot DELETE messages", () =>
    as(j1, async (q) =>
      expect((await q(`delete from messages where id=$1`, [MSG_J1_ASST])).rowCount === 0, "expected 0 deleted")));

  console.log("\naudit_logs (append-only)");
  await check("log_event works for authenticated users", () =>
    as(j1, async (q) => {
      const r = await q(`select log_event('login', '{"ok":true}'::jsonb) as id`);
      expect(typeof r.rows[0].id === "string" || typeof r.rows[0].id === "number", "no id returned");
    }));
  // seed one persistent event for the read tests below (as() rolls back)
  await db.query(`insert into public.audit_logs (user_id, event_type) values ($1, 'login')`, [JAWAN1]);
  await check("jawan cannot read audit logs", () =>
    as(j1, async (q) => expect((await q(`select id from audit_logs`)).rowCount === 0, "expected 0 rows")));
  await check("admin can read audit logs", () =>
    as(adm, async (q) => expect(((await q(`select id from audit_logs`)).rowCount ?? 0) >= 1, "expected ≥1 row")));
  await check("UPDATE is rejected at the DB level (even admin)", () =>
    as(adm, (q) => expectError(/permission denied/, () => q(`update audit_logs set event_type='tampered'`))));
  await check("DELETE is rejected at the DB level (even admin)", () =>
    as(adm, (q) => expectError(/permission denied/, () => q(`delete from audit_logs`))));
  await check("direct INSERT is rejected (log_event is the only door)", () =>
    as(adm, (q) => expectError(/permission denied/, () => q(`insert into audit_logs (event_type) values ('forged')`))));
  await check("anon cannot call log_event", () =>
    as(anon, (q) => expectError(/permission denied/, () => q(`select log_event('x')`))));

  console.log("\napp_settings");
  await check("authenticated users read settings", () =>
    as(j1, async (q) => expect(((await q(`select key from app_settings`)).rowCount ?? 0) >= 6, "expected ≥6 keys")));
  await check("admin cannot write settings", () =>
    as(adm, async (q) =>
      expect((await q(`update app_settings set value='9'::jsonb where key='message_retention_days'`)).rowCount === 0, "expected 0 updated")));
  await check("super_admin can write settings", () =>
    as(sup, async (q) =>
      expect((await q(`update app_settings set value='0.5'::jsonb where key='rerank_refusal_threshold'`)).rowCount === 1, "expected 1 updated")));
  await check("anon cannot read settings", () =>
    as(anon, (q) => expectError(/permission denied/, () => q(`select key from app_settings`))));

  console.log("\nquery_analytics & login_attempts");
  await check("service role records query events", () =>
    as(svc, async (q) => {
      const r = await q(`select record_query_event('pension kaise milega', 'hinglish', false, 0.71) as id`);
      expect(r.rows[0].id != null, "no id returned");
    }));
  await check("authenticated users cannot call record_query_event", () =>
    as(j1, (q) => expectError(/permission denied/, () => q(`select record_query_event('x','en',false,null)`))));
  await db.query(`select public.record_query_event('seeded query', 'en', true, 0.12)`);
  await check("admin reads the unanswered-queries backlog", () =>
    as(adm, async (q) => expect(((await q(`select id from query_analytics`)).rowCount ?? 0) >= 1, "expected ≥1 row")));
  await check("jawans cannot read query analytics", () =>
    as(j1, async (q) => expect((await q(`select id from query_analytics`)).rowCount === 0, "expected 0 rows")));
  await check("login_attempts is service-role-only", () =>
    as(j1, (q) => expectError(/permission denied/, () => q(`select id from login_attempts`))));
  await check("service role writes login_attempts", () =>
    as(svc, async (q) =>
      expect((await q(`insert into login_attempts (service_number, success) values ('JWN001', false) returning id`)).rowCount === 1, "insert failed")));

  console.log("\nrate limiting");
  await check("check_rate_limit true while under the limit", () =>
    as(j1, async (q) => expect((await q(`select check_rate_limit(20, '5 minutes') as ok`)).rows[0].ok === true, "expected true")));
  await check("check_rate_limit false once limit is hit", () =>
    as(j1, async (q) => expect((await q(`select check_rate_limit(1, '1 hour') as ok`)).rows[0].ok === false, "expected false")));

  console.log(`\n${pass} passed, ${fail} failed`);
  if (fail > 0) {
    console.log("\nFailures:");
    for (const f of failures) console.log(`  - ${f}`);
  }
  await db.end();
  process.exit(fail > 0 ? 1 : 0);
}

main().catch(async (e) => {
  console.error("fatal:", e);
  try { await db.end(); } catch {}
  process.exit(1);
});
