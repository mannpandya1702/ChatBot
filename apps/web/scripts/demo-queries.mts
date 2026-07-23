/**
 * Demo: drive the real pipeline against the persistent sample KB. Shows grounded
 * citations, TIER-BASED ACCESS CONTROL (a tier-1 user cannot see tier-2 content),
 * bilingual answering, and fail-closed refusal. Creates temp users, runs queries,
 * deletes the users (keeps the KB).
 */
process.env.NO_PROXY = [process.env.NO_PROXY, "127.0.0.1", "localhost"].filter(Boolean).join(",");
import { setGlobalDispatcher, EnvHttpProxyAgent } from "undici";
setGlobalDispatcher(new EnvHttpProxyAgent({ headersTimeout: 600_000, bodyTimeout: 600_000 }));

import { randomUUID } from "node:crypto";
import { createClient } from "@supabase/supabase-js";
import { serviceClient } from "../lib/supabase/service";
import { createRagClient } from "../lib/rag/client";
import { getLlmProvider } from "../lib/llm";
import { createChatDb } from "../lib/chat/db";
import { loadSettings } from "../lib/chat/settings";
import { runChat } from "../lib/chat/pipeline";
import { env } from "../lib/env";

const svc = serviceClient();
const rag = createRagClient();

async function makeUser(tier: number): Promise<{ uid: string; token: string }> {
  const email = `demo_t${tier}_${Date.now()}@sainik.internal`;
  const password = randomUUID() + "aA1!";
  const { data: u, error } = await svc.auth.admin.createUser({ email, password, email_confirm: true });
  if (error || !u.user) throw new Error("create user: " + error?.message);
  await svc.from("profiles").insert({
    id: u.user.id, service_number: `DEMO${tier}${Date.now() % 100000}`,
    full_name: `Demo Jawan T${tier}`, role: "jawan", access_tier: tier,
    is_active: true, must_change_password: false,
  });
  const anon = createClient(env.supabaseUrl, env.supabaseAnonKey, { auth: { persistSession: false } });
  const { data: s, error: se } = await anon.auth.signInWithPassword({ email, password });
  if (se || !s.session) throw new Error("signin: " + se?.message);
  return { uid: u.user.id, token: s.session.access_token };
}

async function conv(uid: string): Promise<string> {
  const { data } = await svc.from("conversations").insert({ user_id: uid, title: "demo" }).select("id").single();
  return (data as { id: string }).id;
}

async function ask(token: string, uid: string, message: string) {
  const deps = { ...getLlmProvider(), rag, db: createChatDb(token), settings: await loadSettings() };
  const r = await runChat(deps, { conversationId: await conv(uid), message, history: [] });
  const cites = r.citations.map((c) => c.document_title).join("; ");
  console.log(`\nQ: ${message}`);
  console.log(`   refused=${r.refused} class=${r.classification}${r.citations.length ? " cites=[" + cites + "]" : ""}`);
  console.log(`   A: ${r.text.replace(/\s+/g, " ").slice(0, 240)}${r.text.length > 240 ? "…" : ""}`);
}

async function main() {
  const t1 = await makeUser(1);
  const t2 = await makeUser(2);
  try {
    console.log("========== TIER-1 JAWAN ==========");
    await ask(t1.token, t1.uid, "How do I apply for annual leave?");
    console.log("\n--- tier-1 asks about TIER-2 content (AGIF loan) — access control must block it ---");
    await ask(t1.token, t1.uid, "What are the AGIF housing loan eligibility rules?");
    console.log("\n--- Hindi query (bilingual) ---");
    await ask(t1.token, t1.uid, "पारिवारिक पेंशन का दावा कैसे करें?");
    console.log("\n--- out of scope ---");
    await ask(t1.token, t1.uid, "What is the capital of France?");

    console.log("\n\n========== TIER-2 JAWAN (same AGIF question) ==========");
    await ask(t2.token, t2.uid, "What are the AGIF housing loan eligibility rules?");
  } finally {
    await svc.auth.admin.deleteUser(t1.uid);
    await svc.auth.admin.deleteUser(t2.uid);
    await svc.from("query_analytics").delete().gte("id", 0);
    console.log("\n(demo users removed; sample KB retained)");
  }
}

main().then(() => process.exit(0)).catch((e) => { console.error("DEMO FAILED:", e); process.exit(1); });
