/**
 * Phase 2 live DoD: drive the REAL pipeline against live Supabase + Anthropic +
 * the local rag-service. Seeds a tiny KB, creates a tier-3 user, runs a grounded
 * question and an out-of-KB question through runChat(), then cleans up.
 * Route fetches through the sandbox proxy (localhost bypassed via NO_PROXY).
 */
process.env.NO_PROXY = [process.env.NO_PROXY, "127.0.0.1", "localhost"].filter(Boolean).join(",");
import { setGlobalDispatcher, EnvHttpProxyAgent } from "undici";
// Generous timeouts: cold BGE model loads in the rag-service can exceed
// undici's default headers timeout on the first request.
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

const KB = [
  { content: "Annual leave may be applied through the unit adjutant. Applications should be submitted at least seven days in advance. The sanctioning authority is the officer commanding.", page: 3 },
  { content: "Casual leave is limited to thirty days in a calendar year and cannot be combined with annual leave. Requests are recorded in the unit leave register.", page: 4 },
  { content: "Family pension is payable to the next of kin. The claim is submitted with the death certificate and service documents to the pension section.", page: 12 },
];

async function seedKb(): Promise<string> {
  const docId = randomUUID();
  await svc.from("documents").insert({
    id: docId, title: "Leave & Pension Rules (smoke)", original_filename: "smoke.pdf",
    storage_path: "kb/smoke.pdf", sha256: "smoke-" + docId, access_tier: 1,
    status: "ready", page_count: 12, chunk_count: KB.length,
  });
  const embeddings = await rag.embed(KB.map((k) => k.content));
  for (let i = 0; i < KB.length; i++) {
    const { error } = await svc.from("chunks").insert({
      document_id: docId, chunk_index: i, content: KB[i].content,
      embedding: "[" + embeddings[i].join(",") + "]",
      page_start: KB[i].page, page_end: KB[i].page, access_tier: 1,
    });
    if (error) throw new Error("chunk insert: " + error.message);
  }
  return docId;
}

async function makeUser(): Promise<{ uid: string; token: string }> {
  const email = `smoke_${Date.now()}@sainik.internal`;
  const password = randomUUID() + "aA1!";
  const { data: u, error } = await svc.auth.admin.createUser({ email, password, email_confirm: true });
  if (error || !u.user) throw new Error("create user: " + error?.message);
  await svc.from("profiles").insert({
    id: u.user.id, service_number: "SMOKE" + Date.now(), full_name: "Smoke Test",
    role: "jawan", access_tier: 3, is_active: true, must_change_password: false,
  });
  const anon = createClient(env.supabaseUrl, env.supabaseAnonKey, { auth: { persistSession: false } });
  const { data: s, error: se } = await anon.auth.signInWithPassword({ email, password });
  if (se || !s.session) throw new Error("signin: " + se?.message);
  return { uid: u.user.id, token: s.session.access_token };
}

async function conv(uid: string): Promise<string> {
  const { data, error } = await svc.from("conversations").insert({ user_id: uid, title: "smoke" }).select("id").single();
  if (error) throw new Error("conv: " + error.message);
  return (data as { id: string }).id;
}

async function main() {
  await rag.embed(["warmup"]);
  await rag.rerank("warmup", [{ id: "x", text: "warmup" }], 1);

  const docId = await seedKb();
  const { uid, token } = await makeUser();
  const deps = { ...getLlmProvider(), rag, db: createChatDb(token), settings: await loadSettings() };

  try {
    const r1 = await runChat(deps, { conversationId: await conv(uid), message: "How do I apply for annual leave?", history: [] });
    console.log("\n=== Q1 (in-KB): How do I apply for annual leave? ===");
    console.log("refused:", r1.refused, "| classification:", r1.classification, "| model:", r1.model);
    console.log("answer:", r1.text);
    console.log("citations:", JSON.stringify(r1.citations.map((c) => ({ s: c.s, doc: c.document_title, pp: [c.page_start, c.page_end] }))));

    const r2 = await runChat(deps, { conversationId: await conv(uid), message: "What is the capital of France?", history: [] });
    console.log("\n=== Q2 (out-of-KB): What is the capital of France? ===");
    console.log("refused:", r2.refused, "| classification:", r2.classification);
    console.log("answer:", r2.text);

    console.log("\n=== DoD checks ===");
    console.log("Q1 grounded + cited:", r1.refused === false && r1.citations.length > 0 && /\[S\d+\]/.test(r1.text));
    console.log("Q2 refused (NOT_FOUND):", r2.refused === true);
  } finally {
    await svc.from("documents").delete().eq("id", docId); // cascades chunks
    await svc.auth.admin.deleteUser(uid); // cascades profile, conversations, messages
    await svc.from("query_analytics").delete().gte("id", 0);
    console.log("\ncleanup done (doc+chunks+user+conversations+analytics removed)");
  }
}

main().then(() => process.exit(0)).catch((e) => { console.error("SMOKE FAILED:", e); process.exit(1); });
