import { NextResponse } from "next/server";
import { chatRequestSchema } from "@/lib/chat/schema";
import { getLlmProvider } from "@/lib/llm";
import { createRagClient } from "@/lib/rag/client";
import { createChatDb } from "@/lib/chat/db";
import { loadSettings } from "@/lib/chat/settings";
import { userClient } from "@/lib/supabase/service";
import { runChat } from "@/lib/chat/pipeline";
import type { ConversationMessage } from "@/lib/chat/types";

export const runtime = "nodejs";

/**
 * Chat endpoint (spec §3). Phase 2 authenticates via a Bearer access token;
 * Phase 3 adds the cookie session + AAL2 + is_active middleware in front of it.
 * Retrieval runs under the user's JWT so RLS tier-scopes the knowledge base.
 */
export async function POST(req: Request): Promise<Response> {
  const authz = req.headers.get("authorization") ?? "";
  const token = authz.startsWith("Bearer ") ? authz.slice(7) : "";
  if (!token) return NextResponse.json({ error: "unauthorized" }, { status: 401 });

  const uc = userClient(token);
  const { data: auth, error: authErr } = await uc.auth.getUser(token);
  if (authErr || !auth?.user) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const userId = auth.user.id;

  const body = await req.json().catch(() => null);
  const parsed = chatRequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "invalid request" }, { status: 400 });
  }
  const { message, conversationId: convIn } = parsed.data;

  const { data: underLimit } = await uc.rpc("check_rate_limit", {
    p_limit: 20,
    p_window: "5 minutes",
  });
  if (underLimit === false) {
    return NextResponse.json({ error: "rate_limited" }, { status: 429 });
  }

  let conversationId = convIn ?? null;
  if (!conversationId) {
    const { data: conv, error } = await uc
      .from("conversations")
      .insert({ user_id: userId, title: message.slice(0, 60) })
      .select("id")
      .single();
    if (error || !conv) {
      return NextResponse.json({ error: "could not start conversation" }, { status: 500 });
    }
    conversationId = (conv as { id: string }).id;
  }

  const { data: hist } = await uc
    .from("messages")
    .select("role,content")
    .eq("conversation_id", conversationId)
    .order("created_at", { ascending: false })
    .limit(6);
  const history: ConversationMessage[] = ((hist ?? []) as ConversationMessage[])
    .slice()
    .reverse();

  const provider = getLlmProvider();
  const deps = {
    ...provider,
    rag: createRagClient(),
    db: createChatDb(token),
    settings: await loadSettings(),
  };

  try {
    const result = await runChat(deps, { conversationId, message, history });
    return NextResponse.json(result);
  } catch (e) {
    // Never leak internals (spec §11); the failure is logged server-side.
    console.error("chat pipeline error", e);
    return NextResponse.json({ error: "internal error" }, { status: 500 });
  }
}
