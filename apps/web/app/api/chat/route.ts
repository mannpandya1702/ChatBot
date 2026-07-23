import { NextResponse } from "next/server";
import { chatRequestSchema } from "@/lib/chat/schema";
import { getLlmProvider } from "@/lib/llm";
import { createRagClient } from "@/lib/rag/client";
import { createChatDb } from "@/lib/chat/db";
import { loadSettings } from "@/lib/chat/settings";
import { userClient } from "@/lib/supabase/service";
import { createServerSupabase } from "@/lib/supabase/server";
import { runChat } from "@/lib/chat/pipeline";
import type { ConversationMessage } from "@/lib/chat/types";

export const runtime = "nodejs";

/**
 * Resolve the caller's access token + id from either the cookie session (the
 * browser UI, gated by the AAL2 middleware) or a Bearer token (smoke scripts).
 * The token drives an RLS-scoped client so retrieval is tier-limited.
 */
async function resolveAuth(req: Request): Promise<{ token: string; userId: string } | null> {
  const authz = req.headers.get("authorization") ?? "";
  if (authz.startsWith("Bearer ")) {
    const token = authz.slice(7);
    const { data, error } = await userClient(token).auth.getUser(token);
    return error || !data?.user ? null : { token, userId: data.user.id };
  }
  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  const { data: { session } } = await supabase.auth.getSession();
  if (!user || !session?.access_token) return null;
  return { token: session.access_token, userId: user.id };
}

/**
 * Chat endpoint (spec §3). Auth via cookie session (AAL2 middleware) or Bearer.
 * Retrieval runs under the user's JWT so RLS tier-scopes the knowledge base.
 */
export async function POST(req: Request): Promise<Response> {
  const auth = await resolveAuth(req);
  if (!auth) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { token, userId } = auth;

  const uc = userClient(token);

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
    return NextResponse.json({ ...result, conversationId });
  } catch (e) {
    // Never leak internals (spec §11); the failure is logged server-side.
    console.error("chat pipeline error", e);
    return NextResponse.json({ error: "internal error" }, { status: 500 });
  }
}
