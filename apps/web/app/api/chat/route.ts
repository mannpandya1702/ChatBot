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

/** Read the `aal` claim from an access-token JWT without verifying it (the
 *  token itself is validated separately by getUser). */
function tokenAal(token: string): string | null {
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const json = Buffer.from(part.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
    return (JSON.parse(json) as { aal?: string }).aal ?? null;
  } catch {
    return null;
  }
}

/**
 * Resolve the caller's access token + id from either the cookie session (the
 * browser UI, gated by the AAL2 middleware) or a Bearer token. The token drives
 * an RLS-scoped client so retrieval is tier-limited. The Bearer path enforces
 * AAL2 + an active profile itself, so the endpoint is safe independent of the
 * middleware matcher.
 */
async function resolveAuth(req: Request): Promise<{ token: string; userId: string } | null> {
  const authz = req.headers.get("authorization") ?? "";
  if (authz.startsWith("Bearer ")) {
    const token = authz.slice(7);
    const uc = userClient(token);
    const { data, error } = await uc.auth.getUser(token);
    if (error || !data?.user) return null;
    if (tokenAal(token) !== "aal2") return null;
    const { data: profile } = await uc.from("profiles").select("is_active").eq("id", data.user.id).single();
    if (!profile || (profile as { is_active: boolean }).is_active !== true) return null;
    return { token, userId: data.user.id };
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

  const { data: underLimit, error: rlErr } = await uc.rpc("check_rate_limit", {
    p_limit: 20,
    p_window: "5 minutes",
  });
  // Fail closed: only an explicit "under the limit" proceeds; an error or null
  // is treated as rate-limited rather than waved through.
  if (rlErr) console.error("[chat] rate-limit check failed, denying:", rlErr.message);
  if (underLimit !== true) {
    return NextResponse.json({ error: "rate_limited" }, { status: 429 });
  }

  let conversationId: string;
  if (convIn) {
    // Ownership gate: messages are written via the service role (RLS-bypassing),
    // so a client-supplied conversationId MUST be verified against the caller
    // here. The user client is RLS-scoped, so this returns a row only if the
    // caller owns it; 404 (not 403) avoids confirming another user's UUID.
    const { data: owned } = await uc
      .from("conversations")
      .select("id")
      .eq("id", convIn)
      .maybeSingle();
    if (!owned) {
      return NextResponse.json({ error: "conversation not found" }, { status: 404 });
    }
    conversationId = convIn;
  } else {
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
