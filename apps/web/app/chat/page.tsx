import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";
import { ChatClient, type Msg } from "./chat-client";
import type { ConversationSummary } from "./conversation-sidebar";

export const metadata = { title: "Chat — Sainik Sahayak" };

// Enough history to scroll through without turning the sidebar into a data dump.
const HISTORY_LIMIT = 50;

export default async function ChatPage({
  searchParams,
}: {
  searchParams: Promise<{ c?: string }>;
}) {
  const { c: requested } = await searchParams;
  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  // Both reads run on the session client, so RLS scopes them to this user.
  const [{ data: profile }, { data: convRows }] = await Promise.all([
    supabase
      .from("profiles")
      .select("full_name, service_number, role, access_tier")
      .eq("id", user.id)
      .single(),
    supabase
      .from("conversations")
      .select("id, title, updated_at")
      .order("updated_at", { ascending: false })
      .limit(HISTORY_LIMIT),
  ]);

  const conversations = (convRows ?? []) as ConversationSummary[];

  let activeId: string | null = null;
  let initialMessages: Msg[] = [];

  if (requested) {
    // Ownership is enforced by RLS, not by scanning the list above (which is
    // capped): someone else's id — or a malformed one — simply returns no row,
    // and we fall back to a fresh chat rather than erroring.
    const { data: owned } = await supabase
      .from("conversations")
      .select("id")
      .eq("id", requested)
      .maybeSingle();

    if (owned) {
      activeId = requested;
      const { data: rows } = await supabase
        .from("messages")
        .select("role, content, citations, refused")
        .eq("conversation_id", requested)
        .order("created_at", { ascending: true });

      initialMessages = (rows ?? []).map((r) => {
        const m = r as {
          role: string;
          content: string;
          citations: Msg["citations"] | null;
          refused: boolean | null;
        };
        return {
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content,
          citations: m.citations ?? undefined,
          refused: m.refused ?? false,
        } satisfies Msg;
      });
    }
  }

  return (
    <ChatClient
      // Remount on conversation switch so the transcript state starts clean
      // from the server-loaded history rather than merging two conversations.
      key={activeId ?? "new"}
      user={{
        fullName: profile?.full_name ?? "Jawan",
        serviceNumber: profile?.service_number ?? "",
        role: (profile?.role as string) ?? "jawan",
      }}
      conversations={conversations}
      activeId={activeId}
      initialMessages={initialMessages}
    />
  );
}
