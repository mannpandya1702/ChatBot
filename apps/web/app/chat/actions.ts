"use server";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";

/**
 * Delete one of the caller's own conversations; its messages cascade.
 *
 * The delete runs on the session-scoped client, so RLS — not application logic —
 * confines it to the owner: another user's conversation id simply matches no row
 * (admins cannot read or delete chat content either, per the access matrix).
 */
export async function deleteConversationAction(form: FormData): Promise<void> {
  const id = String(form.get("id") ?? "");
  const activeId = String(form.get("activeId") ?? "");
  if (!id) return;

  const supabase = await createServerSupabase();
  await supabase.from("conversations").delete().eq("id", id);

  revalidatePath("/chat");
  // Only leave the current view if the chat being deleted is the one on screen.
  if (id === activeId) redirect("/chat");
}
