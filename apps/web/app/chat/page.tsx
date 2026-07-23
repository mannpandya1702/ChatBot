import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";
import { ChatClient } from "./chat-client";

export const metadata = { title: "Chat — Sainik Sahayak" };

export default async function ChatPage() {
  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: profile } = await supabase
    .from("profiles")
    .select("full_name, service_number, role, access_tier")
    .eq("id", user.id)
    .single();

  return (
    <ChatClient
      user={{
        fullName: profile?.full_name ?? "Jawan",
        serviceNumber: profile?.service_number ?? "",
        role: (profile?.role as string) ?? "jawan",
      }}
    />
  );
}
