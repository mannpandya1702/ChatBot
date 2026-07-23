import { redirect } from "next/navigation";
import { createServerSupabase } from "../supabase/server";

export interface AdminProfile {
  userId: string;
  role: "admin" | "super_admin";
  fullName: string;
  serviceNumber: string;
}

/**
 * Server-side gate for /admin routes. The middleware guarantees an active AAL2
 * session; this additionally requires an admin/super_admin role and hands back
 * the caller's profile. Non-admins are sent to /chat.
 */
export async function requireAdmin(): Promise<AdminProfile> {
  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: p } = await supabase
    .from("profiles")
    .select("role, full_name, service_number")
    .eq("id", user.id)
    .single();

  const role = p?.role;
  if (role !== "admin" && role !== "super_admin") redirect("/chat");
  return {
    userId: user.id,
    role,
    fullName: (p as { full_name: string }).full_name,
    serviceNumber: (p as { service_number: string }).service_number,
  };
}
