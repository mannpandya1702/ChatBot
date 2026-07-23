import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { env } from "../env";

/** Service-role client — bypasses RLS. Server-only; writes messages/analytics. */
export function serviceClient(): SupabaseClient {
  return createClient(env.supabaseUrl, env.supabaseServiceRoleKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/** Client bound to a user's access token — RLS applies as that user. Used for
 * hybrid_search so retrieval is tier-scoped, and for owner-scoped reads. */
export function userClient(accessToken: string): SupabaseClient {
  return createClient(env.supabaseUrl, env.supabaseAnonKey, {
    auth: { persistSession: false, autoRefreshToken: false },
    global: { headers: { Authorization: `Bearer ${accessToken}` } },
  });
}
