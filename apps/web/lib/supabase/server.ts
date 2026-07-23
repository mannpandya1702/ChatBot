import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { cookies } from "next/headers";
import { env } from "../env";

type CookieToSet = { name: string; value: string; options: CookieOptions };

/** Cookie-bound Supabase client for Server Components / Route Handlers. */
export async function createServerSupabase() {
  const store = await cookies();
  return createServerClient(env.supabaseUrl, env.supabaseAnonKey, {
    cookies: {
      getAll() {
        return store.getAll();
      },
      setAll(list: CookieToSet[]) {
        try {
          for (const { name, value, options } of list) store.set(name, value, options);
        } catch {
          // called from a Server Component render — safe to ignore; the
          // middleware refreshes the session cookie.
        }
      },
    },
  });
}
