"use client";
import { createBrowserClient } from "@supabase/ssr";

/**
 * Browser Supabase client (cookie-backed session, shared with the middleware).
 * Used by the login and onboarding forms for password auth and TOTP MFA.
 *
 * Reads only the two NEXT_PUBLIC_ vars (inlined at build) — never lib/env,
 * which holds server secrets and must not enter the client bundle.
 */
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  );
}
