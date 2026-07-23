import { NextResponse } from "next/server";
import { createServerSupabase } from "@/lib/supabase/server";

/** Sign out and return to /login. POST-only (state-changing). */
export async function POST(req: Request): Promise<Response> {
  const supabase = await createServerSupabase();
  await supabase.auth.signOut();
  return NextResponse.redirect(new URL("/login", req.url), { status: 303 });
}
