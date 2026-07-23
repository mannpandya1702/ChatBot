/**
 * Deny-by-default gate on every app route (spec §5). A request reaches a
 * protected page only with: a valid session, AAL2 (TOTP verified), an active
 * profile, and (if configured) a whitelisted source IP. Any failure redirects
 * to /login or /onboarding, or signs the user out. Fail closed throughout.
 */
import { createServerClient, type CookieOptions } from "@supabase/ssr";
import { NextResponse, type NextRequest } from "next/server";
import { env } from "@/lib/env";
import { ipAllowed } from "@/lib/auth/ip";

// Reachable without a completed (AAL2 + active) session.
const PUBLIC_PREFIXES = ["/login", "/auth"];
const ONBOARDING = "/onboarding";

function clientIp(req: NextRequest): string | null {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0]!.trim();
  return req.headers.get("x-real-ip");
}

export async function middleware(req: NextRequest) {
  // 1. IP allowlist (VPN-only deployments) — enforced before anything else.
  if (!ipAllowed(clientIp(req), env.ipAllowlist)) {
    return new NextResponse("Forbidden", { status: 403 });
  }

  const res = NextResponse.next({ request: req });
  const supabase = createServerClient(env.supabaseUrl, env.supabaseAnonKey, {
    cookies: {
      getAll() {
        return req.cookies.getAll();
      },
      setAll(list: { name: string; value: string; options: CookieOptions }[]) {
        for (const { name, value, options } of list) res.cookies.set(name, value, options);
      },
    },
  });

  const path = req.nextUrl.pathname;
  const isPublic = PUBLIC_PREFIXES.some((p) => path === p || path.startsWith(p + "/"));
  const isOnboarding = path === ONBOARDING || path.startsWith(ONBOARDING + "/");
  // API routes get a 401 JSON on denial instead of an HTML redirect.
  const isApi = path.startsWith("/api/");
  const deny = (to: string) => (isApi ? unauthorized() : redirect(req, to));

  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return isPublic ? res : deny("/login");
  }

  // 2. AAL2 — must have completed TOTP. Below AAL2 → funnel to onboarding.
  const { data: aal } = await supabase.auth.mfa.getAuthenticatorAssuranceLevel();
  const atAal2 = aal?.currentLevel === "aal2";

  // 3. Active profile; forced password change funnels to onboarding too.
  const { data: profile } = await supabase
    .from("profiles")
    .select("is_active, must_change_password")
    .eq("id", user.id)
    .single();

  if (!profile || profile.is_active !== true) {
    await supabase.auth.signOut();
    return deny("/login");
  }

  const mustOnboard = !atAal2 || profile.must_change_password === true;
  if (mustOnboard) {
    return isOnboarding || isPublic ? res : deny("/onboarding");
  }

  // Fully authenticated — keep them out of login/onboarding.
  if (isPublic || isOnboarding) return redirect(req, "/chat");
  return res;
}

function redirect(req: NextRequest, to: string): NextResponse {
  const url = req.nextUrl.clone();
  url.pathname = to;
  url.search = "";
  return NextResponse.redirect(url);
}

function unauthorized(): NextResponse {
  return new NextResponse(JSON.stringify({ error: "unauthorized" }), {
    status: 401,
    headers: { "content-type": "application/json" },
  });
}

export const config = {
  // Everything except Next internals and static assets.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|manifest.webmanifest|.*\\.[\\w]+$).*)"],
};
