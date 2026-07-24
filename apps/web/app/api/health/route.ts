import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Liveness probe for load balancers / the container healthcheck. Public (the
 * middleware lets it through before any auth/IP check) and intentionally
 * minimal — it proves the web tier is up without leaking any internal state.
 */
export function GET(): Response {
  return NextResponse.json({ status: "ok", service: "web" });
}
