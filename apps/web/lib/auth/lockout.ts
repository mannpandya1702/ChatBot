/**
 * Server-side login lockout (spec §5): 5 failed attempts lock an account for 15
 * minutes. Tracked in login_attempts (service-role only), never client-side.
 */
import { serviceClient } from "../supabase/service";

const MAX_FAILURES = 5;
const WINDOW_MINUTES = 15;

export async function recordAttempt(
  serviceNumber: string,
  ip: string | null,
  success: boolean,
): Promise<void> {
  const svc = serviceClient();
  await svc.from("login_attempts").insert({
    service_number: serviceNumber,
    ip,
    success,
  });
}

/** True if the account has ≥5 failures in the last 15 minutes since its last
 * success. A successful login clears the lock (failures before it don't count). */
export async function isLockedOut(serviceNumber: string): Promise<boolean> {
  const svc = serviceClient();
  const since = new Date(Date.now() - WINDOW_MINUTES * 60_000).toISOString();
  const { data } = await svc
    .from("login_attempts")
    .select("success, created_at")
    .eq("service_number", serviceNumber)
    .gte("created_at", since)
    .order("created_at", { ascending: false });

  let failures = 0;
  for (const row of (data ?? []) as { success: boolean }[]) {
    if (row.success) break; // most recent success resets the counter
    failures++;
  }
  return failures >= MAX_FAILURES;
}
