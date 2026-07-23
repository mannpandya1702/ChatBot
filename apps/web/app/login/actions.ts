"use server";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";
import { isLockedOut, recordAttempt } from "@/lib/auth/lockout";

export interface LoginState {
  error?: string;
  step?: "password" | "mfa";
}

function emailFor(serviceNumber: string): string {
  return `${serviceNumber.trim().toLowerCase()}@sainik.internal`;
}

async function clientIp(): Promise<string | null> {
  const h = await headers();
  const xff = h.get("x-forwarded-for");
  return xff ? xff.split(",")[0]!.trim() : h.get("x-real-ip");
}

/** Step 1: password. Enforces the 5/15min lockout, then decides MFA vs done. */
export async function signInAction(_prev: LoginState, form: FormData): Promise<LoginState> {
  const serviceNumber = String(form.get("serviceNumber") ?? "").trim();
  const password = String(form.get("password") ?? "");
  if (!serviceNumber || !password) return { error: "Enter your service number and password." };

  if (await isLockedOut(serviceNumber)) {
    return { error: "Too many attempts. Try again in 15 minutes. / बहुत अधिक प्रयास। 15 मिनट बाद पुनः प्रयास करें।" };
  }

  const ip = await clientIp();
  const supabase = await createServerSupabase();
  const { error } = await supabase.auth.signInWithPassword({ email: emailFor(serviceNumber), password });
  if (error) {
    await recordAttempt(serviceNumber, ip, false);
    return { error: "Invalid service number or password. / सेवा संख्या या पासवर्ड गलत है।" };
  }
  await recordAttempt(serviceNumber, ip, true);

  // Enrolled TOTP factor ⇒ collect the 6-digit code before granting AAL2.
  const { data: aal } = await supabase.auth.mfa.getAuthenticatorAssuranceLevel();
  if (aal?.currentLevel === "aal1" && aal?.nextLevel === "aal2") {
    return { step: "mfa" };
  }
  // No factor yet (first login) or already AAL2 — let the middleware route.
  redirect("/");
}

/** Step 2: verify the TOTP code to reach AAL2. */
export async function verifyMfaAction(_prev: LoginState, form: FormData): Promise<LoginState> {
  const code = String(form.get("code") ?? "").trim();
  if (!/^\d{6}$/.test(code)) return { step: "mfa", error: "Enter the 6-digit code. / 6 अंकों का कोड दर्ज करें।" };

  const supabase = await createServerSupabase();
  const { data: factors } = await supabase.auth.mfa.listFactors();
  const factor = factors?.totp?.[0];
  if (!factor) return { step: "mfa", error: "No authenticator enrolled." };

  const { data: challenge, error: cErr } = await supabase.auth.mfa.challenge({ factorId: factor.id });
  if (cErr || !challenge) return { step: "mfa", error: "Could not start verification. Try again." };

  const { error: vErr } = await supabase.auth.mfa.verify({
    factorId: factor.id,
    challengeId: challenge.id,
    code,
  });
  if (vErr) return { step: "mfa", error: "Invalid code. / कोड गलत है।" };

  redirect("/");
}
