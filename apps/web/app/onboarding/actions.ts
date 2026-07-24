"use server";
import { redirect } from "next/navigation";
import { createServerSupabase } from "@/lib/supabase/server";
import { serviceClient } from "@/lib/supabase/service";

export interface PwState { error?: string; ok?: boolean }
export interface EnrollState { factorId?: string; qr?: string; secret?: string; error?: string }
export interface VerifyState { error?: string }

const MIN_PASSWORD = 10;

/** Step 1: set a new password and clear the forced-change flag. */
export async function changePasswordAction(_prev: PwState, form: FormData): Promise<PwState> {
  const pw = String(form.get("password") ?? "");
  const confirm = String(form.get("confirm") ?? "");
  if (pw.length < MIN_PASSWORD) return { error: `Password must be at least ${MIN_PASSWORD} characters. / पासवर्ड कम से कम ${MIN_PASSWORD} अक्षरों का हो।` };
  if (pw !== confirm) return { error: "Passwords do not match. / पासवर्ड मेल नहीं खाते।" };

  const supabase = await createServerSupabase();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { error } = await supabase.auth.updateUser({ password: pw });
  if (error) return { error: "Could not update password. Choose a stronger one and retry." };

  // must_change_password is not client-updatable (RLS); flip it service-side.
  await serviceClient().from("profiles").update({ must_change_password: false }).eq("id", user.id);
  return { ok: true };
}

/**
 * Step 2 (first-time only): issue a fresh TOTP secret + QR.
 *
 * SECURITY: refuse to enroll a NEW factor when the user already has a verified
 * one. Otherwise a password-only (AAL1) attacker could open /onboarding, enroll
 * their own authenticator, verify it, and reach AAL2 — bypassing MFA entirely.
 * Users who already have a factor must instead CHALLENGE it (see below).
 */
export async function enrollAction(): Promise<EnrollState> {
  const supabase = await createServerSupabase();

  const { data: list } = await supabase.auth.mfa.listFactors();
  if ((list?.all ?? []).some((f) => f.status === "verified")) {
    return { error: "An authenticator is already set up for this account." };
  }

  // Clear any half-finished (unverified) factors so enroll never name-clashes.
  for (const f of list?.all ?? []) {
    if (f.status !== "verified") await supabase.auth.mfa.unenroll({ factorId: f.id });
  }

  const { data, error } = await supabase.auth.mfa.enroll({ factorType: "totp" });
  if (error || !data) return { error: "Could not start authenticator setup. Try again." };
  return { factorId: data.id, qr: data.totp.qr_code, secret: data.totp.secret };
}

/** Step 3 (first-time): verify the freshly-enrolled code → AAL2 → done. */
export async function verifyEnrollAction(_prev: VerifyState, form: FormData): Promise<VerifyState> {
  const factorId = String(form.get("factorId") ?? "");
  const code = String(form.get("code") ?? "").trim();
  if (!factorId) return { error: "Setup expired. Reload and try again." };
  if (!/^\d{6}$/.test(code)) return { error: "Enter the 6-digit code. / 6 अंकों का कोड दर्ज करें।" };

  const supabase = await createServerSupabase();

  // Defense in depth: only the factor we just enrolled (still unverified) may be
  // verified here — never elevate against a pre-existing verified factor.
  const { data: list } = await supabase.auth.mfa.listFactors();
  const target = (list?.all ?? []).find((f) => f.id === factorId);
  if (!target || target.status === "verified") return { error: "Setup expired. Reload and try again." };

  const { data: challenge, error: cErr } = await supabase.auth.mfa.challenge({ factorId });
  if (cErr || !challenge) return { error: "Could not verify. Try again." };

  const { error: vErr } = await supabase.auth.mfa.verify({ factorId, challengeId: challenge.id, code });
  if (vErr) return { error: "Invalid code. / कोड गलत है।" };

  redirect("/");
}

/**
 * Challenge an EXISTING verified factor to reach AAL2 (for a returning user the
 * middleware funnelled to /onboarding at AAL1). This is the only elevation path
 * once a factor exists — it proves possession of the real authenticator.
 */
export async function challengeExistingAction(_prev: VerifyState, form: FormData): Promise<VerifyState> {
  const code = String(form.get("code") ?? "").trim();
  if (!/^\d{6}$/.test(code)) return { error: "Enter the 6-digit code. / 6 अंकों का कोड दर्ज करें।" };

  const supabase = await createServerSupabase();
  const { data: list } = await supabase.auth.mfa.listFactors();
  const factor = (list?.all ?? []).find((f) => f.status === "verified");
  if (!factor) return { error: "No authenticator found. Contact your unit admin." };

  const { data: challenge, error: cErr } = await supabase.auth.mfa.challenge({ factorId: factor.id });
  if (cErr || !challenge) return { error: "Could not verify. Try again." };

  const { error: vErr } = await supabase.auth.mfa.verify({ factorId: factor.id, challengeId: challenge.id, code });
  if (vErr) return { error: "Invalid code. / कोड गलत है।" };

  redirect("/");
}
