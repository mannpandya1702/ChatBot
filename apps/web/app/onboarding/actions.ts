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

/** Step 2: (re)issue a fresh TOTP secret + QR for the authenticator app. */
export async function enrollAction(): Promise<EnrollState> {
  const supabase = await createServerSupabase();

  // Clear any half-finished (unverified) factors so enroll never name-clashes.
  const { data: list } = await supabase.auth.mfa.listFactors();
  for (const f of list?.all ?? []) {
    if (f.status !== "verified") await supabase.auth.mfa.unenroll({ factorId: f.id });
  }

  const { data, error } = await supabase.auth.mfa.enroll({ factorType: "totp" });
  if (error || !data) return { error: "Could not start authenticator setup. Try again." };
  return { factorId: data.id, qr: data.totp.qr_code, secret: data.totp.secret };
}

/** Step 3: verify the 6-digit code → session reaches AAL2 → done. */
export async function verifyEnrollAction(_prev: VerifyState, form: FormData): Promise<VerifyState> {
  const factorId = String(form.get("factorId") ?? "");
  const code = String(form.get("code") ?? "").trim();
  if (!factorId) return { error: "Setup expired. Reload and try again." };
  if (!/^\d{6}$/.test(code)) return { error: "Enter the 6-digit code. / 6 अंकों का कोड दर्ज करें।" };

  const supabase = await createServerSupabase();
  const { data: challenge, error: cErr } = await supabase.auth.mfa.challenge({ factorId });
  if (cErr || !challenge) return { error: "Could not verify. Try again." };

  const { error: vErr } = await supabase.auth.mfa.verify({ factorId, challengeId: challenge.id, code });
  if (vErr) return { error: "Invalid code. / कोड गलत है।" };

  redirect("/");
}
