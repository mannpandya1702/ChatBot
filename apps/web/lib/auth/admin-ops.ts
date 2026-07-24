/**
 * Sensitive admin operations (spec §5): authenticator reset, password reset, and
 * access (role/tier) changes. Service-role only — never import into a client
 * component.
 *
 * Why these exist: the middleware hard-requires AAL2 and enrollment is refused
 * once a verified factor exists (the MFA-bypass fix). That is correct, but it
 * means a user who loses or wipes their phone can never log in again without an
 * operator-side reset. `resetAuthenticator` is that recovery path — and it is
 * privileged, so it is authorized and audited like one.
 *
 * Authorization mirrors the invite rules: admins may act on jawans only;
 * super_admins may act on anyone. Role/tier changes are super_admin-only and
 * never on yourself, so the last super_admin cannot demote themselves out of the
 * console. The DB trigger that guards these columns exempts the service role
 * (auth.uid() is null), so THIS module is the real gate — not the database.
 */
import { randomBytes } from "node:crypto";
import { serviceClient } from "../supabase/service";
import { createServerSupabase } from "../supabase/server";
import { assertAccessInput, canActOn, type Role } from "./admin-policy";

interface Authorized {
  callerRole: Role;
  targetRole: Role;
  targetServiceNumber: string;
}

async function authorize(callerUserId: string, targetId: string): Promise<Authorized> {
  if (!targetId) throw new Error("No user selected.");
  const svc = serviceClient();

  const { data: caller } = await svc
    .from("profiles")
    .select("role, is_active")
    .eq("id", callerUserId)
    .single();
  if (!caller || (caller as { is_active: boolean }).is_active !== true) {
    throw new Error("Not authorized.");
  }
  const callerRole = (caller as { role: Role }).role;

  const { data: target } = await svc
    .from("profiles")
    .select("role, service_number")
    .eq("id", targetId)
    .single();
  if (!target) throw new Error("User not found.");
  const targetRole = (target as { role: Role }).role;

  if (!canActOn(callerRole, targetRole)) throw new Error("Not authorized.");
  return {
    callerRole,
    targetRole,
    targetServiceNumber: (target as { service_number: string }).service_number,
  };
}

/**
 * Audit trail for privileged actions. Written on the caller's session client so
 * auth.uid() attributes the row to the acting admin (log_event also stamps a
 * server-computed origin marker). Best-effort: the operation has already
 * happened, so a failed audit write is logged rather than surfaced as a failure
 * that would mislead the operator into retrying.
 */
async function audit(eventType: string, detail: Record<string, unknown>): Promise<void> {
  try {
    const supabase = await createServerSupabase();
    const { error } = await supabase.rpc("log_event", {
      p_event_type: eventType,
      p_detail: detail,
    });
    if (error) throw new Error(error.message);
  } catch (e) {
    console.error(`[admin] audit write failed (${eventType}):`, e instanceof Error ? e.message : e);
  }
}

/**
 * Remove every enrolled MFA factor so the user can enroll a new authenticator at
 * next login (the "lost/replaced phone" recovery). Deleting a verified factor
 * also invalidates the user's active sessions, so a stolen phone loses access
 * immediately rather than at token expiry.
 */
export async function resetAuthenticator(
  callerUserId: string,
  targetId: string,
): Promise<{ removed: number }> {
  const { targetServiceNumber } = await authorize(callerUserId, targetId);
  const svc = serviceClient();

  const { data, error } = await svc.auth.admin.mfa.listFactors({ userId: targetId });
  if (error) throw new Error(`Could not read authenticators: ${error.message}`);

  const factors = data?.factors ?? [];
  for (const f of factors) {
    const { error: dErr } = await svc.auth.admin.mfa.deleteFactor({ id: f.id, userId: targetId });
    if (dErr) throw new Error(`Could not remove authenticator: ${dErr.message}`);
  }

  await audit("admin_reset_authenticator", {
    target_user_id: targetId,
    target_service_number: targetServiceNumber,
    factors_removed: factors.length,
  });
  return { removed: factors.length };
}

/**
 * Issue a new one-time password and force a change at next login. Returned once
 * for the operator to hand over securely; never stored.
 *
 * must_change_password also neutralises any session the user (or an attacker
 * holding one) still has: the middleware funnels a must-change session to
 * /onboarding, so it cannot reach chat or admin with the old credential.
 */
export async function resetPassword(
  callerUserId: string,
  targetId: string,
): Promise<{ tempPassword: string; serviceNumber: string }> {
  const { targetServiceNumber } = await authorize(callerUserId, targetId);
  const svc = serviceClient();

  const tempPassword = randomBytes(15).toString("base64url");
  const { error } = await svc.auth.admin.updateUserById(targetId, { password: tempPassword });
  if (error) throw new Error(`Could not reset password: ${error.message}`);

  const { error: pErr } = await svc
    .from("profiles")
    .update({ must_change_password: true })
    .eq("id", targetId);
  if (pErr) throw new Error(`Could not flag password change: ${pErr.message}`);

  await audit("admin_reset_password", {
    target_user_id: targetId,
    target_service_number: targetServiceNumber,
  });
  return { tempPassword, serviceNumber: targetServiceNumber };
}

/** Change a user's role and/or access tier. super_admin only, never on yourself. */
export async function updateUserAccess(
  callerUserId: string,
  targetId: string,
  input: { role?: string; accessTier?: number },
): Promise<void> {
  const { callerRole, targetServiceNumber } = await authorize(callerUserId, targetId);
  if (callerRole !== "super_admin") {
    throw new Error("Only a super admin may change role or access tier.");
  }
  if (targetId === callerUserId) {
    throw new Error("You cannot change your own role or tier.");
  }
  assertAccessInput(input.role, input.accessTier);

  const patch: Record<string, unknown> = {};
  if (input.role !== undefined) patch.role = input.role;
  if (input.accessTier !== undefined) patch.access_tier = input.accessTier;

  const svc = serviceClient();
  const { error } = await svc.from("profiles").update(patch).eq("id", targetId);
  if (error) throw new Error(`Could not update access: ${error.message}`);

  await audit("admin_update_access", {
    target_user_id: targetId,
    target_service_number: targetServiceNumber,
    ...patch,
  });
}
