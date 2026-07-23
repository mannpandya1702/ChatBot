/**
 * Admin invite (spec §5). Creates the auth user + profile and returns a
 * one-time temp password shown once. First login forces a password change then
 * TOTP enrollment (middleware funnels them through /onboarding until AAL2).
 *
 * Authorization: super_admin may invite any role/tier; admin may invite jawans
 * only, at tier 1 (role/tier changes are super_admin-only per §4). Service-role
 * only — never import into a client component.
 */
import { randomBytes } from "node:crypto";
import { serviceClient } from "../supabase/service";

export interface InviteInput {
  serviceNumber: string;
  fullName: string;
  rank?: string;
  unit?: string;
  accessTier: number;
  role: "jawan" | "admin" | "super_admin";
}

export interface InviteResult {
  userId: string;
  email: string;
  tempPassword: string; // display once; never stored
}

export async function createInvite(
  callerUserId: string,
  input: InviteInput,
): Promise<InviteResult> {
  const svc = serviceClient();

  const { data: caller } = await svc
    .from("profiles")
    .select("role, is_active")
    .eq("id", callerUserId)
    .single();
  if (!caller || caller.is_active !== true) throw new Error("not authorized");
  const callerRole = (caller as { role: string }).role;
  if (callerRole !== "super_admin" && callerRole !== "admin") {
    throw new Error("not authorized");
  }

  let role = input.role;
  let tier = input.accessTier;
  if (callerRole === "admin") {
    if (input.role !== "jawan") throw new Error("admins may only invite jawans");
    role = "jawan";
    tier = 1; // only super_admin sets role/tier
  }
  if (!Number.isInteger(tier) || tier < 1 || tier > 3) {
    throw new Error("invalid access tier");
  }

  const email = `${input.serviceNumber.toLowerCase()}@sainik.internal`;
  const tempPassword = randomBytes(15).toString("base64url");

  const { data: created, error } = await svc.auth.admin.createUser({
    email,
    password: tempPassword,
    email_confirm: true,
  });
  if (error || !created.user) throw new Error(`create user: ${error?.message}`);

  const { error: pErr } = await svc.from("profiles").insert({
    id: created.user.id,
    service_number: input.serviceNumber,
    full_name: input.fullName,
    rank: input.rank ?? null,
    unit: input.unit ?? null,
    role,
    access_tier: tier,
    is_active: true,
    must_change_password: true,
    created_by: callerUserId,
  });
  if (pErr) {
    await svc.auth.admin.deleteUser(created.user.id); // roll back the orphan
    throw new Error(`create profile: ${pErr.message}`);
  }

  return { userId: created.user.id, email, tempPassword };
}
