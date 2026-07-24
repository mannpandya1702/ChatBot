"use server";
import { revalidatePath } from "next/cache";
import { requireAdmin } from "@/lib/auth/admin-guard";
import { createInvite } from "@/lib/auth/invite";
import {
  resetAuthenticator,
  resetPassword,
  updateUserAccess,
} from "@/lib/auth/admin-ops";
import { serviceClient } from "@/lib/supabase/service";

export interface InviteState {
  error?: string;
  ok?: boolean;
  email?: string;
  tempPassword?: string;
}

export async function inviteAction(_prev: InviteState, form: FormData): Promise<InviteState> {
  const admin = await requireAdmin();
  const serviceNumber = String(form.get("serviceNumber") ?? "").trim();
  const fullName = String(form.get("fullName") ?? "").trim();
  if (!serviceNumber || !fullName) return { error: "Service number and name are required." };

  const role = (String(form.get("role") ?? "jawan") as "jawan" | "admin" | "super_admin");
  const accessTier = Number(form.get("accessTier") ?? "1");

  try {
    const res = await createInvite(admin.userId, {
      serviceNumber,
      fullName,
      rank: String(form.get("rank") ?? "").trim() || undefined,
      unit: String(form.get("unit") ?? "").trim() || undefined,
      role,
      accessTier,
    });
    revalidatePath("/admin/users");
    return { ok: true, email: res.email, tempPassword: res.tempPassword };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "invite failed";
    // Surface the common, actionable case cleanly.
    return { error: /duplicate|already/i.test(msg) ? "That service number already has an account." : msg };
  }
}

export async function setActiveAction(form: FormData): Promise<void> {
  const admin = await requireAdmin();
  const targetId = String(form.get("targetId") ?? "");
  const active = String(form.get("active") ?? "") === "true";
  if (!targetId) return;
  if (targetId === admin.userId) return; // never lock yourself out

  const svc = serviceClient();
  const { data: target } = await svc.from("profiles").select("role").eq("id", targetId).single();
  if (!target) return;
  const targetRole = (target as { role: string }).role;

  // admins may only toggle jawans; super_admins may toggle anyone (but not self, above).
  if (admin.role === "admin" && targetRole !== "jawan") return;

  await svc.from("profiles").update({ is_active: active }).eq("id", targetId);
  revalidatePath("/admin/users");
}

/**
 * Row-level admin operations. These return a result (rather than redirecting) so
 * the console can show the one-time password or a precise error inline; the
 * library layer owns authorization and auditing.
 */
export interface OpResult {
  ok?: boolean;
  error?: string;
  tempPassword?: string;
  serviceNumber?: string;
  removed?: number;
}

function failure(e: unknown): OpResult {
  return { error: e instanceof Error ? e.message : "Operation failed." };
}

/** Clear enrolled authenticators so the user can set up a new phone at next login. */
export async function resetAuthenticatorAction(targetId: string): Promise<OpResult> {
  const admin = await requireAdmin();
  try {
    const { removed } = await resetAuthenticator(admin.userId, targetId);
    revalidatePath("/admin/users");
    return { ok: true, removed };
  } catch (e) {
    return failure(e);
  }
}

/** Issue a new one-time password (shown once) and force a change at next login. */
export async function resetPasswordAction(targetId: string): Promise<OpResult> {
  const admin = await requireAdmin();
  try {
    const { tempPassword, serviceNumber } = await resetPassword(admin.userId, targetId);
    revalidatePath("/admin/users");
    return { ok: true, tempPassword, serviceNumber };
  } catch (e) {
    return failure(e);
  }
}

/** Change a user's role and/or access tier (super_admin only). */
export async function updateAccessAction(
  targetId: string,
  role: string,
  accessTier: number,
): Promise<OpResult> {
  const admin = await requireAdmin();
  try {
    await updateUserAccess(admin.userId, targetId, { role, accessTier });
    revalidatePath("/admin/users");
    return { ok: true };
  } catch (e) {
    return failure(e);
  }
}
