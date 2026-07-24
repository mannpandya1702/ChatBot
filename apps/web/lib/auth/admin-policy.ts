/**
 * Admin authorization policy — pure decisions, no I/O.
 *
 * Kept separate from admin-ops.ts (which talks to Supabase) so the rules that
 * decide who may act on whom are unit-tested directly, not inferred from
 * integration behaviour.
 */
export type Role = "jawan" | "admin" | "super_admin";

export const ROLES: readonly Role[] = ["jawan", "admin", "super_admin"];

/**
 * Who may perform a privileged operation on whom.
 * super_admin: anyone. admin: jawans only. Everyone else: nobody.
 */
export function canActOn(callerRole: string, targetRole: string): boolean {
  if (callerRole === "super_admin") return true;
  if (callerRole === "admin") return targetRole === "jawan";
  return false;
}

/** Validate an access (role/tier) change; throws with a user-facing message. */
export function assertAccessInput(role: string | undefined, tier: number | undefined): void {
  if (role === undefined && tier === undefined) {
    throw new Error("Nothing to change.");
  }
  if (role !== undefined && !ROLES.includes(role as Role)) {
    throw new Error("Invalid role.");
  }
  if (tier !== undefined && (!Number.isInteger(tier) || tier < 1 || tier > 3)) {
    throw new Error("Access tier must be 1, 2 or 3.");
  }
}
