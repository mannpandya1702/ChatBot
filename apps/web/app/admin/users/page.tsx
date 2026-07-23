import { requireAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { UsersManager, type UserRow } from "./users-manager";

export const metadata = { title: "Users — Admin" };

export default async function UsersPage() {
  const admin = await requireAdmin();
  const { data } = await serviceClient()
    .from("profiles")
    .select("id, service_number, full_name, rank, unit, role, access_tier, is_active, must_change_password, last_login_at")
    .order("created_at", { ascending: false });

  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Users</h1>
      <p className="mb-5 text-sm text-muted-foreground">
        Invite personnel and manage access. Accounts are invite-only; new users must set a password and enroll TOTP before any access.
      </p>
      <UsersManager
        users={(data ?? []) as UserRow[]}
        callerRole={admin.role}
        callerId={admin.userId}
      />
    </div>
  );
}
