"use client";
import { useActionState, useState } from "react";
import { Button } from "@/lib/ui/button";
import { Input } from "@/lib/ui/input";
import { Label, Card, Alert, Spinner } from "@/lib/ui/misc";
import { inviteAction, setActiveAction, type InviteState } from "./actions";

export interface UserRow {
  id: string;
  service_number: string;
  full_name: string;
  rank: string | null;
  unit: string | null;
  role: string;
  access_tier: number;
  is_active: boolean;
  must_change_password: boolean;
  last_login_at: string | null;
}

export function UsersManager({
  users,
  callerRole,
  callerId,
}: {
  users: UserRow[];
  callerRole: "admin" | "super_admin";
  callerId: string;
}) {
  const [invite, doInvite, inviting] = useActionState(inviteAction, {} as InviteState);
  const [showForm, setShowForm] = useState(false);
  const [copied, setCopied] = useState(false);
  const isSuper = callerRole === "super_admin";

  // Works on a secure origin AND on a plain-HTTP LAN (where navigator.clipboard
  // is undefined) via an execCommand fallback, with visible confirmation.
  async function copyPassword(text: string) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // leave the password visible on screen for manual copy
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Button onClick={() => setShowForm((v) => !v)} variant={showForm ? "outline" : "default"}>
          {showForm ? "Close" : "+ Invite user"}
        </Button>
      </div>

      {showForm && (
        <Card className="p-5">
          {invite.ok ? (
            <div className="space-y-3">
              <Alert variant="success">
                Account created for <strong>{invite.email}</strong>.
              </Alert>
              <div>
                <Label>One-time temporary password (shown once)</Label>
                <div className="mt-1 flex items-center gap-2">
                  <code className="flex-1 break-all rounded bg-muted px-3 py-2 font-mono text-sm">{invite.tempPassword}</code>
                  <Button type="button" variant="outline" size="sm" onClick={() => copyPassword(invite.tempPassword ?? "")}>
                    {copied ? "Copied!" : "Copy"}
                  </Button>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Share it securely. The user changes it and sets up TOTP at first login.
                </p>
              </div>
              <Button variant="outline" onClick={() => window.location.reload()}>Done</Button>
            </div>
          ) : (
            <form action={doInvite} className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="serviceNumber">Service number *</Label>
                <Input id="serviceNumber" name="serviceNumber" required autoCapitalize="characters" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="fullName">Full name *</Label>
                <Input id="fullName" name="fullName" required />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="rank">Rank</Label>
                <Input id="rank" name="rank" />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="unit">Unit</Label>
                <Input id="unit" name="unit" />
              </div>
              {isSuper ? (
                <>
                  <div className="space-y-1.5">
                    <Label htmlFor="role">Role</Label>
                    <select id="role" name="role" defaultValue="jawan" className="flex h-10 w-full rounded-md border border-input bg-card px-3 text-sm">
                      <option value="jawan">Jawan</option>
                      <option value="admin">Admin</option>
                      <option value="super_admin">Super admin</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="accessTier">Access tier</Label>
                    <select id="accessTier" name="accessTier" defaultValue="1" className="flex h-10 w-full rounded-md border border-input bg-card px-3 text-sm">
                      <option value="1">Tier 1</option>
                      <option value="2">Tier 2</option>
                      <option value="3">Tier 3</option>
                    </select>
                  </div>
                </>
              ) : (
                <p className="sm:col-span-2 text-xs text-muted-foreground">
                  As an admin you can invite jawans at tier 1. Role and tier changes are super-admin only.
                </p>
              )}
              {invite.error && <div className="sm:col-span-2"><Alert variant="error">{invite.error}</Alert></div>}
              <div className="sm:col-span-2">
                <Button type="submit" disabled={inviting}>{inviting && <Spinner />} Create invite</Button>
              </div>
            </form>
          )}
        </Card>
      )}

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Service no.</th>
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">Role</th>
                <th className="px-4 py-2 font-medium">Tier</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {users.map((u) => {
                const isSelf = u.id === callerId;
                const canToggle = !isSelf && (isSuper || u.role === "jawan");
                return (
                  <tr key={u.id} className={u.is_active ? "" : "opacity-60"}>
                    <td className="px-4 py-2 font-mono text-xs">{u.service_number}</td>
                    <td className="px-4 py-2">
                      {u.full_name}
                      {u.rank ? <span className="text-muted-foreground"> · {u.rank}</span> : ""}
                    </td>
                    <td className="px-4 py-2 capitalize">{u.role.replace("_", " ")}</td>
                    <td className="px-4 py-2">{u.access_tier}</td>
                    <td className="px-4 py-2">
                      <span className={`inline-flex rounded-full px-2 py-0.5 text-xs ${u.is_active ? "bg-primary/15 text-primary" : "bg-muted text-muted-foreground"}`}>
                        {u.is_active ? "Active" : "Inactive"}
                      </span>
                      {u.must_change_password && u.is_active && (
                        <span className="ml-1 text-[10px] text-muted-foreground">· pending setup</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right">
                      {canToggle ? (
                        <form action={setActiveAction} className="inline">
                          <input type="hidden" name="targetId" value={u.id} />
                          <input type="hidden" name="active" value={(!u.is_active).toString()} />
                          <Button type="submit" size="sm" variant={u.is_active ? "outline" : "default"}>
                            {u.is_active ? "Deactivate" : "Activate"}
                          </Button>
                        </form>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
              {users.length === 0 && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">No users yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
