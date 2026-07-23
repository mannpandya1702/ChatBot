import type { ReactNode } from "react";
import Link from "next/link";
import { requireAdmin } from "@/lib/auth/admin-guard";
import { Button } from "@/lib/ui/button";

export default async function AdminLayout({ children }: { children: ReactNode }) {
  const admin = await requireAdmin();
  const nav = [
    { href: "/admin/users", label: "Users" },
    { href: "/admin/documents", label: "Documents" },
    { href: "/admin/analytics", label: "Analytics" },
  ];
  return (
    <div className="min-h-dvh bg-background">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Link href="/chat" className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-sm font-bold text-primary-foreground" aria-hidden="true">सै</span>
              <span className="text-sm font-semibold">Admin Console</span>
            </Link>
            <nav className="hidden gap-1 sm:flex">
              {nav.map((n) => (
                <Link key={n.href} href={n.href} className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-accent hover:text-foreground">
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden text-xs text-muted-foreground sm:inline">{admin.fullName} · {admin.role}</span>
            <Link href="/chat" className="text-sm text-muted-foreground hover:text-foreground">← Chat</Link>
            <form action="/auth/signout" method="post">
              <Button type="submit" variant="ghost" size="sm">Sign out</Button>
            </form>
          </div>
        </div>
        <nav className="flex gap-1 border-t border-border px-4 py-1 sm:hidden">
          {nav.map((n) => (
            <Link key={n.href} href={n.href} className="rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-accent">
              {n.label}
            </Link>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>
    </div>
  );
}
