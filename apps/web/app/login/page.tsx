import { Card } from "@/lib/ui/misc";
import { LoginForm } from "./login-form";

export const metadata = { title: "Sign in — Sainik Sahayak" };

export default function LoginPage() {
  return (
    <main className="flex min-h-dvh items-center justify-center bg-background px-4 py-10">
      <div className="w-full max-w-sm animate-fade-in">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            {/* Placeholder mark — swap for the unit/formation emblem. */}
            <span className="text-lg font-bold" aria-hidden="true">सै</span>
          </div>
          <h1 className="text-xl font-semibold text-foreground">Sainik Sahayak</h1>
          <p className="mt-1 text-sm text-muted-foreground">सैनिक सहायक · Authorized personnel only</p>
        </div>
        <Card className="p-6">
          <LoginForm />
        </Card>
        <p className="mt-4 text-center text-xs text-muted-foreground">
          Access is invite-only and logged. Contact your unit admin for an account.
        </p>
      </div>
    </main>
  );
}
