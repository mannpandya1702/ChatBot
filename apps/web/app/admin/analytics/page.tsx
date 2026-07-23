import { requireAdmin } from "@/lib/auth/admin-guard";

export const metadata = { title: "Analytics — Admin" };

export default async function AnalyticsPage() {
  await requireAdmin();
  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Analytics</h1>
      <p className="text-sm text-muted-foreground">Top questions and the unanswered-query gap — building next.</p>
    </div>
  );
}
