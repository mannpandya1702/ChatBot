import { requireAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { Card } from "@/lib/ui/misc";

export const metadata = { title: "Analytics — Admin" };

interface Row { query_text: string; language: string | null; refused: boolean; created_at: string }

function tally(rows: Row[]): { text: string; count: number }[] {
  const m = new Map<string, { text: string; count: number }>();
  for (const r of rows) {
    const key = r.query_text.trim().toLowerCase();
    const e = m.get(key);
    if (e) e.count++;
    else m.set(key, { text: r.query_text.trim(), count: 1 });
  }
  return [...m.values()].sort((a, b) => b.count - a.count);
}

export default async function AnalyticsPage() {
  await requireAdmin();
  // Most recent window (no user linkage stored — see query_analytics design note).
  const { data } = await serviceClient()
    .from("query_analytics")
    .select("query_text, language, refused, created_at")
    .order("created_at", { ascending: false })
    .limit(2000);

  const rows = (data ?? []) as Row[];
  const total = rows.length;
  const refusedRows = rows.filter((r) => r.refused);
  const refusalRate = total ? Math.round((refusedRows.length / total) * 100) : 0;
  const topAnswered = tally(rows.filter((r) => !r.refused)).slice(0, 12);
  const topGaps = tally(refusedRows).slice(0, 12);
  const maxAnswered = topAnswered[0]?.count ?? 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="mb-1 text-lg font-semibold">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          Based on the {total.toLocaleString()} most recent questions. Query text is stored without any user linkage.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Questions (recent)" value={total.toLocaleString()} />
        <Stat label="Answered from KB" value={(total - refusedRows.length).toLocaleString()} />
        <Stat label="Refusal rate" value={`${refusalRate}%`} hint="Questions the KB couldn’t answer" />
      </div>

      <Card className="p-5">
        <h2 className="mb-3 text-sm font-semibold">Top answered questions</h2>
        {topAnswered.length === 0 ? (
          <Empty />
        ) : (
          <ul className="space-y-2">
            {topAnswered.map((q) => (
              <li key={q.text} className="text-sm">
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate">{q.text}</span>
                  <span className="shrink-0 text-xs text-muted-foreground">{q.count}</span>
                </div>
                <div className="mt-1 h-1.5 rounded-full bg-muted">
                  <div className="h-1.5 rounded-full bg-primary" style={{ width: `${(q.count / maxAnswered) * 100}%` }} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card className="p-5">
        <h2 className="mb-1 text-sm font-semibold">Unanswered — knowledge gaps</h2>
        <p className="mb-3 text-xs text-muted-foreground">
          Questions that returned NOT_FOUND. These are candidates for new documents.
        </p>
        {topGaps.length === 0 ? (
          <Empty />
        ) : (
          <ul className="divide-y divide-border">
            {topGaps.map((q) => (
              <li key={q.text} className="flex items-center justify-between gap-3 py-2 text-sm">
                <span className="truncate">{q.text}</span>
                <span className="shrink-0 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs text-amber-800 dark:text-amber-400">{q.count}×</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card className="p-4">
      <div className="text-2xl font-semibold">{value}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
      {hint && <div className="mt-1 text-[11px] text-muted-foreground">{hint}</div>}
    </Card>
  );
}

function Empty() {
  return <p className="py-6 text-center text-sm text-muted-foreground">No data yet — insights appear as jawans ask questions.</p>;
}
