import { requireAdmin } from "@/lib/auth/admin-guard";

export const metadata = { title: "Documents — Admin" };

export default async function DocumentsPage() {
  await requireAdmin();
  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Documents</h1>
      <p className="text-sm text-muted-foreground">Knowledge-base upload, ingestion status, and tier management — building next.</p>
    </div>
  );
}
