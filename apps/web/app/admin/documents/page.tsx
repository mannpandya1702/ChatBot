import { requireAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { DocumentsManager, type DocRow } from "./documents-manager";

export const metadata = { title: "Documents — Admin" };

export default async function DocumentsPage() {
  await requireAdmin();
  const { data } = await serviceClient()
    .from("documents")
    .select("id, title, original_filename, access_tier, status, page_count, chunk_count, error, created_at")
    .order("created_at", { ascending: false });

  return (
    <div>
      <h1 className="mb-1 text-lg font-semibold">Documents</h1>
      <p className="mb-5 text-sm text-muted-foreground">
        Upload approved PDFs to the knowledge base. Each is extracted (OCR fallback for scans), chunked, embedded, and
        made retrievable at its access tier. Answers are drawn only from these documents.
      </p>
      <DocumentsManager documents={(data ?? []) as DocRow[]} />
    </div>
  );
}
