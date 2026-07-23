import { env } from "../env";

export interface IngestOutcome {
  document_id: string;
  status: "ready" | "failed" | "processing";
  page_count?: number | null;
  chunk_count?: number | null;
  error?: string | null;
}

/**
 * Trigger server-side ingestion of an already-uploaded document. The rag-service
 * pulls the file from the private `kb` Storage bucket, extracts (OCR fallback),
 * chunks, embeds, upserts chunks, and updates documents.status itself.
 */
export async function triggerIngest(documentId: string): Promise<IngestOutcome> {
  const r = await fetch(`${env.ragServiceUrl}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Service-Secret": env.ragServiceSecret },
    body: JSON.stringify({ document_id: documentId }),
  });
  if (!r.ok) throw new Error(`rag /ingest ${r.status}`);
  return (await r.json()) as IngestOutcome;
}
