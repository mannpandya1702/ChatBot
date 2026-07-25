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
 *
 * `background` (the default) returns as soon as the job is accepted — extraction
 * with OCR on a large scan runs for minutes, far past a serverless function's
 * execution cap, so the web tier must not wait for it. The UI polls
 * documents.status, which the rag-service is the sole writer of. Pass
 * `background: false` only where blocking is genuinely wanted (scripts, tests).
 */
export async function triggerIngest(
  documentId: string,
  { background = true }: { background?: boolean } = {},
): Promise<IngestOutcome> {
  const r = await fetch(`${env.ragServiceUrl}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Service-Secret": env.ragServiceSecret },
    body: JSON.stringify({ document_id: documentId, background }),
    // Queueing is a DB read plus an enqueue; only the blocking path needs to
    // sit through extract + OCR + embed.
    signal: AbortSignal.timeout(background ? 20_000 : 900_000),
  });
  if (!r.ok) throw new Error(`rag /ingest ${r.status}`);
  return (await r.json()) as IngestOutcome;
}
