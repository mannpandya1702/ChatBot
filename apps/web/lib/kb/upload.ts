/**
 * Knowledge-base upload contract, shared by the two halves of the admin upload.
 *
 * PDFs go **browser → Supabase Storage directly**, via a short-lived signed
 * upload URL, and never through a Next.js route handler. That is not a
 * micro-optimisation: a serverless request body is capped well below the 50 MB
 * document cap (4.5 MB on Vercel), so posting the file to our own API would
 * reject every real document. Uploading straight to Storage also keeps a slow
 * upload off the function execution budget entirely.
 *
 * The exchange is therefore three steps:
 *   1. POST /api/admin/documents/upload-url  → mint a signed URL for one path
 *   2. PUT  <signed URL>                     → the bytes, direct to Storage
 *   3. POST /api/admin/documents             → record the row, queue ingestion
 */
import { z } from "zod";
import { MAX_TITLE_CHARS, MAX_UPLOAD_BYTES } from "./limits";

export { KB_BUCKET, MAX_TITLE_CHARS, MAX_UPLOAD_BYTES, storagePathFor } from "./limits";

const accessTier = z.coerce.number().int().min(1).max(3);
const pdfFilename = z
  .string()
  .min(1)
  .max(255)
  .refine((n) => n.toLowerCase().endsWith(".pdf"), "only PDF files are accepted");
/** Lowercase hex SHA-256, as produced by WebCrypto in the browser. */
const sha256 = z.string().regex(/^[0-9a-f]{64}$/, "invalid checksum");

/** Step 1 — what the client must state before we hand out an upload URL. */
export const uploadUrlRequestSchema = z.object({
  filename: pdfFilename,
  size: z.coerce.number().int().positive().max(MAX_UPLOAD_BYTES, "file exceeds 50 MB"),
  sha256,
});

/** Step 3 — what the client sends once the bytes are in Storage. */
export const registerRequestSchema = z.object({
  documentId: z.string().uuid(),
  filename: pdfFilename,
  sha256,
  accessTier,
  title: z.string().trim().max(MAX_TITLE_CHARS).optional(),
});

export type UploadUrlRequest = z.infer<typeof uploadUrlRequestSchema>;
export type RegisterRequest = z.infer<typeof registerRequestSchema>;

/** First zod issue as a short, user-facing sentence. */
export function firstIssue(error: z.ZodError): string {
  const issue = error.issues[0];
  return issue?.message ?? "invalid request";
}
