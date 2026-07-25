import { NextResponse } from "next/server";
import { currentAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { triggerIngest } from "@/lib/rag/ingest";
import {
  KB_BUCKET,
  MAX_UPLOAD_BYTES,
  firstIssue,
  registerRequestSchema,
  storagePathFor,
} from "@/lib/kb/upload";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  const admin = await currentAdmin();
  if (!admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { data } = await serviceClient()
    .from("documents")
    .select("id, title, original_filename, access_tier, status, page_count, chunk_count, error, created_at")
    .order("created_at", { ascending: false });
  return NextResponse.json({ documents: data ?? [] });
}

/**
 * Step 3 of the upload (lib/kb/upload.ts): the bytes are already in Storage —
 * record the document and queue ingestion. The file itself never passes through
 * here, so this stays a small, fast JSON call well inside any function timeout.
 *
 * What the client asserts is checked against what Storage actually holds: the
 * object must exist at the path derived from the id, and be within the size cap.
 * The client-computed sha256 is verified against the real bytes by the
 * rag-service when it downloads them, which fails the document on a mismatch —
 * so a wrong checksum can never quietly become a document's identity.
 */
export async function POST(req: Request): Promise<Response> {
  const admin = await currentAdmin();
  if (!admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const parsed = registerRequestSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: firstIssue(parsed.error) }, { status: 400 });
  }
  const { documentId, filename, sha256, accessTier, title } = parsed.data;

  const svc = serviceClient();
  const storagePath = storagePathFor(documentId);

  // The upload has to have actually landed. Without this an interrupted PUT
  // would still produce a documents row that only fails later, during ingest.
  // Listed rather than looked up with info(): list() is in every storage-api
  // version, including the self-hosted bundles the air-gap install ships with.
  const { data: matches } = await svc.storage
    .from(KB_BUCKET)
    .list("", { search: storagePath, limit: 1 });
  const object = matches?.find((o) => o.name === storagePath);
  if (!object) {
    return NextResponse.json({ error: "upload not found — please try again" }, { status: 400 });
  }
  if ((object.metadata?.size ?? 0) > MAX_UPLOAD_BYTES) {
    await svc.storage.from(KB_BUCKET).remove([storagePath]).catch(() => {});
    return NextResponse.json({ error: "file exceeds 50 MB" }, { status: 400 });
  }

  const { error: insErr } = await svc.from("documents").insert({
    id: documentId,
    title: title || filename.replace(/\.pdf$/i, ""),
    original_filename: filename,
    storage_path: storagePath,
    sha256,
    access_tier: accessTier,
    status: "processing",
    uploaded_by: admin.userId,
  });
  if (insErr) {
    // Drop the orphaned object either way — nothing references it now.
    await svc.storage.from(KB_BUCKET).remove([storagePath]).catch(() => {});
    // 23505: the sha256 unique constraint — a concurrent upload of the same
    // file won the race, or the pre-flight check was skipped.
    if ((insErr as { code?: string }).code === "23505") {
      return NextResponse.json({ error: "This document has already been uploaded." }, { status: 409 });
    }
    return NextResponse.json({ error: "could not record document" }, { status: 500 });
  }

  // Queued, not awaited: extraction with OCR runs for minutes. The rag-service
  // owns documents.status from here and the UI polls it.
  try {
    await triggerIngest(documentId);
    return NextResponse.json({ ok: true, documentId, status: "processing" });
  } catch {
    await svc
      .from("documents")
      .update({ status: "failed", error: "ingestion service unreachable" })
      .eq("id", documentId);
    return NextResponse.json({
      ok: true,
      documentId,
      status: "failed",
      warning: "ingestion failed — you can retry",
    });
  }
}
