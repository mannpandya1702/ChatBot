import { NextResponse } from "next/server";
import { createHash, randomUUID } from "node:crypto";
import { currentAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { triggerIngest } from "@/lib/rag/ingest";

export const runtime = "nodejs";
const MAX_BYTES = 50 * 1024 * 1024;
const BUCKET = "kb";

export async function GET(): Promise<Response> {
  const admin = await currentAdmin();
  if (!admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });
  const { data } = await serviceClient()
    .from("documents")
    .select("id, title, original_filename, access_tier, status, page_count, chunk_count, error, created_at")
    .order("created_at", { ascending: false });
  return NextResponse.json({ documents: data ?? [] });
}

export async function POST(req: Request): Promise<Response> {
  const admin = await currentAdmin();
  if (!admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const form = await req.formData().catch(() => null);
  const file = form?.get("file");
  const title = String(form?.get("title") ?? "").trim();
  const accessTier = Number(form?.get("accessTier") ?? "1");
  if (!(file instanceof File)) return NextResponse.json({ error: "no file" }, { status: 400 });
  // Require a positive PDF signal (content-type OR .pdf name) — an unknown type
  // with a non-.pdf name is rejected here; the rag magic-byte check backstops it.
  const looksPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!looksPdf) {
    return NextResponse.json({ error: "only PDF files are accepted" }, { status: 400 });
  }
  if (file.size > MAX_BYTES) return NextResponse.json({ error: "file exceeds 50 MB" }, { status: 400 });
  if (!Number.isInteger(accessTier) || accessTier < 1 || accessTier > 3) {
    return NextResponse.json({ error: "invalid access tier" }, { status: 400 });
  }

  const buffer = Buffer.from(await file.arrayBuffer());
  const sha256 = createHash("sha256").update(buffer).digest("hex");
  const svc = serviceClient();

  const { data: dup } = await svc.from("documents").select("id, title").eq("sha256", sha256).maybeSingle();
  if (dup) return NextResponse.json({ error: `Already uploaded as “${(dup as { title: string }).title}”.` }, { status: 409 });

  // Ensure the private bucket exists (idempotent; ignore "already exists").
  await svc.storage.createBucket(BUCKET, { public: false }).catch(() => {});

  const id = randomUUID();
  const storagePath = `${id}.pdf`;
  const up = await svc.storage.from(BUCKET).upload(storagePath, buffer, {
    contentType: "application/pdf",
    upsert: false,
  });
  if (up.error) return NextResponse.json({ error: "storage upload failed" }, { status: 500 });

  const { error: insErr } = await svc.from("documents").insert({
    id,
    title: title || file.name.replace(/\.pdf$/i, ""),
    original_filename: file.name,
    storage_path: storagePath,
    sha256,
    access_tier: accessTier,
    status: "processing",
    uploaded_by: admin.userId,
  });
  if (insErr) {
    await svc.storage.from(BUCKET).remove([storagePath]).catch(() => {});
    return NextResponse.json({ error: "could not record document" }, { status: 500 });
  }

  // Ingest synchronously; the rag-service updates documents.status itself.
  try {
    const outcome = await triggerIngest(id);
    return NextResponse.json({ ok: true, documentId: id, status: outcome.status });
  } catch {
    await svc.from("documents").update({ status: "failed", error: "ingestion service unreachable" }).eq("id", id);
    return NextResponse.json({ ok: true, documentId: id, status: "failed", warning: "ingestion failed — you can retry" });
  }
}
