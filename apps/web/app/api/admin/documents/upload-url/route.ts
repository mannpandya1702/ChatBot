import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { currentAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { KB_BUCKET, firstIssue, storagePathFor, uploadUrlRequestSchema } from "@/lib/kb/upload";

export const runtime = "nodejs";

/**
 * Mint a one-object signed upload URL so the browser can PUT the PDF straight
 * into the private `kb` bucket (see lib/kb/upload.ts for why). The URL is
 * scoped to a single path we generate, is good for two hours, and grants
 * nothing else — the bucket stays private and unreadable without the service
 * key. No documents row is created here: an abandoned upload leaves at most an
 * orphaned object, never a phantom document.
 */
export async function POST(req: Request): Promise<Response> {
  const admin = await currentAdmin();
  if (!admin) return NextResponse.json({ error: "forbidden" }, { status: 403 });

  const parsed = uploadUrlRequestSchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: firstIssue(parsed.error) }, { status: 400 });
  }

  const svc = serviceClient();

  // Cheap pre-flight so a duplicate costs nothing to discover; the sha256
  // unique constraint is what actually enforces it at register time.
  const { data: dup } = await svc
    .from("documents")
    .select("title")
    .eq("sha256", parsed.data.sha256)
    .maybeSingle();
  if (dup) {
    const { title } = dup as { title: string };
    return NextResponse.json({ error: `Already uploaded as “${title}”.` }, { status: 409 });
  }

  // Idempotent; ignores "already exists".
  await svc.storage.createBucket(KB_BUCKET, { public: false }).catch(() => {});

  const documentId = randomUUID();
  const path = storagePathFor(documentId);
  const { data, error } = await svc.storage.from(KB_BUCKET).createSignedUploadUrl(path);
  if (error || !data) {
    return NextResponse.json({ error: "could not prepare upload" }, { status: 500 });
  }

  return NextResponse.json({ documentId, path, token: data.token, signedUrl: data.signedUrl });
}
