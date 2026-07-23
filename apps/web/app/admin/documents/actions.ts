"use server";
import { revalidatePath } from "next/cache";
import { currentAdmin } from "@/lib/auth/admin-guard";
import { serviceClient } from "@/lib/supabase/service";
import { triggerIngest } from "@/lib/rag/ingest";

const BUCKET = "kb";

export async function deleteDocumentAction(form: FormData): Promise<void> {
  const admin = await currentAdmin();
  if (!admin) return;
  const id = String(form.get("documentId") ?? "");
  if (!id) return;

  const svc = serviceClient();
  const { data: doc } = await svc.from("documents").select("storage_path").eq("id", id).single();
  if (doc?.storage_path) await svc.storage.from(BUCKET).remove([doc.storage_path as string]).catch(() => {});
  await svc.from("documents").delete().eq("id", id); // chunks cascade
  revalidatePath("/admin/documents");
}

export async function reingestAction(form: FormData): Promise<void> {
  const admin = await currentAdmin();
  if (!admin) return;
  const id = String(form.get("documentId") ?? "");
  if (!id) return;

  const svc = serviceClient();
  await svc.from("documents").update({ status: "processing", error: null }).eq("id", id);
  try {
    await triggerIngest(id);
  } catch {
    await svc.from("documents").update({ status: "failed", error: "ingestion service unreachable" }).eq("id", id);
  }
  revalidatePath("/admin/documents");
}
