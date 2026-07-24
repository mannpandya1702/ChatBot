"use client";
import { useEffect, useRef, useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/lib/ui/button";
import { Input } from "@/lib/ui/input";
import { Label, Card, Alert, Spinner } from "@/lib/ui/misc";
import { deleteDocumentAction, reingestAction } from "./actions";

export interface DocRow {
  id: string;
  title: string;
  original_filename: string;
  access_tier: number;
  status: "processing" | "ready" | "failed";
  page_count: number | null;
  chunk_count: number | null;
  error: string | null;
  created_at: string;
}

function StatusBadge({ status }: { status: DocRow["status"] }) {
  if (status === "ready") return <span className="inline-flex rounded-full bg-primary/15 px-2 py-0.5 text-xs text-primary">Ready</span>;
  if (status === "failed") return <span className="inline-flex rounded-full bg-destructive/15 px-2 py-0.5 text-xs text-destructive">Failed</span>;
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 px-2 py-0.5 text-xs text-amber-700 dark:text-amber-400">
      <Spinner className="h-3 w-3" /> Processing
    </span>
  );
}

export function DocumentsManager({ documents }: { documents: DocRow[] }) {
  const router = useRouter();
  const formRef = useRef<HTMLFormElement>(null);
  const [uploading, setUploading] = useState(false);
  const [msg, setMsg] = useState<{ kind: "error" | "success"; text: string } | null>(null);

  // Auto-refresh while anything is still processing.
  useEffect(() => {
    if (!documents.some((d) => d.status === "processing")) return;
    const t = setInterval(() => router.refresh(), 4000);
    return () => clearInterval(t);
  }, [documents, router]);

  async function onUpload(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    if (!(form.get("file") as File)?.size) {
      setMsg({ kind: "error", text: "Choose a PDF to upload." });
      return;
    }
    setUploading(true);
    setMsg(null);
    try {
      const res = await fetch("/api/admin/documents", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        setMsg({ kind: "error", text: data.error ?? "Upload failed." });
      } else {
        setMsg({
          kind: data.status === "failed" ? "error" : "success",
          text:
            data.status === "ready"
              ? "Uploaded and ingested."
              : data.status === "failed"
                ? "Uploaded, but ingestion failed — use Re-ingest once the service is up."
                : "Uploaded; processing…",
        });
        formRef.current?.reset();
        router.refresh();
      }
    } catch {
      setMsg({ kind: "error", text: "Network error during upload." });
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <form ref={formRef} onSubmit={onUpload} className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5 sm:col-span-2">
            <Label htmlFor="file">PDF file *</Label>
            <Input id="file" name="file" type="file" accept="application/pdf,.pdf" required />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="title">Title (optional)</Label>
            <Input id="title" name="title" placeholder="Defaults to the filename" />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="accessTier">Access tier</Label>
            <select id="accessTier" name="accessTier" defaultValue="1" className="flex h-10 w-full rounded-md border border-input bg-card px-3 text-sm">
              <option value="1">Tier 1 — all jawans</option>
              <option value="2">Tier 2</option>
              <option value="3">Tier 3 — most restricted</option>
            </select>
          </div>
          {msg && <div className="sm:col-span-2"><Alert variant={msg.kind}>{msg.text}</Alert></div>}
          <div className="sm:col-span-2">
            <Button type="submit" disabled={uploading}>{uploading && <Spinner />} Upload & ingest</Button>
          </div>
        </form>
      </Card>

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Title</th>
                <th className="px-4 py-2 font-medium">Tier</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Pages</th>
                <th className="px-4 py-2 font-medium">Chunks</th>
                <th className="px-4 py-2 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {documents.map((d) => (
                <tr key={d.id}>
                  <td className="px-4 py-2">
                    <div className="font-medium">{d.title}</div>
                    <div className="text-xs text-muted-foreground">{d.original_filename}</div>
                    {d.status === "failed" && d.error && <div className="mt-0.5 text-xs text-destructive">{d.error}</div>}
                  </td>
                  <td className="px-4 py-2">{d.access_tier}</td>
                  <td className="px-4 py-2"><StatusBadge status={d.status} /></td>
                  <td className="px-4 py-2">{d.page_count ?? "—"}</td>
                  <td className="px-4 py-2">{d.chunk_count ?? "—"}</td>
                  <td className="px-4 py-2">
                    <div className="flex justify-end gap-1">
                      {d.status !== "processing" && (
                        <form action={reingestAction}>
                          <input type="hidden" name="documentId" value={d.id} />
                          <Button type="submit" size="sm" variant="ghost">Re-ingest</Button>
                        </form>
                      )}
                      <form action={deleteDocumentAction}>
                        <input type="hidden" name="documentId" value={d.id} />
                        <Button type="submit" size="sm" variant="ghost" className="text-destructive hover:bg-destructive/10">Delete</Button>
                      </form>
                    </div>
                  </td>
                </tr>
              ))}
              {documents.length === 0 && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">No documents yet. Upload a PDF to get started.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
