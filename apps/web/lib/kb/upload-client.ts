"use client";
/**
 * Browser half of the three-step upload described in lib/kb/upload.ts.
 * Kept out of the component so the wire details stay in one place.
 */
import { sha256HexOf } from "./sha256";

export type UploadPhase = "hashing" | "uploading" | "finishing";

export interface UploadProgress {
  phase: UploadPhase;
  /** 0–100 during "uploading"; undefined for the indeterminate phases. */
  percent?: number;
}

export class UploadError extends Error {}

/**
 * PUT the file at a Supabase signed upload URL.
 *
 * Done with XHR rather than `fetch`, because only XHR reports upload progress —
 * and a 50 MB scan over a phone tether with no progress bar looks like a hang.
 * The body shape is Storage's documented multipart upload form: a `cacheControl`
 * field and the file under an empty field name.
 */
function putToSignedUrl(
  signedUrl: string,
  file: File,
  onProgress: (percent: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    body.append("cacheControl", "3600");
    body.append("", file);

    const xhr = new XMLHttpRequest();
    xhr.open("PUT", signedUrl);
    xhr.setRequestHeader("x-upsert", "false");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new UploadError(`Storage rejected the upload (${xhr.status}).`));
    xhr.onerror = () => reject(new UploadError("Network error during upload."));
    xhr.onabort = () => reject(new UploadError("Upload cancelled."));
    xhr.send(body);
  });
}

async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new UploadError((data as { error?: string }).error ?? "Upload failed.");
  return data as T;
}

export interface UploadResult {
  documentId: string;
  status: "processing" | "failed";
  warning?: string;
}

/** Hash → mint signed URL → PUT to Storage → register + queue ingestion. */
export async function uploadDocument(
  { file, title, accessTier }: { file: File; title: string; accessTier: number },
  onProgress: (p: UploadProgress) => void,
): Promise<UploadResult> {
  onProgress({ phase: "hashing" });
  const sha256 = await sha256HexOf(new Uint8Array(await file.arrayBuffer()));

  const { documentId, signedUrl } = await postJson<{ documentId: string; signedUrl: string }>(
    "/api/admin/documents/upload-url",
    { filename: file.name, size: file.size, sha256 },
  );

  onProgress({ phase: "uploading", percent: 0 });
  await putToSignedUrl(signedUrl, file, (percent) => onProgress({ phase: "uploading", percent }));

  onProgress({ phase: "finishing" });
  return postJson<UploadResult>("/api/admin/documents", {
    documentId,
    filename: file.name,
    sha256,
    accessTier,
    title: title || undefined,
  });
}
