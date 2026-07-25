/**
 * Upload limits and paths — deliberately dependency-free.
 *
 * The client component needs the size cap, and the request schemas next door in
 * upload.ts need it too. Keeping these here means importing a constant in the
 * browser doesn't drag zod into the page bundle with it.
 */
export const KB_BUCKET = "kb";
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // mirrors rag `max_pdf_bytes`
export const MAX_TITLE_CHARS = 200;

/** Storage object path for a document. Derived from the id we generate — never
 * taken from the client, so one upload can never target another's object. */
export function storagePathFor(documentId: string): string {
  return `${documentId}.pdf`;
}
