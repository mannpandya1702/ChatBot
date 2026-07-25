import { describe, expect, it } from "vitest";
import {
  MAX_UPLOAD_BYTES,
  registerRequestSchema,
  storagePathFor,
  uploadUrlRequestSchema,
} from "./upload";

const SHA = "a".repeat(64);
const ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301";

describe("uploadUrlRequestSchema", () => {
  const valid = { filename: "leave-policy.pdf", size: 1024, sha256: SHA };

  it("accepts a well-formed request", () => {
    expect(uploadUrlRequestSchema.safeParse(valid).success).toBe(true);
  });

  it("accepts a PDF regardless of extension case", () => {
    expect(uploadUrlRequestSchema.safeParse({ ...valid, filename: "SOP.PDF" }).success).toBe(true);
  });

  it("rejects anything that is not a .pdf", () => {
    for (const filename of ["notes.txt", "scan.pdf.exe", "archive.zip", "pdf"]) {
      expect(uploadUrlRequestSchema.safeParse({ ...valid, filename }).success, filename).toBe(false);
    }
  });

  it("rejects an empty file and one over the 50 MB cap", () => {
    expect(uploadUrlRequestSchema.safeParse({ ...valid, size: 0 }).success).toBe(false);
    expect(uploadUrlRequestSchema.safeParse({ ...valid, size: MAX_UPLOAD_BYTES + 1 }).success).toBe(false);
    expect(uploadUrlRequestSchema.safeParse({ ...valid, size: MAX_UPLOAD_BYTES }).success).toBe(true);
  });

  it("rejects a checksum that is not 64 lowercase hex characters", () => {
    for (const sha256 of [SHA.toUpperCase(), SHA.slice(0, 63), SHA + "a", "z".repeat(64), ""]) {
      expect(uploadUrlRequestSchema.safeParse({ ...valid, sha256 }).success, sha256).toBe(false);
    }
  });
});

describe("registerRequestSchema", () => {
  const valid = { documentId: ID, filename: "sop.pdf", sha256: SHA, accessTier: 1 };

  it("accepts a well-formed request and an optional title", () => {
    expect(registerRequestSchema.safeParse(valid).success).toBe(true);
    expect(registerRequestSchema.safeParse({ ...valid, title: "Leave SOP" }).success).toBe(true);
  });

  it("holds the access tier inside the 1–3 range", () => {
    for (const accessTier of [0, 4, -1, 1.5]) {
      expect(registerRequestSchema.safeParse({ ...valid, accessTier }).success, `${accessTier}`).toBe(false);
    }
    for (const accessTier of [1, 2, 3]) {
      expect(registerRequestSchema.safeParse({ ...valid, accessTier }).success, `${accessTier}`).toBe(true);
    }
  });

  it("rejects a document id that is not a uuid — the storage path derives from it", () => {
    for (const documentId of ["../secrets", "not-a-uuid", `${ID}/../other`, ""]) {
      expect(registerRequestSchema.safeParse({ ...valid, documentId }).success, documentId).toBe(false);
    }
  });

  it("trims the title and caps its length", () => {
    const parsed = registerRequestSchema.safeParse({ ...valid, title: "  Pension rules  " });
    expect(parsed.success && parsed.data.title).toBe("Pension rules");
    expect(registerRequestSchema.safeParse({ ...valid, title: "x".repeat(201) }).success).toBe(false);
  });
});

describe("storagePathFor", () => {
  it("keeps the object under a name derived solely from the id", () => {
    expect(storagePathFor(ID)).toBe(`${ID}.pdf`);
  });
});
