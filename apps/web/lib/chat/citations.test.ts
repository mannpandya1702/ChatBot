import { describe, it, expect } from "vitest";
import {
  buildContext,
  citationPostCheck,
  citedIndices,
  type RetrievedSource,
} from "./citations";

const sources: RetrievedSource[] = [
  { index: 1, chunkId: "c1", documentTitle: "Leave Rules 2025", pageStart: 3, pageEnd: 4, content: "Apply via adjutant.", score: 0.9 },
  { index: 2, chunkId: "c2", documentTitle: "Pension Manual", pageStart: 12, pageEnd: null, content: "Family pension steps.", score: 0.7 },
];

describe("buildContext", () => {
  it("labels sources S1.. with title and page ranges, wrapped as untrusted data", () => {
    const ctx = buildContext(sources);
    expect(ctx.startsWith("<context>")).toBe(true);
    expect(ctx.trimEnd().endsWith("</context>")).toBe(true);
    expect(ctx).toContain('[S1] "Leave Rules 2025" (pp. 3–4)');
    expect(ctx).toContain('[S2] "Pension Manual" (p. 12)');
    expect(ctx).toContain("Apply via adjutant.");
  });
});

describe("citedIndices", () => {
  it("extracts distinct cited source numbers", () => {
    expect(citedIndices("Do X [S1]. Then Y [S2][S1].").sort()).toEqual([1, 2]);
    expect(citedIndices("No citations here.")).toEqual([]);
  });
});

describe("citationPostCheck (fail closed)", () => {
  const NF = "This information is not available.";

  it("keeps a grounded, cited answer and maps citations", () => {
    const r = citationPostCheck("Apply via the adjutant [S1].", sources, NF);
    expect(r.refused).toBe(false);
    expect(r.text).toContain("[S1]");
    expect(r.citations).toEqual([
      { s: 1, chunk_id: "c1", document_title: "Leave Rules 2025", page_start: 3, page_end: 4 },
    ]);
  });

  it("refuses an answer with NO citation", () => {
    const r = citationPostCheck("The capital of France is Paris.", sources, NF);
    expect(r.refused).toBe(true);
    expect(r.text).toBe(NF);
    expect(r.citations).toEqual([]);
  });

  it("refuses a HALLUCINATED citation to a non-existent source", () => {
    const r = citationPostCheck("Some claim [S9].", sources, NF);
    expect(r.refused).toBe(true);
    expect(r.text).toBe(NF);
  });

  it("keeps only valid citations when mixed with an invalid one", () => {
    const r = citationPostCheck("A [S2] and bogus [S7].", sources, NF);
    expect(r.refused).toBe(false);
    expect(r.citations.map((c) => c.s)).toEqual([2]);
  });
});
