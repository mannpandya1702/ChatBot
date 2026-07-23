/**
 * Context building + the mandatory citation post-check (spec §3, §8).
 *
 * Fail closed: an assistant answer that does not cite at least one VALID source
 * (a [S#] that maps to a source we actually provided) is discarded and replaced
 * with the NOT_FOUND message. A citation to a non-existent source index counts
 * as no citation — a hallucinated reference must never survive.
 */

export interface RetrievedSource {
  index: number; // 1-based label used as [S{index}]
  chunkId: string;
  documentTitle: string;
  pageStart: number | null;
  pageEnd: number | null;
  content: string;
  score: number;
}

export interface Citation {
  s: number;
  chunk_id: string;
  document_title: string;
  page_start: number | null;
  page_end: number | null;
}

const CITE = /\[S(\d+)\]/g;

function pages(s: RetrievedSource): string {
  if (s.pageStart == null) return "";
  if (s.pageEnd == null || s.pageEnd === s.pageStart) return ` (p. ${s.pageStart})`;
  return ` (pp. ${s.pageStart}–${s.pageEnd})`;
}

/** Wrap the top sources as untrusted data for the generation model. */
export function buildContext(sources: RetrievedSource[]): string {
  const blocks = sources.map(
    (s) => `[S${s.index}] "${s.documentTitle}"${pages(s)}\n${s.content}`,
  );
  return `<context>\n${blocks.join("\n\n")}\n</context>`;
}

/** Distinct source indices actually cited in the text. */
export function citedIndices(text: string): number[] {
  const found = new Set<number>();
  for (const m of text.matchAll(CITE)) found.add(Number(m[1]));
  return [...found];
}

export interface PostCheckResult {
  text: string;
  refused: boolean;
  citations: Citation[];
}

/**
 * Enforce the citation rule. `notFound` is the message to substitute when the
 * answer is not grounded.
 */
export function citationPostCheck(
  answer: string,
  sources: RetrievedSource[],
  notFound: string,
): PostCheckResult {
  const valid = citedIndices(answer).filter(
    (n) => n >= 1 && n <= sources.length,
  );
  if (valid.length === 0) {
    return { text: notFound, refused: true, citations: [] };
  }
  const byIndex = new Map(sources.map((s) => [s.index, s]));
  const citations: Citation[] = valid
    .sort((a, b) => a - b)
    .map((n) => {
      const s = byIndex.get(n)!;
      return {
        s: n,
        chunk_id: s.chunkId,
        document_title: s.documentTitle,
        page_start: s.pageStart,
        page_end: s.pageEnd,
      };
    });
  return { text: answer, refused: false, citations };
}
