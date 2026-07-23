/**
 * Chat request pipeline (spec §3). Fail-closed and deny-by-default:
 *
 *   classify → (greeting | injection | out_of_scope handled inline)
 *   kb_question → rewrite → embed → hybrid_search (RLS-scoped) → rerank
 *     → best score < threshold ⇒ NOT_FOUND
 *     → generate (S7 prompt, context as untrusted data)
 *     → citation post-check: no valid [S#] ⇒ NOT_FOUND
 *   persist message + citations + latency; write audit event; record analytics.
 *
 * Audit stores metadata only (message id, refused, latency, classification) —
 * never the query text. The rewritten query goes only to query_analytics with
 * no user linkage.
 */
import {
  buildContext,
  citationPostCheck,
  type RetrievedSource,
} from "./citations";
import { greeting, injectionScope, notFoundMessage } from "./canned";
import { detectLanguage } from "../lang";
import { SYSTEM_PROMPT } from "../prompts/system";
import type { PipelineDeps, PipelineInput, PipelineResult } from "./types";

const TOP_K_SEARCH = 20;
const TOP_K_RERANK = 5;
const MAX_OUTPUT_TOKENS = 1200; // spec §8 output cap

function generationPayload(context: string, notFound: string, question: string): string {
  return (
    `If the numbered sources below do not answer the question, reply with exactly ` +
    `this message and nothing else:\n"${notFound}"\n\n` +
    `${context}\n\nQuestion: ${question}`
  );
}

export async function runChat(
  deps: PipelineDeps,
  input: PipelineInput,
): Promise<PipelineResult> {
  const now = deps.now ?? Date.now;
  const t0 = now();
  const lang = detectLanguage(input.message);
  const notFound = notFoundMessage(deps.settings, lang);

  await deps.db.persistUserMessage(input.conversationId, input.message);

  const classification = await deps.classifier.classify(input.message);

  // ── non-retrieval branches ────────────────────────────────────────────────
  if (classification === "greeting") {
    return finish(deps, input, {
      text: greeting(lang),
      refused: false,
      classification,
      citations: [],
      model: null,
      latencyMs: now() - t0,
      auditType: "greeting",
    });
  }
  if (classification === "injection_attempt") {
    return finish(deps, input, {
      text: injectionScope(lang),
      refused: false,
      classification,
      citations: [],
      model: null,
      latencyMs: now() - t0,
      auditType: "injection_attempt",
    });
  }
  if (classification === "out_of_scope") {
    return finish(deps, input, {
      text: notFound,
      refused: true,
      classification,
      citations: [],
      model: null,
      latencyMs: now() - t0,
      auditType: "out_of_scope",
    });
  }

  // ── kb_question: retrieval + grounded generation ──────────────────────────
  const query = (await deps.rewriter.rewrite(input.history, input.message)).trim()
    || input.message;

  const [embedding] = await deps.rag.embed([query]);
  const chunks = await deps.db.hybridSearch(embedding, query, TOP_K_SEARCH);

  const refuse = async (topScore: number | null): Promise<PipelineResult> => {
    await deps.db.recordQueryEvent(query, lang, true, topScore);
    return finish(deps, input, {
      text: notFound,
      refused: true,
      classification,
      citations: [],
      model: null,
      latencyMs: now() - t0,
      auditType: "refusal",
    });
  };

  if (chunks.length === 0) return refuse(null);

  const reranked = await deps.rag.rerank(
    query,
    chunks.map((c) => ({ id: c.chunkId, text: c.content })),
    TOP_K_RERANK,
  );
  const bestScore = reranked.length ? reranked[0].score : null;
  if (bestScore == null || bestScore < deps.settings.rerankRefusalThreshold) {
    return refuse(bestScore);
  }

  const byId = new Map(chunks.map((c) => [c.chunkId, c]));
  const sources: RetrievedSource[] = reranked.map((r, i) => {
    const c = byId.get(r.id)!;
    return {
      index: i + 1,
      chunkId: c.chunkId,
      documentTitle: c.documentTitle,
      pageStart: c.pageStart,
      pageEnd: c.pageEnd,
      content: c.content,
      score: r.score,
    };
  });

  const { text: answer, model } = await deps.generator.generate({
    system: SYSTEM_PROMPT,
    userPayload: generationPayload(buildContext(sources), notFound, query),
    maxOutputTokens: MAX_OUTPUT_TOKENS,
  });

  const checked = citationPostCheck(answer, sources, notFound);
  await deps.db.recordQueryEvent(query, lang, checked.refused, bestScore);
  return finish(deps, input, {
    text: checked.text,
    refused: checked.refused,
    classification,
    citations: checked.citations,
    model,
    latencyMs: now() - t0,
    auditType: checked.refused ? "refusal" : "query",
  });
}

async function finish(
  deps: PipelineDeps,
  input: PipelineInput,
  r: Omit<PipelineResult, "assistantMessageId"> & { auditType: string },
): Promise<PipelineResult> {
  const assistantMessageId = await deps.db.persistAssistantMessage(
    input.conversationId,
    {
      content: r.text,
      citations: r.citations,
      model: r.model,
      latencyMs: r.latencyMs,
      refused: r.refused,
    },
  );
  // Metadata only — never the query text (log minimisation, spec §4).
  await deps.db.logEvent(r.auditType, {
    conversation_id: input.conversationId,
    message_id: assistantMessageId,
    refused: r.refused,
    latency_ms: r.latencyMs,
    classification: r.classification,
  });
  return {
    text: r.text,
    refused: r.refused,
    classification: r.classification,
    citations: r.citations,
    model: r.model,
    latencyMs: r.latencyMs,
    assistantMessageId,
  };
}
