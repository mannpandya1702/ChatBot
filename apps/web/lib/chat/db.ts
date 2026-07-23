import { serviceClient, userClient } from "../supabase/service";
import type { ChatDb, RetrievedChunk } from "./types";

function toVectorLiteral(embedding: number[]): string {
  return `[${embedding.join(",")}]`;
}

/**
 * Real ChatDb. hybrid_search runs on the USER client so RLS tier-scopes
 * retrieval; message/analytics writes run on the service client (message
 * inserts are service-role only by RLS); log_event runs on the user client so
 * the audit row is attributed to the user.
 */
export function createChatDb(userAccessToken: string): ChatDb {
  const user = userClient(userAccessToken);
  const svc = serviceClient();

  return {
    async hybridSearch(embedding, queryText, matchCount) {
      const { data, error } = await user.rpc("hybrid_search", {
        p_query_embedding: toVectorLiteral(embedding),
        p_query_text: queryText,
        p_match_count: matchCount,
      });
      if (error) throw new Error(`hybrid_search: ${error.message}`);
      return ((data ?? []) as Record<string, unknown>[]).map(
        (r): RetrievedChunk => ({
          chunkId: String(r.chunk_id),
          documentId: String(r.document_id),
          documentTitle: String(r.doc_title),
          content: String(r.content),
          pageStart: (r.page_start as number) ?? null,
          pageEnd: (r.page_end as number) ?? null,
        }),
      );
    },

    async persistUserMessage(conversationId, content) {
      const { data, error } = await svc
        .from("messages")
        .insert({ conversation_id: conversationId, role: "user", content })
        .select("id")
        .single();
      if (error) throw new Error(`persist user message: ${error.message}`);
      return (data as { id: string }).id;
    },

    async persistAssistantMessage(conversationId, m) {
      const { data, error } = await svc
        .from("messages")
        .insert({
          conversation_id: conversationId,
          role: "assistant",
          content: m.content,
          citations: m.citations,
          model: m.model,
          latency_ms: m.latencyMs,
          refused: m.refused,
        })
        .select("id")
        .single();
      if (error) throw new Error(`persist assistant message: ${error.message}`);
      return (data as { id: string }).id;
    },

    async logEvent(eventType, detail) {
      const { error } = await user.rpc("log_event", {
        p_event_type: eventType,
        p_detail: detail,
      });
      if (error) throw new Error(`log_event: ${error.message}`);
    },

    async recordQueryEvent(queryText, language, refused, topScore) {
      const { error } = await svc.rpc("record_query_event", {
        p_query_text: queryText,
        p_language: language,
        p_refused: refused,
        p_top_score: topScore,
      });
      if (error) throw new Error(`record_query_event: ${error.message}`);
    },
  };
}
