import { env } from "../env";
import type { RagClient } from "../chat/types";

export function createRagClient(): RagClient {
  const headers = {
    "Content-Type": "application/json",
    "X-Service-Secret": env.ragServiceSecret,
  };
  return {
    async embed(texts) {
      const r = await fetch(`${env.ragServiceUrl}/embed`, {
        method: "POST",
        headers,
        body: JSON.stringify({ texts }),
        signal: AbortSignal.timeout(120_000),
      });
      if (!r.ok) throw new Error(`rag /embed ${r.status}`);
      return ((await r.json()) as { embeddings: number[][] }).embeddings;
    },
    async rerank(query, passages, topK) {
      const r = await fetch(`${env.ragServiceUrl}/rerank`, {
        method: "POST",
        headers,
        body: JSON.stringify({ query, passages, top_k: topK }),
        signal: AbortSignal.timeout(120_000),
      });
      if (!r.ok) throw new Error(`rag /rerank ${r.status}`);
      return ((await r.json()) as {
        results: { id: string; score: number; text: string }[];
      }).results;
    },
  };
}
