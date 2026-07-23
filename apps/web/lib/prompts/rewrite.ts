/**
 * Multi-turn query rewrite (Haiku). Turns a possibly context-dependent latest
 * message into a standalone retrieval query using the last few turns. Returns
 * only the rewritten query.
 */
export const REWRITE_SYSTEM = `You rewrite a user's latest message into a single standalone search query for a knowledge base of Indian Army service documents.

Rules:
- Resolve pronouns and references using the conversation so far ("it", "that", "uske liye", etc.).
- Keep the user's original language and terminology.
- Output ONLY the rewritten query text — no preamble, no quotes, no explanation.
- If the latest message is already standalone, return it unchanged.`;

export function rewriteUserPayload(
  history: { role: "user" | "assistant"; content: string }[],
  latest: string,
): string {
  const convo = history
    .slice(-6)
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
    .join("\n");
  return `Conversation so far:\n${convo || "(none)"}\n\nLatest message: ${latest}\n\nStandalone query:`;
}
