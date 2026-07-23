/**
 * Ollama provider for air-gapped deployments (spec §2). Same prompts as the
 * Anthropic path — only the transport differs. Dependency-free (plain fetch to
 * the Ollama /api/chat endpoint), selected by LLM_PROVIDER=ollama.
 */
import { env } from "../env";
import { CLASSIFY_SYSTEM, classifyUserMessage } from "../prompts/classify";
import { REWRITE_SYSTEM, rewriteUserPayload } from "../prompts/rewrite";
import { parseLabel } from "./anthropic";
import type { Classifier, Generator, Rewriter } from "../chat/types";

async function ollamaChat(
  system: string,
  user: string,
  numPredict: number,
): Promise<string> {
  const res = await fetch(`${env.ollamaBaseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: env.generationModel, // for ollama, GENERATION_MODEL holds the ollama tag
      stream: false,
      options: { num_predict: numPredict, temperature: 0 },
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama /api/chat ${res.status}`);
  const data = (await res.json()) as { message?: { content?: string } };
  return data.message?.content ?? "";
}

export const ollamaClassifier: Classifier = {
  async classify(input) {
    return parseLabel(await ollamaChat(CLASSIFY_SYSTEM, classifyUserMessage(input), 16));
  },
};

export const ollamaRewriter: Rewriter = {
  async rewrite(history, latest) {
    return (await ollamaChat(REWRITE_SYSTEM, rewriteUserPayload(history, latest), 256)).trim();
  },
};

export const ollamaGenerator: Generator = {
  async generate({ system, userPayload, maxOutputTokens }) {
    const text = await ollamaChat(system, userPayload, maxOutputTokens);
    return { text, model: env.generationModel };
  },
};
