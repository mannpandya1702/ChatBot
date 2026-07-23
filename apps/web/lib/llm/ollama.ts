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
  model: string,
  system: string,
  user: string,
  numPredict: number,
): Promise<string> {
  const res = await fetch(`${env.ollamaBaseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
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

// Generation uses the full model; the cheap classify/rewrite hops use the
// lighter tag if configured (OLLAMA_CLASSIFY_MODEL), else the same model.
const CLASSIFY_MODEL = env.ollamaClassifyModel || env.generationModel;

export const ollamaClassifier: Classifier = {
  async classify(input) {
    return parseLabel(await ollamaChat(CLASSIFY_MODEL, CLASSIFY_SYSTEM, classifyUserMessage(input), 16));
  },
};

export const ollamaRewriter: Rewriter = {
  async rewrite(history, latest) {
    return (await ollamaChat(CLASSIFY_MODEL, REWRITE_SYSTEM, rewriteUserPayload(history, latest), 256)).trim();
  },
};

export const ollamaGenerator: Generator = {
  async generate({ system, userPayload, maxOutputTokens }) {
    const text = await ollamaChat(env.generationModel, system, userPayload, maxOutputTokens);
    return { text, model: env.generationModel };
  },
};
