import { createAnthropic } from "@ai-sdk/anthropic";
import { generateText, streamText } from "ai";
import { env } from "../env";
import { CLASSIFY_SYSTEM, classifyUserMessage } from "../prompts/classify";
import { REWRITE_SYSTEM, rewriteUserPayload } from "../prompts/rewrite";
import type { Classification, Classifier, Generator, Rewriter } from "../chat/types";

const anthropic = createAnthropic({ apiKey: env.anthropicApiKey });

const LABELS: Classification[] = [
  "greeting",
  "kb_question",
  "out_of_scope",
  "injection_attempt",
];

/** Parse the model's label output; fail toward retrieval (kb_question) so the
 * citation gate — not the classifier — has the final say on grounding. */
export function parseLabel(raw: string): Classification {
  const t = raw.toLowerCase();
  for (const l of LABELS) if (t.includes(l)) return l;
  return "kb_question";
}

export const anthropicClassifier: Classifier = {
  async classify(input) {
    const { text } = await generateText({
      model: anthropic(env.classifierModel),
      system: CLASSIFY_SYSTEM,
      prompt: classifyUserMessage(input),
      maxOutputTokens: 16,
    });
    return parseLabel(text);
  },
};

export const anthropicRewriter: Rewriter = {
  async rewrite(history, latest) {
    const { text } = await generateText({
      model: anthropic(env.classifierModel),
      system: REWRITE_SYSTEM,
      prompt: rewriteUserPayload(history, latest),
      maxOutputTokens: 256,
    });
    return text.trim();
  },
};

export const anthropicGenerator: Generator = {
  async generate({ system, userPayload, maxOutputTokens }) {
    // Stream from Anthropic (spec §3) but buffer the full text server-side so
    // the citation post-check can fail closed before anything is committed.
    const result = streamText({
      model: anthropic(env.generationModel),
      system,
      prompt: userPayload,
      maxOutputTokens,
    });
    const text = await result.text;
    return { text, model: env.generationModel };
  },
};
