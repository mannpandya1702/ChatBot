/** Provider factory — one env var (LLM_PROVIDER) swaps the whole stack. */
import { env } from "../env";
import type { Classifier, Generator, Rewriter } from "../chat/types";
import {
  anthropicClassifier,
  anthropicGenerator,
  anthropicRewriter,
} from "./anthropic";
import { ollamaClassifier, ollamaGenerator, ollamaRewriter } from "./ollama";

export interface LlmProvider {
  classifier: Classifier;
  rewriter: Rewriter;
  generator: Generator;
}

export function getLlmProvider(): LlmProvider {
  if (env.llmProvider === "ollama") {
    return {
      classifier: ollamaClassifier,
      rewriter: ollamaRewriter,
      generator: ollamaGenerator,
    };
  }
  return {
    classifier: anthropicClassifier,
    rewriter: anthropicRewriter,
    generator: anthropicGenerator,
  };
}
