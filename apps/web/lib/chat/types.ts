import type { Citation } from "./citations";

export type Classification =
  | "greeting"
  | "kb_question"
  | "out_of_scope"
  | "injection_attempt";

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RetrievedChunk {
  chunkId: string;
  documentId: string;
  documentTitle: string;
  content: string;
  pageStart: number | null;
  pageEnd: number | null;
}

export interface Bilingual {
  hi: string;
  en: string;
}

export interface AppSettings {
  notFound: Bilingual;
  escalationContact: Bilingual;
  rerankRefusalThreshold: number;
}

// ── injected collaborators (real impls in lib/llm, lib/rag, lib/chat/db) ──────
export interface Classifier {
  classify(input: string): Promise<Classification>;
}

export interface Rewriter {
  rewrite(history: ConversationMessage[], latest: string): Promise<string>;
}

export interface Generator {
  generate(p: {
    system: string;
    userPayload: string;
    maxOutputTokens: number;
  }): Promise<{ text: string; model: string }>;
}

export interface RagClient {
  embed(texts: string[]): Promise<number[][]>;
  rerank(
    query: string,
    passages: { id: string; text: string }[],
    topK: number,
  ): Promise<{ id: string; score: number; text: string }[]>;
}

export interface ChatDb {
  /** RLS-scoped to the caller's tier — MUST run with the user's JWT. */
  hybridSearch(
    embedding: number[],
    queryText: string,
    matchCount: number,
  ): Promise<RetrievedChunk[]>;
  persistUserMessage(conversationId: string, content: string): Promise<string>;
  persistAssistantMessage(
    conversationId: string,
    m: {
      content: string;
      citations: Citation[];
      model: string | null;
      latencyMs: number;
      refused: boolean;
    },
  ): Promise<string>;
  logEvent(eventType: string, detail: Record<string, unknown>): Promise<void>;
  recordQueryEvent(
    queryText: string,
    language: string,
    refused: boolean,
    topScore: number | null,
  ): Promise<void>;
}

export interface PipelineDeps {
  classifier: Classifier;
  rewriter: Rewriter;
  generator: Generator;
  rag: RagClient;
  db: ChatDb;
  settings: AppSettings;
  now?: () => number; // injectable clock for deterministic tests
}

export interface PipelineInput {
  conversationId: string;
  message: string;
  history: ConversationMessage[];
}

export interface PipelineResult {
  text: string;
  refused: boolean;
  classification: Classification;
  citations: Citation[];
  model: string | null;
  latencyMs: number;
  assistantMessageId: string;
}
