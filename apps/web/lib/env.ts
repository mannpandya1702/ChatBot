/** Server-side environment (spec §16). Never import this into a client component. */
export const env = {
  llmProvider: (process.env.LLM_PROVIDER ?? "anthropic") as "anthropic" | "ollama",
  anthropicApiKey: process.env.ANTHROPIC_API_KEY ?? "",
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  generationModel: process.env.GENERATION_MODEL ?? "claude-sonnet-4-6",
  classifierModel: process.env.CLASSIFIER_MODEL ?? "claude-haiku-4-5",
  supabaseUrl: process.env.NEXT_PUBLIC_SUPABASE_URL ?? "",
  supabaseAnonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "",
  supabaseServiceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY ?? "",
  ragServiceUrl: process.env.RAG_SERVICE_URL ?? "http://localhost:8000",
  ragServiceSecret: process.env.RAG_SERVICE_SECRET ?? "",
  rerankRefusalThreshold: Number(process.env.RERANK_REFUSAL_THRESHOLD ?? "0.35"),
  ipAllowlist: (process.env.IP_ALLOWLIST ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
};
