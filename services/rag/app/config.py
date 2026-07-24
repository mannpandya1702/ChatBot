"""Environment-driven configuration for the rag-service."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # Shared-secret header every endpoint requires (spec §6).
    service_secret: str = "dev-insecure-secret-change-me"

    # Postgres the service writes chunks/documents to. In production this is the
    # Supabase Postgres connection string using the service role (bypasses RLS).
    database_url: str = "postgresql:///sainik_test?host=/var/run/postgresql"

    # Backends: 'bge' loads the real models; 'hashing' is a deterministic test
    # double (1024-dim) used when torch/FlagEmbedding is unavailable.
    embedding_backend: str = "bge"
    rerank_backend: str = "bge"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024
    embed_batch_size: int = 32
    # Reranker (query, passage) truncation window. Must exceed a max-size chunk
    # (chunk_max_tokens) + the query, or answers in a chunk's tail get truncated
    # away before scoring and can be wrongly refused. bge-reranker-v2-m3 supports
    # long context, so 1024 comfortably covers a 900-token chunk.
    rerank_max_length: int = 1024

    # Token counter for chunk sizing. 'auto' uses the real XLM-R tokenizer when
    # transformers is installed (production accuracy) and falls back to the
    # script-aware heuristic otherwise. Pinned explicitly so chunk boundaries
    # are never silently altered by what happens to be installed. 'hf' forces
    # the model tokenizer (errors if unavailable); 'heuristic' forces the
    # dependency-free counter (used in tests for determinism).
    token_backend: str = "auto"  # auto | hf | heuristic

    # Chunking (spec §6).
    chunk_target_tokens: int = 600
    chunk_max_tokens: int = 900
    chunk_overlap_ratio: float = 0.15
    ocr_min_chars: int = 40  # pages below this many extracted chars get OCR'd
    ocr_languages: str = "hin+eng"
    ocr_dpi: int = 300

    # Ingestion guards.
    max_pdf_bytes: int = 50 * 1024 * 1024  # 50 MB cap

    # Supabase Storage (production file source).
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    storage_bucket: str = "kb"


settings = Settings()
