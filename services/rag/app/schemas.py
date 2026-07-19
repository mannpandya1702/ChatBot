"""Request/response models for the rag-service API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int


class RerankPassage(BaseModel):
    id: str  # opaque caller-supplied id (e.g. chunk id) echoed back
    text: str


class RerankRequest(BaseModel):
    query: str
    passages: list[RerankPassage] = Field(min_length=1)
    top_k: int = 5


class RerankResult(BaseModel):
    id: str
    score: float  # sigmoid-normalised relevance in (0, 1)
    text: str


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str


class IngestRequest(BaseModel):
    document_id: str


class IngestResponse(BaseModel):
    document_id: str
    status: str            # 'ready' | 'failed'
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    rerank_backend: str
    ocr_languages: list[str]
