"""FastAPI surface for the rag-service.

Endpoints (all require header X-Service-Secret == RAG_SERVICE_SECRET, spec §6):
  GET  /health   liveness + backend/OCR report
  POST /embed    texts -> 1024-dim vectors
  POST /rerank   query + passages -> top_k sigmoid-scored results
  POST /ingest   {document_id} -> pull from Storage, extract/chunk/embed, upsert
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status

from . import db
from .config import settings
from .embedder import get_embedder
from .ingest import ingest_pdf_bytes
from .reranker import get_reranker
from .schemas import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RerankRequest,
    RerankResponse,
)
from .sources import fetch_supabase_storage

app = FastAPI(title="Sainik Sahayak rag-service", version="0.1.0")


def _validate_service_secret() -> None:
    """Refuse to start with an unset or built-in-default shared secret: an empty
    secret would let an empty header authenticate, and the default is published
    in the repo. Local dev may override with RAG_ALLOW_INSECURE_SECRET=1."""
    import os

    if settings.service_secret in ("", "dev-insecure-secret-change-me") and \
            os.getenv("RAG_ALLOW_INSECURE_SECRET") != "1":
        raise RuntimeError(
            "RAG_SERVICE_SECRET is unset or the built-in default — refusing to start. "
            "Set a strong secret (e.g. `openssl rand -hex 32`); for local dev only, "
            "export RAG_ALLOW_INSECURE_SECRET=1."
        )


_validate_service_secret()


def require_secret(x_service_secret: str = Header(default="")) -> None:
    # Constant-time compare to avoid leaking the secret via timing.
    import hmac

    if not hmac.compare_digest(x_service_secret, settings.service_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Service-Secret",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Report configured backends without forcing a model load.
    return HealthResponse(
        status="ok",
        embedding_backend=settings.embedding_backend,
        rerank_backend=settings.rerank_backend,
        ocr_languages=settings.ocr_languages.split("+"),
    )


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_secret)])
def embed(req: EmbedRequest) -> EmbedResponse:
    embedder = get_embedder()
    vecs = embedder.embed(req.texts)
    return EmbedResponse(embeddings=vecs, model=settings.embedding_model, dim=len(vecs[0]))


@app.post("/rerank", response_model=RerankResponse, dependencies=[Depends(require_secret)])
def rerank(req: RerankRequest) -> RerankResponse:
    reranker = get_reranker()
    results = reranker.rerank(req.query, req.passages, req.top_k)
    return RerankResponse(results=results, model=settings.rerank_model)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_secret)])
def ingest(req: IngestRequest) -> IngestResponse:
    with db.connect() as conn:
        doc = db.get_document(conn, req.document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        data = fetch_supabase_storage(doc["storage_path"])
        outcome = ingest_pdf_bytes(conn, req.document_id, data, doc["access_tier"])
    return IngestResponse(
        document_id=req.document_id,
        status=outcome.status,
        page_count=outcome.page_count,
        chunk_count=outcome.chunk_count,
        error=outcome.error,
    )
