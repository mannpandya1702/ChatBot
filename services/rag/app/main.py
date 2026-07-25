"""FastAPI surface for the rag-service.

Endpoints (all require header X-Service-Secret == RAG_SERVICE_SECRET, spec §6):
  GET  /health   liveness + backend/OCR report
  POST /embed    texts -> 1024-dim vectors
  POST /rerank   query + passages -> top_k sigmoid-scored results
  POST /ingest   {document_id, background?} -> pull from Storage,
                 extract/chunk/embed, upsert. With background=true the work is
                 queued and the call returns 202 immediately; the caller polls
                 documents.status, which this service owns.
"""
from __future__ import annotations

import logging
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

logger = logging.getLogger("rag")

from . import db
from .config import settings
from .embedder import get_embedder
from .ingest import IngestOutcome, ingest_pdf_bytes
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


def _warm_models() -> None:
    """Preload the embed/rerank models so the FIRST user request isn't a multi-GB
    cold load that would blow past client timeouts. Best-effort: on failure, log
    and fall back to lazy load on first use."""
    import logging

    for label, load in (("embedder", get_embedder), ("reranker", get_reranker)):
        try:
            load()
        except Exception as exc:  # pragma: no cover — best effort
            logging.getLogger("rag").warning("model warmup (%s) failed: %s", label, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Runs before the server accepts traffic → doubles as a readiness gate.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "rag-service starting (embed=%s rerank=%s ocr=%s)",
        settings.embedding_backend, settings.rerank_backend, settings.ocr_languages,
    )
    _warm_models()
    _start_ingest_workers()
    yield


app = FastAPI(title="Sainik Sahayak rag-service", version="0.1.0", lifespan=lifespan)

# Queued background ingests, drained by dedicated worker threads.
#
# Deliberately not FastAPI's BackgroundTasks: those run on the shared anyio
# threadpool, which is the same pool that serves /embed and /rerank. Since an
# ingest occupies its thread for minutes, a bulk upload would sit on those
# threads and stall live chat. Owning the threads keeps the two workloads
# independent, and makes the concurrency limit exact rather than advisory —
# `ingest_concurrency` is simply how many workers exist.
_INGEST_QUEUE: "queue.Queue[str]" = queue.Queue()
_workers_started = False


def _ingest_worker() -> None:
    while True:
        document_id = _INGEST_QUEUE.get()
        try:
            _run_ingest_background(document_id)
        finally:
            _INGEST_QUEUE.task_done()


def _start_ingest_workers() -> None:
    """Idempotent: the lifespan runs again per TestClient, and starting a fresh
    set of workers each time would quietly multiply the concurrency limit."""
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    for i in range(max(1, settings.ingest_concurrency)):
        threading.Thread(target=_ingest_worker, name=f"ingest-{i}", daemon=True).start()
    logger.info("ingest workers started count=%d", max(1, settings.ingest_concurrency))


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


def _ingest_document(document_id: str) -> IngestOutcome:
    """Fetch from Storage and run the pipeline. Shared by the synchronous and
    background paths. Raises if the document row is gone."""
    with db.connect() as conn:
        doc = db.get_document(conn, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        data = fetch_supabase_storage(doc["storage_path"])
        return ingest_pdf_bytes(
            conn, document_id, data, doc["access_tier"], expected_sha256=doc["sha256"]
        )


def _log_outcome(document_id: str, outcome: IngestOutcome) -> None:
    if outcome.status == "ready":
        logger.info(
            "ingest ok document_id=%s pages=%s chunks=%s",
            document_id, outcome.page_count, outcome.chunk_count,
        )
    else:
        logger.error("ingest failed document_id=%s error=%s", document_id, outcome.error)


def _run_ingest_background(document_id: str) -> None:
    """Run one queued ingest. Unlike the synchronous path there is no caller left
    to turn an exception into a failed status, so every escape route ends in an
    explicit mark_failed — otherwise the document sits in 'processing' forever
    and only the stale sweeper would ever clear it."""
    logger.info("ingest start (background) document_id=%s", document_id)
    try:
        _log_outcome(document_id, _ingest_document(document_id))
    except HTTPException:
        # Row deleted between queueing and running — nothing left to mark.
        logger.warning("ingest document not found document_id=%s", document_id)
    except Exception as exc:  # noqa: BLE001 — storage/network faults land here
        logger.error("ingest errored document_id=%s error=%s", document_id, exc)
        try:
            with db.connect() as conn:
                db.mark_failed(conn, document_id, str(exc))
                conn.commit()
        except Exception:  # pragma: no cover — DB itself is unreachable
            logger.exception("could not mark document failed document_id=%s", document_id)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_secret)])
def ingest(req: IngestRequest, response: Response) -> IngestResponse:
    if req.background:
        # Confirm the row exists before promising to process it, so a bad id is
        # still a synchronous 404 rather than a silent no-op.
        with db.connect() as conn:
            if db.get_document(conn, req.document_id) is None:
                logger.warning("ingest document not found document_id=%s", req.document_id)
                raise HTTPException(status_code=404, detail="document not found")
        _start_ingest_workers()  # no-op once running; covers a non-lifespan host
        _INGEST_QUEUE.put(req.document_id)
        logger.info("ingest queued document_id=%s", req.document_id)
        response.status_code = status.HTTP_202_ACCEPTED
        return IngestResponse(document_id=req.document_id, status="processing")

    logger.info("ingest start document_id=%s", req.document_id)
    outcome = _ingest_document(req.document_id)
    _log_outcome(req.document_id, outcome)
    return IngestResponse(
        document_id=req.document_id,
        status=outcome.status,
        page_count=outcome.page_count,
        chunk_count=outcome.chunk_count,
        error=outcome.error,
    )
