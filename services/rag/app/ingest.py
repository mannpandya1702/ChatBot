"""Ingestion orchestration (spec §6).

pull file → validate (magic bytes, size) → extract (+OCR fallback) → chunk →
embed in batches → upsert chunks → mark document ready. Any failure marks the
document failed with the real error string (never a silent drop).
"""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass

from . import db
from .config import settings
from .chunker import chunk_blocks
from .embedder import get_embedder
from .extract import extract_pdf


class IngestError(Exception):
    pass


@dataclass
class IngestOutcome:
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    language: str | None = None


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) > settings.max_pdf_bytes:
        raise IngestError(
            f"file exceeds {settings.max_pdf_bytes // (1024 * 1024)} MB cap"
        )
    if data[:5] != b"%PDF-":
        raise IngestError("not a PDF (missing %PDF- magic bytes)")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest_pdf_bytes(
    conn,
    document_id: str,
    data: bytes,
    access_tier: int,
) -> IngestOutcome:
    """Extract → chunk → embed → upsert for one already-fetched PDF. Commits on
    success; marks the document failed (committed) on any error."""
    try:
        validate_pdf_bytes(data)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            extracted = extract_pdf(tmp.name)

        chunks = chunk_blocks(extracted.blocks)
        if not chunks:
            raise IngestError("no extractable text (even after OCR)")

        embedder = get_embedder()
        embeddings: list[list[float]] = []
        bs = settings.embed_batch_size
        for i in range(0, len(chunks), bs):
            batch = [c.content for c in chunks[i : i + bs]]
            embeddings.extend(embedder.embed(batch))

        db.replace_chunks(conn, document_id, access_tier, chunks, embeddings)
        db.mark_ready(
            conn, document_id, extracted.page_count, len(chunks), extracted.language
        )
        conn.commit()
        return IngestOutcome(
            status="ready",
            page_count=extracted.page_count,
            chunk_count=len(chunks),
            language=extracted.language,
        )
    except Exception as exc:  # noqa: BLE001 — record every failure honestly
        conn.rollback()
        db.mark_failed(conn, document_id, str(exc))
        conn.commit()
        return IngestOutcome(status="failed", error=str(exc))
