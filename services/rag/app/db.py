"""Database access for ingestion.

In production the service connects to Supabase Postgres with the service-role
credential (bypasses RLS). Writes are limited to documents/chunks — the chat
pipeline and RLS live in the Next.js app; this service only fills the KB.
"""
from __future__ import annotations

from contextlib import contextmanager

import psycopg

from .config import settings


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


@contextmanager
def connect():
    conn = psycopg.connect(settings.database_url, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


def get_document(conn: "psycopg.Connection", document_id: str) -> dict | None:
    row = conn.execute(
        "select id, storage_path, sha256, access_tier, status "
        "from public.documents where id = %s",
        (document_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "storage_path": row[1],
        "sha256": row[2],
        "access_tier": row[3],
        "status": row[4],
    }


def replace_chunks(
    conn: "psycopg.Connection",
    document_id: str,
    access_tier: int,
    chunks,
    embeddings: list[list[float]],
) -> None:
    """Delete any existing chunks for the doc and insert the new set. Used by
    both first ingest and reindex; the caller commits."""
    conn.execute("delete from public.chunks where document_id = %s", (document_id,))
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "insert into public.chunks "
                "(document_id, chunk_index, content, embedding, page_start, "
                " page_end, section_path, token_count, access_tier) "
                "values (%s,%s,%s,%s::vector,%s,%s,%s,%s,%s)",
                (
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    _vec_literal(emb),
                    chunk.page_start,
                    chunk.page_end,
                    chunk.section_path or None,
                    chunk.token_count,
                    access_tier,
                ),
            )


def mark_ready(conn, document_id, page_count, chunk_count, language) -> None:
    conn.execute(
        "update public.documents set status='ready', page_count=%s, "
        "chunk_count=%s, language=%s, error=null where id=%s",
        (page_count, chunk_count, language, document_id),
    )


def mark_failed(conn, document_id, error: str) -> None:
    conn.execute(
        "update public.documents set status='failed', error=%s where id=%s",
        (error[:2000], document_id),
    )
