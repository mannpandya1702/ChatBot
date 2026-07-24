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


def lock_document(conn: "psycopg.Connection", document_id: str) -> None:
    """Serialise ingestion per document with a transaction-scoped advisory lock.

    Two concurrent ingests of the SAME document (e.g. a double-submitted re-index)
    would otherwise race the delete+insert in replace_chunks — colliding on the
    (document_id, chunk_index) unique key or leaving a half-written chunk set — and
    duplicate the expensive extract+embed work. The lock is keyed on a stable hash
    of the document id and auto-releases at commit/rollback, so it blocks only the
    matching document and never a different one.
    """
    conn.execute(
        "select pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (document_id,),
    )


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


def sweep_stale_documents(conn, older_than_minutes: int) -> list[str]:
    """Mark documents orphaned in 'processing' past the threshold as 'failed'.

    A document sits in 'processing' only between row creation and the end of its
    (synchronous) ingest; if the ingesting process is killed mid-run the row is
    stranded there forever and the admin sees a spinner that never resolves.
    created_at is a reliable proxy for processing-start because no code path
    re-enters 'processing' on an already-created row (re-index keeps the prior
    status until it finishes). Returns the swept ids so the caller can report
    them. Caller commits.
    """
    rows = conn.execute(
        "update public.documents "
        "set status='failed', "
        "    error='ingestion did not finish; marked failed by stale sweep' "
        "where status='processing' "
        "  and created_at < now() - make_interval(mins => %s) "
        "returning id",
        (older_than_minutes,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def purge_old_data(
    conn,
    conversations_days: int | None = None,
    audit_days: int | None = None,
    analytics_days: int | None = None,
    login_attempts_days: int | None = None,
) -> dict[str, int]:
    """Delete data past its retention window (spec §4 log-minimisation / M3).

    Each window is independent and skipped when None, so an operator opts in per
    table. Messages cascade with their parent conversation. login_attempts only
    feed the short lockout window, so they can be purged aggressively. Returns
    per-table deleted-row counts. Caller commits.
    """
    plan = (
        ("conversations", "updated_at", conversations_days),
        ("audit_logs", "created_at", audit_days),
        ("query_analytics", "created_at", analytics_days),
        ("login_attempts", "created_at", login_attempts_days),
    )
    counts: dict[str, int] = {}
    for table, ts_col, days in plan:
        if days is None:
            continue
        # table/column are fixed literals from `plan`, never caller input.
        cur = conn.execute(
            f"delete from public.{table} "
            f"where {ts_col} < now() - make_interval(days => %s)",
            (days,),
        )
        counts[table] = cur.rowcount
    return counts
