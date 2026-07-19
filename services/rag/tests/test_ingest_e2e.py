"""End-to-end ingestion into a real Postgres+pgvector DB (previews Phase 1 DoD).

Uses the synthetic text PDF and the deterministic hashing embedder, so it runs
without the multi-GB BGE download. The identical path with RAG_EMBEDDING_BACKEND=
bge and a real PDF is the final Phase 1 DoD run.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from app import db
from app.ingest import ingest_pdf_bytes, sha256_of


@pytest.fixture()
def conn(e2e_dsn, monkeypatch):
    if e2e_dsn is None:
        pytest.skip("local Postgres unavailable")
    monkeypatch.setattr(db.settings, "database_url", e2e_dsn)
    c = psycopg.connect(e2e_dsn, autocommit=False)
    yield c
    c.close()


def test_ingest_writes_chunks_and_embeddings(conn, text_pdf):
    data = open(text_pdf, "rb").read()
    document_id = str(uuid.uuid4())
    conn.execute(
        "insert into public.documents "
        "(id, title, original_filename, storage_path, sha256, access_tier, status) "
        "values (%s,'Leave Rules','leave.pdf','kb/leave.pdf',%s,1,'processing')",
        (document_id, sha256_of(data)),
    )
    conn.commit()

    outcome = ingest_pdf_bytes(conn, document_id, data, access_tier=1)
    assert outcome.status == "ready", outcome.error
    assert outcome.page_count == 2
    assert outcome.chunk_count and outcome.chunk_count > 0

    # documents row updated
    status, page_count, chunk_count = conn.execute(
        "select status, page_count, chunk_count from public.documents where id=%s",
        (document_id,),
    ).fetchone()
    assert status == "ready"
    assert page_count == 2
    assert chunk_count == outcome.chunk_count

    # chunks exist with 1024-dim embeddings and tier denormalised
    rows = conn.execute(
        "select count(*), min(vector_dims(embedding)), max(vector_dims(embedding)), "
        "bool_and(access_tier = 1), bool_and(embedding is not null) "
        "from public.chunks where document_id=%s",
        (document_id,),
    ).fetchone()
    count, min_dim, max_dim, all_tier1, all_embedded = rows
    assert count == outcome.chunk_count
    assert min_dim == 1024 and max_dim == 1024, "embeddings must be 1024-dim"
    assert all_tier1, "chunk access_tier must be denormalised from the document"
    assert all_embedded, "every chunk must carry an embedding"


def test_reindex_replaces_chunks(conn, text_pdf):
    data = open(text_pdf, "rb").read()
    document_id = str(uuid.uuid4())
    conn.execute(
        "insert into public.documents "
        "(id, title, original_filename, storage_path, sha256, access_tier, status) "
        "values (%s,'Leave','leave2.pdf','kb/leave2.pdf',%s,1,'processing')",
        (document_id, sha256_of(data) + "-2"),
    )
    conn.commit()
    first = ingest_pdf_bytes(conn, document_id, data, 1)
    again = ingest_pdf_bytes(conn, document_id, data, 1)  # reindex
    assert again.status == "ready"
    # No duplicate accumulation — chunk count stays stable across reindex.
    total = conn.execute(
        "select count(*) from public.chunks where document_id=%s", (document_id,)
    ).fetchone()[0]
    assert total == again.chunk_count == first.chunk_count
