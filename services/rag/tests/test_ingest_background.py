"""Background ingestion: the path a serverless caller must use.

A Vercel function is killed at 60s, while extract+OCR+embed on a large scan runs
for minutes — so `/ingest` has to be able to accept the job and answer at once,
with documents.status as the only progress channel. These cover the contract
that makes that safe: queue only after the row is confirmed, and never leave a
document stuck in 'processing' when the work explodes.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import db, main
from app.ingest import sha256_of

AUTH = {"X-Service-Secret": "test-secret"}


@pytest.fixture()
def conn(e2e_dsn, monkeypatch):
    if e2e_dsn is None:
        pytest.skip("local Postgres unavailable")
    monkeypatch.setattr(db.settings, "database_url", e2e_dsn)
    c = psycopg.connect(e2e_dsn, autocommit=False)
    yield c
    c.close()


@pytest.fixture()
def insert_document(conn):
    """Create documents rows and remove them afterwards. The e2e database is
    session-scoped and documents.sha256 is unique, so a row left behind here
    would collide with any later test that ingests the same fixture PDF."""
    created: list[str] = []

    def _insert(sha: str) -> str:
        document_id = str(uuid.uuid4())
        conn.execute(
            "insert into public.documents "
            "(id, title, original_filename, storage_path, sha256, access_tier, status) "
            "values (%s,'Leave Rules','leave.pdf',%s,%s,1,'processing')",
            (document_id, f"{document_id}.pdf", sha),
        )
        conn.commit()
        created.append(document_id)
        return document_id

    yield _insert

    for document_id in created:  # chunks cascade
        conn.execute("delete from public.documents where id=%s", (document_id,))
    conn.commit()


def _status(conn, document_id: str) -> tuple[str, str | None]:
    return conn.execute(
        "select status, error from public.documents where id=%s", (document_id,)
    ).fetchone()


def test_background_ingest_returns_202_and_completes(conn, insert_document, text_pdf, monkeypatch):
    data = open(text_pdf, "rb").read()
    document_id = insert_document(sha256_of(data))
    monkeypatch.setattr(main, "fetch_supabase_storage", lambda _path: data)

    with TestClient(main.app) as client:
        r = client.post("/ingest", json={"document_id": document_id, "background": True}, headers=AUTH)
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "processing"
        assert body["document_id"] == document_id
    main._INGEST_QUEUE.join()

    status, error = _status(conn, document_id)
    assert status == "ready", error


def test_background_ingest_records_failure_rather_than_hanging(conn, insert_document, monkeypatch):
    """A Storage fetch that blows up has no caller left to turn it into a status.
    If this regresses, the document sits in 'processing' until an operator runs
    the stale sweeper — and the admin UI spins forever."""
    document_id = insert_document("0" * 64)

    def _boom(_path):
        raise RuntimeError("storage unreachable")

    monkeypatch.setattr(main, "fetch_supabase_storage", _boom)
    with TestClient(main.app) as client:
        assert client.post(
            "/ingest", json={"document_id": document_id, "background": True}, headers=AUTH
        ).status_code == 202
    main._INGEST_QUEUE.join()

    status, error = _status(conn, document_id)
    assert status == "failed"
    assert "storage unreachable" in error


def test_background_ingest_rejects_unknown_document_synchronously(conn):
    """A bad id must 404 on the spot — queueing it would report success for work
    that can never happen."""
    with TestClient(main.app) as client:
        r = client.post(
            "/ingest", json={"document_id": str(uuid.uuid4()), "background": True}, headers=AUTH
        )
    assert r.status_code == 404


def test_checksum_mismatch_fails_the_document(conn, insert_document, text_pdf, monkeypatch):
    """The sha256 on the row is computed in the browser — the web tier never sees
    the bytes. This is the only place it gets checked against reality, so a file
    swapped between signing and upload must fail rather than index."""
    data = open(text_pdf, "rb").read()
    document_id = insert_document(sha256_of(b"a completely different file"))
    monkeypatch.setattr(main, "fetch_supabase_storage", lambda _path: data)

    with TestClient(main.app) as client:
        r = client.post("/ingest", json={"document_id": document_id}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "sha256" in r.json()["error"]

    status, _ = _status(conn, document_id)
    assert status == "failed"
    assert conn.execute(
        "select count(*) from public.chunks where document_id=%s", (document_id,)
    ).fetchone()[0] == 0, "a mismatched file must not leave chunks behind"
