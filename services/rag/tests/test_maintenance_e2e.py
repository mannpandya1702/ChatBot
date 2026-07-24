"""Operator maintenance jobs against a real Postgres: the stale-'processing'
sweeper and the retention purge (services/rag/app/db.py). Skips when local
Postgres is unavailable (same fixture as the ingest e2e tests)."""
from __future__ import annotations

import uuid

import psycopg
import pytest

from app import db


@pytest.fixture()
def conn(e2e_dsn, monkeypatch):
    if e2e_dsn is None:
        pytest.skip("local Postgres unavailable")
    monkeypatch.setattr(db.settings, "database_url", e2e_dsn)
    c = psycopg.connect(e2e_dsn, autocommit=False)
    yield c
    c.close()


def _new_doc(conn, *, status: str, age_minutes: int) -> str:
    document_id = str(uuid.uuid4())
    conn.execute(
        "insert into public.documents "
        "(id, title, original_filename, storage_path, sha256, access_tier, status, created_at) "
        "values (%s,'t','f.pdf',%s,%s,1,%s, now() - make_interval(mins => %s))",
        (document_id, f"kb/{document_id}.pdf", document_id, status, age_minutes),
    )
    return document_id


def test_sweep_marks_only_old_processing_docs(conn):
    stale = _new_doc(conn, status="processing", age_minutes=120)
    fresh = _new_doc(conn, status="processing", age_minutes=1)
    ready = _new_doc(conn, status="ready", age_minutes=120)
    conn.commit()

    swept = db.sweep_stale_documents(conn, older_than_minutes=60)
    conn.commit()

    assert swept == [stale]

    def status_of(doc_id):
        return conn.execute(
            "select status, error from public.documents where id=%s", (doc_id,)
        ).fetchone()

    stale_status, stale_error = status_of(stale)
    assert stale_status == "failed"
    assert "stale sweep" in (stale_error or "")
    assert status_of(fresh)[0] == "processing"  # too recent to sweep
    assert status_of(ready)[0] == "ready"        # never in-flight


def test_sweep_no_stale_returns_empty(conn):
    _new_doc(conn, status="processing", age_minutes=1)
    conn.commit()
    assert db.sweep_stale_documents(conn, older_than_minutes=60) == []


def test_purge_deletes_only_rows_past_window(conn):
    # query_analytics + login_attempts carry no FK, so they isolate the purge logic.
    for age_days, txt in ((100, "old"), (1, "new")):
        conn.execute(
            "insert into public.query_analytics (query_text, refused, created_at) "
            "values (%s, false, now() - make_interval(days => %s))",
            (txt, age_days),
        )
        conn.execute(
            "insert into public.login_attempts (service_number, success, created_at) "
            "values (%s, false, now() - make_interval(days => %s))",
            (txt, age_days),
        )
    conn.commit()

    counts = db.purge_old_data(conn, analytics_days=30, login_attempts_days=30)
    conn.commit()

    assert counts == {"query_analytics": 1, "login_attempts": 1}
    remaining_qa = conn.execute(
        "select query_text from public.query_analytics order by query_text"
    ).fetchall()
    assert [r[0] for r in remaining_qa] == ["new"]
    remaining_la = conn.execute(
        "select service_number from public.login_attempts"
    ).fetchall()
    assert [r[0] for r in remaining_la] == ["new"]


def test_purge_cascades_messages_with_conversation(conn):
    user_id = str(uuid.uuid4())
    conn.execute("insert into auth.users (id) values (%s)", (user_id,))
    old_conv, new_conv = str(uuid.uuid4()), str(uuid.uuid4())
    for cid, age_days in ((old_conv, 400), (new_conv, 1)):
        conn.execute(
            "insert into public.conversations (id, user_id, created_at, updated_at) "
            "values (%s,%s, now() - make_interval(days => %s), now() - make_interval(days => %s))",
            (cid, user_id, age_days, age_days),
        )
        conn.execute(
            "insert into public.messages (conversation_id, role, content) values (%s,'user','hi')",
            (cid,),
        )
    conn.commit()

    counts = db.purge_old_data(conn, conversations_days=365)
    conn.commit()

    assert counts == {"conversations": 1}
    # The old conversation and its message are both gone (FK cascade); new stays.
    assert conn.execute(
        "select count(*) from public.conversations where id=%s", (old_conv,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "select count(*) from public.messages where conversation_id=%s", (old_conv,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "select count(*) from public.conversations where id=%s", (new_conv,)
    ).fetchone()[0] == 1


def test_purge_skips_tables_with_no_window(conn):
    # No windows passed → no-op, empty result.
    assert db.purge_old_data(conn) == {}
