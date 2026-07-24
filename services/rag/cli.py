#!/usr/bin/env python3
"""Bulk ingestion CLI (spec §6).

    python services/rag/cli.py ingest ./pdfs --tier 1

Reads local PDFs, inserts a documents row (dedup by sha256), then runs the same
extract/chunk/embed/upsert pipeline the /ingest endpoint uses. Intended for the
operator's initial KB load; connects to RAG_DATABASE_URL with a service-role
credential.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from app import db
from app.ingest import IngestError, ingest_pdf_bytes, sha256_of, validate_pdf_bytes
from app.sources import fetch_local


def _existing_by_sha(conn, sha: str):
    return conn.execute(
        "select id, status from public.documents where sha256 = %s", (sha,)
    ).fetchone()


def ingest_dir(directory: str, tier: int, reindex: bool) -> int:
    pdfs = sorted(Path(directory).glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs found in {directory}", file=sys.stderr)
        return 1

    failures = 0
    with db.connect() as conn:
        for path in pdfs:
            data = path.read_bytes()
            try:
                validate_pdf_bytes(data)
            except IngestError as exc:
                print(f"SKIP  {path.name}: {exc}")
                failures += 1
                continue

            sha = sha256_of(data)
            existing = _existing_by_sha(conn, sha)
            if existing and not reindex:
                print(f"DUP   {path.name}: already ingested ({existing[0]}) — skipping")
                continue

            if existing:
                document_id = str(existing[0])
            else:
                document_id = str(uuid.uuid4())
                conn.execute(
                    "insert into public.documents "
                    "(id, title, original_filename, storage_path, sha256, "
                    " access_tier, status) values (%s,%s,%s,%s,%s,%s,'processing')",
                    (document_id, path.stem, path.name, f"kb/{path.name}", sha, tier),
                )
                conn.commit()

            outcome = ingest_pdf_bytes(conn, document_id, fetch_local(str(path)), tier)
            if outcome.status == "ready":
                print(
                    f"OK    {path.name}: {outcome.page_count} pages, "
                    f"{outcome.chunk_count} chunks, lang={outcome.language}"
                )
            else:
                print(f"FAIL  {path.name}: {outcome.error}")
                failures += 1

    return 1 if failures else 0


def sweep_stale(older_than_minutes: int) -> int:
    """Mark documents stranded in 'processing' as failed (operator maintenance)."""
    with db.connect() as conn:
        swept = db.sweep_stale_documents(conn, older_than_minutes)
        conn.commit()
    if swept:
        print(f"swept {len(swept)} stale document(s) to 'failed':")
        for did in swept:
            print(f"  {did}")
    else:
        print("no stale documents")
    return 0


def purge(args) -> int:
    """Delete data past per-table retention windows (operator maintenance)."""
    with db.connect() as conn:
        counts = db.purge_old_data(
            conn,
            conversations_days=args.conversations_days,
            audit_days=args.audit_days,
            analytics_days=args.analytics_days,
            login_attempts_days=args.login_attempts_days,
        )
        conn.commit()
    if counts:
        for table, n in counts.items():
            print(f"purged {n} row(s) from {table}")
    else:
        print("nothing purged (pass at least one --*-days window)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="cli.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ingest a directory of PDFs")
    ing.add_argument("directory")
    ing.add_argument("--tier", type=int, default=1, choices=(1, 2, 3))
    ing.add_argument("--reindex", action="store_true", help="re-ingest even if sha256 exists")

    sweep = sub.add_parser(
        "sweep-stale", help="mark documents stuck in 'processing' past a threshold as failed"
    )
    sweep.add_argument("--older-than-minutes", type=int, default=60)

    pur = sub.add_parser("purge", help="delete data past retention windows (per table, opt-in)")
    pur.add_argument("--conversations-days", type=int, default=None,
                     help="delete conversations (and their messages) idle this many days")
    pur.add_argument("--audit-days", type=int, default=None,
                     help="delete audit_logs older than this many days")
    pur.add_argument("--analytics-days", type=int, default=None,
                     help="delete query_analytics older than this many days")
    pur.add_argument("--login-attempts-days", type=int, default=None,
                     help="delete login_attempts older than this many days")

    args = parser.parse_args()
    if args.cmd == "ingest":
        return ingest_dir(args.directory, args.tier, args.reindex)
    if args.cmd == "sweep-stale":
        return sweep_stale(args.older_than_minutes)
    if args.cmd == "purge":
        return purge(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
