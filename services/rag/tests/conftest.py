"""Test fixtures: deterministic backends, synthetic PDFs, and a fresh e2e DB.

Env is set here BEFORE any `app.*` import so the settings singleton picks up the
test doubles ('hashing' backends) and test secret.
"""
from __future__ import annotations

import os

os.environ.setdefault("RAG_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("RAG_RERANK_BACKEND", "hashing")
os.environ.setdefault("RAG_SERVICE_SECRET", "test-secret")
# Deterministic token counts regardless of whether transformers is installed,
# so chunk-boundary assertions are stable across environments.
os.environ.setdefault("RAG_TOKEN_BACKEND", "heuristic")

import subprocess  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import fitz  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MIG = REPO_ROOT / "supabase" / "migrations"
SHIM = REPO_ROOT / "scripts" / "local-db" / "shim-auth.sql"
PG_HOST = "/var/run/postgresql"


def _font_path() -> str | None:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            return p
    return None


@pytest.fixture(scope="session")
def text_pdf(tmp_path_factory) -> str:
    """A born-digital PDF: a large heading, body paragraphs, a subheading, and a
    second page. No image — should never trigger OCR."""
    path = tmp_path_factory.mktemp("pdfs") / "leave_rules.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 90), "LEAVE RULES", fontsize=24, fontname="helv")
    p1.insert_text(
        (72, 140),
        "Annual leave may be applied through the unit adjutant. Applications "
        "should be submitted at least seven days in advance. Sanctioning "
        "authority is the officer commanding.",
        fontsize=11,
        fontname="helv",
    )
    p1.insert_text((72, 230), "Casual Leave", fontsize=16, fontname="helv")
    p1.insert_text(
        (72, 270),
        "Casual leave is limited to thirty days in a calendar year and cannot "
        "be combined with annual leave. Requests are recorded in the unit "
        "leave register.",
        fontsize=11,
        fontname="helv",
    )
    p2 = doc.new_page()
    p2.insert_text((72, 90), "Sick Leave", fontsize=16, fontname="helv")
    p2.insert_text(
        (72, 130),
        "Sick leave is granted on the recommendation of the medical officer. "
        "Extended sick leave beyond ninety days requires a medical board.",
        fontsize=11,
        fontname="helv",
    )
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory) -> str:
    """An image-only PDF page (rendered text, no text layer) to force the OCR
    fallback. Uppercase English for reliable Tesseract recognition."""
    from PIL import Image, ImageDraw, ImageFont

    tmp = tmp_path_factory.mktemp("scan")
    img = Image.new("RGB", (1240, 500), "white")
    draw = ImageDraw.Draw(img)
    fp = _font_path()
    font = ImageFont.truetype(fp, 48) if fp else ImageFont.load_default()
    draw.text((60, 80), "SCANNED PENSION CIRCULAR", fill="black", font=font)
    draw.text((60, 200), "FAMILY PENSION IS PAYABLE", fill="black", font=font)
    draw.text((60, 320), "TO THE NEXT OF KIN", fill="black", font=font)
    img_path = tmp / "page.png"
    img.save(img_path)

    path = tmp / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page(width=1240, height=500)
    page.insert_image(fitz.Rect(0, 0, 1240, 500), filename=str(img_path))
    doc.save(path)
    doc.close()
    return str(path)


def _psql(dbname: str, sql_file: Path) -> None:
    subprocess.run(
        ["psql", "-v", "ON_ERROR_STOP=1", "-q", "-h", PG_HOST, "-U", "root",
         "-d", dbname, "-f", str(sql_file)],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture(scope="session")
def e2e_dsn() -> str | None:
    """Create a throwaway DB with the real migrations applied (extensions +
    schema). Returns a DSN, or None if local Postgres is unavailable (test skips)."""
    dbname = "sainik_rag_e2e"
    base = ["-h", PG_HOST, "-U", "root"]
    try:
        subprocess.run(["dropdb", *base, "--if-exists", dbname], check=True, capture_output=True, text=True)
        subprocess.run(["createdb", *base, dbname], check=True, capture_output=True, text=True)
        _psql(dbname, SHIM)
        _psql(dbname, MIG / "20260719000100_extensions.sql")
        _psql(dbname, MIG / "20260719000200_schema.sql")
    except Exception:
        return None
    return f"dbname={dbname} host={PG_HOST} user=root"
