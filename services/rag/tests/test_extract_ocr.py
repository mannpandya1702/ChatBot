"""Extraction + OCR-fallback tests (spec §6)."""
from __future__ import annotations

import pytest

from app.extract import extract_pdf
from app.ingest import IngestError, validate_pdf_bytes


def test_text_pdf_no_ocr(text_pdf):
    res = extract_pdf(text_pdf)
    assert res.page_count == 2
    assert res.ocr_pages == [], "digital text must not trigger OCR"
    assert res.language == "en"
    kinds = {b.kind for b in res.blocks}
    assert "para" in kinds
    assert "heading" in kinds, "large title text should be detected as a heading"
    # The known heading text is present.
    assert any("LEAVE RULES" in b.text for b in res.blocks if b.kind == "heading")


def test_scanned_pdf_triggers_ocr(scanned_pdf):
    res = extract_pdf(scanned_pdf)
    assert res.page_count == 1
    assert res.ocr_pages == [1], "image-only page must go through OCR"
    text = " ".join(b.text.upper() for b in res.blocks if b.source == "ocr")
    # Tesseract should recover the rendered words (allow minor OCR noise).
    hits = sum(w in text for w in ("PENSION", "FAMILY", "PENSION", "KIN"))
    assert hits >= 2, f"OCR text did not contain expected words: {text!r}"


def test_hindi_traineddata_available():
    import pytesseract

    langs = pytesseract.get_languages(config="")
    assert "hin" in langs and "eng" in langs, f"missing OCR langs: {langs}"


def test_validate_rejects_non_pdf():
    with pytest.raises(IngestError, match="magic bytes"):
        validate_pdf_bytes(b"this is not a pdf")


def test_validate_rejects_oversized(monkeypatch):
    from app import ingest

    monkeypatch.setattr(ingest.settings, "max_pdf_bytes", 10)
    with pytest.raises(IngestError, match="cap"):
        validate_pdf_bytes(b"%PDF-" + b"x" * 100)
