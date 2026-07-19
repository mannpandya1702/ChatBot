"""PDF text extraction with OCR fallback (spec §6).

Per page: extract the text layer with PyMuPDF. If a page yields fewer than
`ocr_min_chars` characters (scanned/image-only page), rasterise it at
`ocr_dpi` and OCR with Tesseract using `hin+eng` traineddata. Detected tables
are emitted as atomic blocks so the chunker never splits them; headings are
detected by relative font size to build a section breadcrumb.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from .config import settings

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class Block:
    kind: str          # 'heading' | 'para' | 'table'
    text: str
    page: int          # 1-indexed
    level: int = 0     # heading depth (1 = top); 0 for para/table
    source: str = "text"  # 'text' | 'ocr'


@dataclass
class ExtractResult:
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    language: str = "en"


def _guess_language(text: str) -> str:
    if not text.strip():
        return "en"
    deva = len(_DEVANAGARI.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = deva + latin
    if total == 0:
        return "en"
    ratio = deva / total
    if ratio > 0.6:
        return "hi"
    if ratio > 0.15:
        return "mixed"
    return "en"


def _ocr_page(page: "fitz.Page") -> str:
    """Rasterise a page and OCR it. Imports are local so the module loads even
    where Pillow/pytesseract aren't installed (e.g. pure-chunker unit tests)."""
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=settings.ocr_dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=settings.ocr_languages)


def _table_bboxes(page: "fitz.Page"):
    """Return (bbox, markdown_text) for each detected table. Tolerant of
    PyMuPDF builds without find_tables()."""
    out = []
    try:
        finder = page.find_tables()
    except Exception:
        return out
    for tbl in getattr(finder, "tables", []):
        try:
            rows = tbl.extract()
        except Exception:
            continue
        if not rows:
            continue
        md = _rows_to_markdown(rows)
        if md.strip():
            out.append((fitz.Rect(tbl.bbox), md))
    return out


def _rows_to_markdown(rows: list[list]) -> str:
    def cell(c) -> str:
        return (c or "").replace("\n", " ").replace("|", "\\|").strip()

    lines = []
    for r_i, row in enumerate(rows):
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
        if r_i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")
    return "\n".join(lines)


def _spans_of(block: dict):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span


def extract_pdf(path: str) -> ExtractResult:
    doc = fitz.open(path)
    result = ExtractResult(page_count=doc.page_count)
    all_text_parts: list[str] = []

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        page_no = page_index + 1
        plain = page.get_text("text")

        if len(plain.strip()) < settings.ocr_min_chars:
            ocr_text = _ocr_page(page)
            result.ocr_pages.append(page_no)
            for para in _split_paragraphs(ocr_text):
                result.blocks.append(
                    Block(kind="para", text=para, page=page_no, source="ocr")
                )
            all_text_parts.append(ocr_text)
            continue

        all_text_parts.append(plain)
        tables = _table_bboxes(page)
        data = page.get_text("dict")
        text_blocks = [b for b in data.get("blocks", []) if b.get("type") == 0]

        # Median span size = body text size; larger spans are headings.
        sizes = [round(s["size"], 1) for b in text_blocks for s in _spans_of(b)]
        body_size = statistics.median(sizes) if sizes else 0.0

        # Merge tables + text blocks into one reading-order stream by y-position.
        items: list[tuple[float, float, str, object]] = []
        for bbox, md in tables:
            items.append((bbox.y0, bbox.x0, "table", md))
        for b in text_blocks:
            bx = fitz.Rect(b["bbox"])
            if any(bx.intersects(tb) for tb, _ in tables):
                continue  # text already captured by a table block
            items.append((bx.y0, bx.x0, "text", b))
        items.sort(key=lambda it: (round(it[0], 1), it[1]))

        for _, _, kind, payload in items:
            if kind == "table":
                result.blocks.append(
                    Block(kind="table", text=str(payload), page=page_no)
                )
                continue
            block = payload  # type: ignore[assignment]
            text = _block_text(block)
            if not text.strip():
                continue
            level = _heading_level(block, body_size)
            result.blocks.append(
                Block(
                    kind="heading" if level else "para",
                    text=text,
                    page=page_no,
                    level=level,
                )
            )

    doc.close()
    result.language = _guess_language("\n".join(all_text_parts))
    return result


def _block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        lines.append("".join(s["text"] for s in line.get("spans", [])))
    return "\n".join(lines).strip()


def _heading_level(block: dict, body_size: float) -> int:
    """0 = not a heading; 1/2/3 = heading depth by font size vs body."""
    spans = list(_spans_of(block))
    if not spans or body_size <= 0:
        return 0
    max_size = max(s["size"] for s in spans)
    text = _block_text(block)
    if len(text) > 160 or "\n" in text.strip():
        return 0  # long/multi-line → paragraph, not a heading
    ratio = max_size / body_size
    is_bold = any(bool(s.get("flags", 0) & 16) for s in spans)
    if ratio >= 1.5:
        return 1
    if ratio >= 1.25:
        return 2
    if ratio >= 1.1 or (is_bold and ratio >= 1.02):
        return 3
    return 0


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [re.sub(r"[ \t]+", " ", p.strip()) for p in parts if p.strip()]
