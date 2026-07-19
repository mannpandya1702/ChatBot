"""Chunker unit tests (spec §6 sizing/overlap/table-atomicity rules)."""
from __future__ import annotations

from app.chunker import chunk_blocks
from app.config import settings
from app.extract import Block
from app.tokens import count_tokens


def _para(text: str, page: int = 1) -> Block:
    return Block(kind="para", text=text, page=page)


def test_respects_hard_max_and_splits():
    # One very long paragraph must be split into multiple sub-max chunks.
    sentence = "The sanctioning authority reviews each leave application carefully. "
    big = sentence * 120  # well over 900 tokens
    chunks = chunk_blocks([_para(big)])
    assert len(chunks) >= 2
    assert all(c.token_count <= settings.chunk_max_tokens for c in chunks)


def test_table_is_atomic_and_never_split():
    table_md = "| Rank | Days |\n| --- | --- |\n" + "".join(
        f"| Row{i} | {i} |\n" for i in range(200)
    )
    blocks = [
        _para("Intro paragraph before the table."),
        Block(kind="table", text=table_md, page=1),
        _para("Paragraph after the table."),
    ]
    chunks = chunk_blocks(blocks)
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1, "table must land in exactly one chunk"
    # The table survives intact — every row present and contiguous (trailing
    # whitespace normalisation aside).
    assert table_chunks[0].content == table_md.strip()
    assert table_chunks[0].content.count("| Row") == 200, "every row must survive"
    # A table chunk is atomic: it contains only the table, no prose bled in.
    assert "Intro paragraph" not in table_chunks[0].content
    assert "Paragraph after" not in table_chunks[0].content


def test_overlap_between_consecutive_text_chunks():
    # Distinct filler per paragraph so a shared sentence can ONLY come from the
    # overlap tail, not from coincidentally identical text.
    para_a = "Alpha topic opens here. " + "Alpha detail sentence stands alone. " * 40
    para_b = "Beta topic opens here. " + "Beta detail sentence stands alone. " * 40
    chunks = chunk_blocks([_para(para_a), _para(para_b)])
    assert len(chunks) >= 2
    # A later (beta) chunk must carry an alpha sentence forward as overlap.
    assert any(
        "Alpha detail sentence stands alone" in c.content for c in chunks[1:]
    ), "expected overlap text carried into a later chunk"
    # And the hard-max invariant holds even after overlap is prepended.
    assert all(c.token_count <= settings.chunk_max_tokens for c in chunks)


def test_section_path_tracks_heading_breadcrumb():
    blocks = [
        Block(kind="heading", text="Leave", page=1, level=1),
        _para("General leave provisions."),
        Block(kind="heading", text="Casual Leave", page=1, level=2),
        _para("Casual leave is limited to thirty days."),
        Block(kind="heading", text="Annual Leave", page=2, level=2),
        _para("Annual leave accrues monthly."),
    ]
    chunks = chunk_blocks(blocks)
    by_text = {c.content.split(".")[0]: c.section_path for c in chunks}
    assert by_text["General leave provisions"] == "Leave"
    assert by_text["Casual leave is limited to thirty days"] == "Leave > Casual Leave"
    # Same-level heading replaces the previous one, not nests under it.
    assert by_text["Annual leave accrues monthly"] == "Leave > Annual Leave"


def test_page_range_tracked():
    blocks = [_para("First page content here.", page=3), _para("Still page three.", page=3)]
    chunks = chunk_blocks(blocks)
    assert chunks[0].page_start == 3 and chunks[0].page_end == 3


def test_token_count_is_populated_and_sane():
    blocks = [_para("A short paragraph about leave.")]
    chunks = chunk_blocks(blocks)
    assert chunks[0].token_count == count_tokens(chunks[0].content)
    assert chunks[0].token_count > 0
