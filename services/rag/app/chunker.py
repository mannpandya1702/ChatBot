"""Structure-aware chunking (spec §6).

Rules:
  * Target ~600 tokens per chunk, hard max 900.
  * ~15% token overlap between consecutive text chunks in the same section,
    carried as trailing sentences of the previous chunk — trimmed so it never
    pushes a chunk past the hard max, and never applied across a table.
  * Never split a table: a table block is atomic (its own chunk).
  * Track section_path (heading breadcrumb), page_start/page_end, token_count.

Built in two passes so the hard-max guarantee is unconditional: pass A packs
blocks into base chunks by size only; pass B prepends overlap within the max.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import settings
from .extract import Block
from .tokens import count_tokens

_SENT = re.compile(r"[^.!?।\n]+[.!?।]+|\S[^.!?।\n]*$", re.UNICODE)


@dataclass
class Chunk:
    content: str
    chunk_index: int
    page_start: int
    page_end: int
    section_path: str
    token_count: int
    has_table: bool = False


def _sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENT.finditer(text) if m.group(0).strip()]


def _overlap_tail(text: str, max_tokens: int) -> str:
    """Trailing whole sentences of `text` totalling ≤ max_tokens."""
    if max_tokens <= 0:
        return ""
    tail: list[str] = []
    total = 0
    for s in reversed(_sentences(text)):
        t = count_tokens(s)
        if tail and total + t > max_tokens:
            break
        tail.insert(0, s)
        total += t
    return " ".join(tail)


def _units(text: str, hard_max: int) -> list[str]:
    """A paragraph small enough is one unit; an oversized one is split on
    sentence (then word) boundaries so no unit exceeds the target size."""
    if count_tokens(text) <= hard_max:
        return [text]
    target = settings.chunk_target_tokens
    units: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in _sentences(text):
        for piece in _cap_tokens(sent, target):
            pt = count_tokens(piece)
            if cur and cur_tok + pt > target:
                units.append(" ".join(cur))
                cur, cur_tok = [], 0
            cur.append(piece)
            cur_tok += pt
    if cur:
        units.append(" ".join(cur))
    return units or [text]


def _cap_tokens(sentence: str, cap: int) -> list[str]:
    """Last resort for a single sentence longer than `cap`: split on words."""
    if count_tokens(sentence) <= cap:
        return [sentence]
    words = sentence.split()
    out: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        if count_tokens(" ".join(cur)) >= cap:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


@dataclass
class _Acc:
    parts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    tokens: int = 0

    @property
    def empty(self) -> bool:
        return not self.parts


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    target = settings.chunk_target_tokens
    hard_max = settings.chunk_max_tokens
    overlap_tokens = round(target * settings.chunk_overlap_ratio)

    base: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    acc = _Acc()

    def section_path() -> str:
        return " > ".join(h for _, h in heading_stack)

    def emit(content: str, pages: list[int], has_table: bool) -> None:
        content = content.strip()
        if not content:
            return
        base.append(
            Chunk(
                content=content,
                chunk_index=len(base),
                page_start=min(pages),
                page_end=max(pages),
                section_path=section_path(),
                token_count=count_tokens(content),
                has_table=has_table,
            )
        )

    def flush() -> None:
        nonlocal acc
        if not acc.empty:
            emit("\n\n".join(acc.parts), acc.pages, False)
        acc = _Acc()

    # ── pass A: pack blocks into base chunks by size only ──────────────────
    for block in blocks:
        if block.kind == "heading":
            flush()  # a chunk never straddles a section boundary
            lvl = block.level or 3
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, block.text.strip()))
            continue

        if block.kind == "table":
            flush()
            emit(block.text, [block.page], True)  # atomic table chunk
            continue

        for unit in _units(block.text, hard_max):
            ut = count_tokens(unit)
            if not acc.empty and acc.tokens + ut > target:
                flush()
            acc.parts.append(unit)
            acc.pages.append(block.page)
            acc.tokens += ut
            if acc.tokens >= hard_max:
                flush()
    flush()

    # ── pass B: prepend overlap within the same section, respecting hard_max ─
    for i in range(1, len(base)):
        cur, prev = base[i], base[i - 1]
        if cur.has_table or prev.has_table:
            continue
        if cur.section_path != prev.section_path:
            continue
        budget = min(overlap_tokens, hard_max - cur.token_count)
        tail = _overlap_tail(prev.content, budget)
        if not tail:
            continue
        cur.content = f"{tail}\n\n{cur.content}"
        cur.token_count = count_tokens(cur.content)

    for idx, c in enumerate(base):
        c.chunk_index = idx
    return base
