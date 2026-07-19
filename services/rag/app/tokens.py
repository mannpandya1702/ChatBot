"""Token counting for chunk sizing.

Production uses the BGE-M3 (XLM-RoBERTa) tokenizer for exact counts. When
transformers isn't installed we fall back to a script-aware heuristic that
counts Latin word-pieces and Devanagari aksharas separately — it slightly
*over*-counts Devanagari, which errs toward smaller chunks (safe against the
900-token hard max) rather than oversized ones.
"""
from __future__ import annotations

import re
from functools import lru_cache

_WORD = re.compile(r"[A-Za-z0-9]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_PUNCT = re.compile(r"[^\sA-Za-z0-9ऀ-ॿ]")


@lru_cache(maxsize=1)
def _hf_tokenizer():
    """Return the real XLM-R tokenizer, or None if transformers is unavailable."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        from .config import settings

        return AutoTokenizer.from_pretrained(settings.embedding_model)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    from .config import settings

    if settings.token_backend == "heuristic":
        return _heuristic_tokens(text)
    tok = _hf_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    if settings.token_backend == "hf":
        raise RuntimeError(
            "RAG_TOKEN_BACKEND=hf but the model tokenizer is unavailable "
            "(install transformers)."
        )
    return _heuristic_tokens(text)  # 'auto' fallback


def _heuristic_tokens(text: str) -> int:
    if not text:
        return 0
    # Latin words average ~1.3 sub-word tokens each; Devanagari aksharas and
    # standalone punctuation are ~1 token apiece.
    words = len(_WORD.findall(text))
    deva = len(_DEVANAGARI.findall(text))
    punct = len(_PUNCT.findall(text))
    return max(1, round(words * 1.3) + deva + punct)
