"""Rerank backends.

`bge`     — real BAAI/bge-reranker-v2-m3 cross-encoder, sigmoid-normalised.
`hashing` — deterministic token-overlap relevance passed through a sigmoid.
            Meaningful (shared query terms → higher score) so refusal-threshold
            behaviour can be exercised without the model download. Selected via
            RAG_RERANK_BACKEND=hashing.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from .config import settings
from .schemas import RerankResult

_TOK = re.compile(r"[A-Za-z0-9]+|[ऀ-ॿ]+")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    name: str

    def rerank(self, query: str, passages, top_k: int) -> list[RerankResult]:  # pragma: no cover
        raise NotImplementedError


class HashingReranker(Reranker):
    name = "hashing"

    def _tokens(self, text: str) -> set[str]:
        return {t.lower() for t in _TOK.findall(text)}

    def rerank(self, query: str, passages, top_k: int) -> list[RerankResult]:
        q = self._tokens(query)
        scored: list[RerankResult] = []
        for p in passages:
            pt = self._tokens(p.text)
            overlap = len(q & pt)
            denom = len(q) or 1
            # Map overlap fraction to a logit in roughly [-4, 4] before sigmoid.
            logit = (overlap / denom) * 8.0 - 4.0
            scored.append(RerankResult(id=p.id, score=_sigmoid(logit), text=p.text))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


class BgeReranker(Reranker):
    """bge-reranker-v2-m3 as a standard transformers cross-encoder.

    Run directly via AutoModelForSequenceClassification rather than
    FlagEmbedding's compute_score(), whose slow-tokenizer path breaks on current
    transformers (XLMRobertaTokenizer has no `prepare_for_model`). The model
    emits one relevance logit per (query, passage) pair; sigmoid maps it to the
    (0, 1) score the refusal threshold compares against.
    """

    name = "bge"
    _batch = 16
    _max_length = settings.rerank_max_length

    def __init__(self) -> None:
        import torch  # noqa: F401 — ensure torch is present before model load
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(settings.rerank_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            settings.rerank_model
        )
        self._model.eval()

    def rerank(self, query: str, passages, top_k: int) -> list[RerankResult]:
        pairs = [[query, p.text] for p in passages]
        scores: list[float] = []
        with self._torch.no_grad():
            for i in range(0, len(pairs), self._batch):
                batch = pairs[i : i + self._batch]
                inputs = self._tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                )
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(self._torch.sigmoid(logits).tolist())
        scored = [
            RerankResult(id=p.id, score=float(s), text=p.text)
            for p, s in zip(passages, scores)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    if settings.rerank_backend == "hashing":
        return HashingReranker()
    try:
        return BgeReranker()
    except Exception as exc:
        raise RuntimeError(
            "BGE reranker unavailable and RAG_RERANK_BACKEND != 'hashing'. "
            f"Install FlagEmbedding or set the hashing backend. Cause: {exc}"
        ) from exc
