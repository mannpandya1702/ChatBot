"""Embedding backends.

`bge`     — the real BAAI/bge-m3 model (1024-dim), loaded lazily.
`hashing` — a deterministic, dependency-free test double producing L2-normalised
            1024-dim vectors. It is NOT a production embedder; it exists so the
            chunker, ingestion, and endpoint logic stay fully testable without a
            multi-GB model download. Selected via RAG_EMBEDDING_BACKEND=hashing.

Both return unit-norm vectors so cosine distance (pgvector `<=>`) is meaningful.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

from .config import settings


class Embedder:
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic bag-of-tokens hashing into `dim` buckets, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # empty/whitespace text — return a stable unit vector
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]


class BgeEmbedder(Embedder):
    """Real BGE-M3 dense embeddings via FlagEmbedding."""

    name = "bge"

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore

        self._model = BGEM3FlagModel(settings.embedding_model, use_fp16=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = self._model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            max_length=8192,
        )["dense_vecs"]
        return [v.tolist() for v in out]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    try:
        return BgeEmbedder()
    except Exception as exc:  # torch/model unavailable → fail loud, don't guess
        raise RuntimeError(
            "BGE embedder unavailable and RAG_EMBEDDING_BACKEND != 'hashing'. "
            f"Install FlagEmbedding or set the hashing backend. Cause: {exc}"
        ) from exc
