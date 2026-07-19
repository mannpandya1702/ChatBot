"""API surface tests: shared-secret auth + embed/rerank behaviour."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
AUTH = {"X-Service-Secret": "test-secret"}


def test_health_is_unauthenticated():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ocr_languages"] == ["hin", "eng"]


def test_embed_requires_secret():
    assert client.post("/embed", json={"texts": ["hi"]}).status_code == 401
    assert client.post("/embed", json={"texts": ["hi"]},
                       headers={"X-Service-Secret": "wrong"}).status_code == 401


def test_embed_returns_1024_dim_vectors():
    r = client.post("/embed", json={"texts": ["leave rules", "pension"]}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == 1024
    assert len(body["embeddings"]) == 2
    assert all(len(v) == 1024 for v in body["embeddings"])


def test_rerank_orders_by_relevance_with_sigmoid_scores():
    payload = {
        "query": "how do I apply for annual leave",
        "passages": [
            {"id": "a", "text": "Annual leave is applied through the unit adjutant."},
            {"id": "b", "text": "The mess menu changes every Sunday."},
            {"id": "c", "text": "Leave applications require seven days notice."},
        ],
        "top_k": 2,
    }
    r = client.post("/rerank", json=payload, headers=AUTH)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2, "top_k must cap the result count"
    ids = [x["id"] for x in results]
    assert ids[0] in ("a", "c"), "leave-related passage should rank first"
    assert "b" not in ids, "irrelevant passage should be dropped at top_k=2"
    for x in results:
        assert 0.0 < x["score"] < 1.0, "scores must be sigmoid-normalised"
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True), "results must be sorted desc"


def test_rerank_requires_secret():
    r = client.post("/rerank", json={"query": "x", "passages": [{"id": "1", "text": "y"}]})
    assert r.status_code == 401


def test_ingest_requires_secret():
    assert client.post("/ingest", json={"document_id": "x"}).status_code == 401
