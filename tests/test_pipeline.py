import pytest
from fastapi.testclient import TestClient
from app.main import app, index, model, SIMILARITY_THRESHOLD

client = TestClient(app)


# ── Test 1: model loads ────────────────────────────────────────────────────
def test_model_loads():
    """Model must load and index must have vectors."""
    assert model is not None
    assert index.ntotal > 0


# ── Test 2: health endpoint ────────────────────────────────────────────────
def test_health_endpoint():
    """Health check must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["index_size"] > 0


# ── Test 3: inference runs ─────────────────────────────────────────────────
def test_inference_runs():
    """Retrieve endpoint must return a valid response."""
    response = client.post("/retrieve", json={"query": "What is machine learning?"})
    assert response.status_code == 200


# ── Test 4: output schema is correct ──────────────────────────────────────
def test_output_schema():
    """Response must match the expected schema exactly."""
    response = client.post("/retrieve", json={"query": "What is FAISS?"})
    data = response.json()

    assert "query" in data
    assert "results" in data
    assert "latency_ms" in data
    assert "model" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    assert "chunk" in data["results"][0]
    assert "similarity_score" in data["results"][0]


# ── Test 5: performance gate ───────────────────────────────────────────────
def test_similarity_above_threshold():
    """
    Golden test set — similarity must stay above threshold.
    This is the ML-specific gate that blocks bad model versions.
    """
    golden_queries = [
        "What is machine learning?",
        "How does FAISS work?",
        "What is RAG?",
        "What are embeddings?",
        "What is Docker?",
    ]

    scores = []
    for query in golden_queries:
        response = client.post("/retrieve", json={"query": query})
        data = response.json()
        top_score = data["results"][0]["similarity_score"]
        scores.append(top_score)

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage similarity score: {avg_score:.4f}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")

    assert avg_score >= SIMILARITY_THRESHOLD, (
        f"Performance gate failed: avg_similarity {avg_score:.4f} "
        f"below threshold {SIMILARITY_THRESHOLD}"
    )