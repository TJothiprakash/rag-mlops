from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import time

app = FastAPI(title="RAG Pipeline API")

# ── Load model and build index at startup ──────────────────────────────────
DOCUMENTS = [
    "Python is a high-level, interpreted programming language known for its readability.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation.",
    "Sentence Transformers provide dense vector representations of sentences for semantic search.",
    "MLflow is an open-source platform for managing the ML lifecycle.",
    "Docker is a platform for building, shipping, and running applications in containers.",
    "Prometheus is an open-source systems monitoring and alerting toolkit.",
    "Grafana is an open-source analytics and interactive visualization web application.",
    "Evidently is a Python library for monitoring ML model performance and data drift.",
    "CI/CD stands for Continuous Integration and Continuous Delivery.",
    "A vector database stores high-dimensional vectors for fast similarity search.",
    "Embeddings are numerical representations of data in a continuous vector space.",
    "The transformer architecture introduced in 2017 revolutionized natural language processing.",
    "DagsHub is a platform for data scientists to version and collaborate on ML projects.",
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("Building FAISS index...")
embeddings = model.encode(DOCUMENTS)
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"Index ready — {index.ntotal} vectors")


# ── Request / Response schemas ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class RetrievedChunk(BaseModel):
    chunk: str
    similarity_score: float


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    latency_ms: float
    model: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — used by CI and blue-green deployment."""
    return {
        "status": "ok",
        "model": EMBEDDING_MODEL,
        "index_size": index.ntotal,
    }


@app.post("/retrieve", response_model=QueryResponse)
def retrieve(request: QueryRequest):
    """Retrieve top-k relevant chunks for a query."""
    query_vec = model.encode([request.query]).astype("float32")

    t0 = time.time()
    distances, indices = index.search(query_vec, request.top_k)
    latency_ms = (time.time() - t0) * 1000

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append(RetrievedChunk(
            chunk=DOCUMENTS[idx],
            similarity_score=float(1 / (1 + dist)),
        ))

    return QueryResponse(
        query=request.query,
        results=results,
        latency_ms=latency_ms,
        model=EMBEDDING_MODEL,
    )


@app.get("/")
def root():
    return {"message": "RAG Pipeline API — visit /docs for Swagger UI"}