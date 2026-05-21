"""
drift_check.py
--------------
Nightly drift detection using Evidently.
Compares reference queries vs live queries.
Triggered by GitHub Actions on a schedule.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss

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

REFERENCE_QUERIES = [
    "What is machine learning?",
    "How does FAISS work?",
    "What is RAG?",
    "What are embeddings?",
    "What is Docker?",
    "What is Prometheus?",
    "How does Grafana work?",
    "What is MLflow?",
    "What is CI/CD?",
    "What is a vector database?",
]

LIVE_QUERIES = [
    "How to install Python on Windows?",
    "What is supervised learning?",
    "Explain neural networks",
    "What is gradient descent?",
    "How does backpropagation work?",
    "What is overfitting in ML?",
    "Explain transfer learning",
    "What is a transformer model?",
    "How does attention mechanism work?",
    "What is fine-tuning in deep learning?",
]

DRIFT_THRESHOLD = 0.5


def build_pipeline():
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(DOCUMENTS)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index


def retrieve(query, model, index, top_k=3):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    scores = [float(1 / (1 + d)) for d in distances[0]]
    return {
        "query": query,
        "top_score": scores[0],
        "avg_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }


def run_drift_check():
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    model, index = build_pipeline()

    ref_data = [retrieve(q, model, index) for q in REFERENCE_QUERIES]
    live_data = [retrieve(q, model, index) for q in LIVE_QUERIES]

    ref_df = pd.DataFrame(ref_data).drop(columns=["query"])
    live_df = pd.DataFrame(live_data).drop(columns=["query"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=live_df)
    report.save_html("drift_report.html")

    result = report.as_dict()["metrics"][0]["result"]
    share_drifted = result["share_of_drifted_columns"]
    drifted = result["dataset_drift"]

    print(f"\n{'='*50}")
    print(f"Drift Check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset drifted : {drifted}")
    print(f"Share drifted   : {share_drifted:.0%}")
    print(f"{'='*50}")

    if share_drifted >= DRIFT_THRESHOLD:
        print("🚨 ALERT: Drift threshold exceeded! Retraining recommended.")
        return 1  # exit code 1 = alert
    else:
        print("✅ Drift within acceptable range.")
        return 0


if __name__ == "__main__":
    import sys
    exit_code = run_drift_check()
    sys.exit(exit_code)