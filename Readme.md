# RAG MLOps Pipeline — Full MLOps Loop

![CI/CD](https://github.com/TJothiprakash/rag-mlops/actions/workflows/ci.yml/badge.svg)
![Drift Check](https://github.com/TJothiprakash/rag-mlops/actions/workflows/drift.yml/badge.svg)

A production-grade RAG pipeline with the full MLOps loop — experiment tracking, CI/CD, monitoring, and drift detection. Built as a portfolio project for MLOps Engineer roles.

## Live Links
- 🔬 **MLflow Dashboard:** https://dagshub.com/jothiprakash888/testing-mlflow.mlflow
- 📊 **Grafana Dashboard:** https://jothiprakash888.grafana.net
- 🐙 **GitHub Actions:** https://github.com/TJothiprakash/rag-mlops/actions

## Stack

| Layer | Tool |
|---|---|
| Experiment Tracking | MLflow + DagsHub |
| Model Registry | MLflow Model Registry |
| API Serving | FastAPI |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Metrics | Prometheus |
| Dashboards + Alerts | Grafana Cloud |
| Drift Detection | Evidently |

## Architecture
git push
↓
GitHub Actions
→ ML tests (model load, inference, schema, performance gate)
→ Docker build
→ Smoke test /health endpoint
↓
FastAPI (serves RAG pipeline)
→ /retrieve  — returns top-k chunks for a query
→ /metrics   — Prometheus metrics endpoint
→ /health    — health check
↓
Prometheus scrapes /metrics every 15s
↓
Grafana Cloud dashboard
→ request count, latency p95, similarity score distribution
→ alerts: latency > 500ms, similarity < 0.3
↓
Evidently (nightly GitHub Action)
→ compares today's queries vs baseline
→ flags drift if >50% columns shift
→ uploads drift_report.html as artifact
↓
MLflow + DagsHub
→ all experiment versions tracked
→ best model tagged Production

## Project Structure
rag-mlops/
├── app/
│   └── main.py              # FastAPI app with Prometheus metrics
├── tests/
│   └── test_pipeline.py     # ML-specific tests + performance gate
├── drift/
│   └── drift_check.py       # Evidently nightly drift detection
├── .github/workflows/
│   ├── ci.yml               # CI/CD pipeline
│   └── drift.yml            # Nightly drift check
├── Dockerfile
├── requirements.txt
└── README.md

## Run Locally

```bash
# Build and run
docker build -t rag-mlops .
docker run -p 8000:8000 rag-mlops

# Run tests
docker run rag-mlops pytest tests/ -v

# Run drift check
python drift/drift_check.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Root |
| `/health` | GET | Health check |
| `/retrieve` | POST | Retrieve top-k chunks |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |

## Interview Answers

**"Walk me through how you would deploy and monitor a RAG system in production."**
> "Every push triggers GitHub Actions — ML tests run first, including a performance gate that blocks deployment if similarity drops below threshold. The FastAPI app is containerised with Docker and exposes a /metrics endpoint scraped by Prometheus every 15 seconds. Grafana Cloud dashboards show request rate, p95 latency, and similarity score distribution with alert rules. Evidently runs nightly comparing today's query distribution against baseline — if more than 50% of columns drift, it triggers a retraining alert. All experiment versions are tracked in MLflow on DagsHub with the current production model tagged."

**"What do you test in an ML CI pipeline?"**
> "Model loads correctly, inference runs without error, output schema matches expected structure, and a golden test set keeps average similarity above 0.3. If any test fails — especially the performance gate — the Docker build never happens."

**"What metrics do you monitor for an LLM-based system?"**
> "Request count, latency p50/p95/p99, similarity score distribution, and index size. Unlike classical ML where you monitor accuracy, RAG systems need retrieval quality metrics — similarity scores tell you whether the model is finding relevant chunks."