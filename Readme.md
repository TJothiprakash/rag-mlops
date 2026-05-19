# RAG MLOps Pipeline

![CI/CD](https://github.com/TJothiprakash/rag-mlops/actions/workflows/ci.yml/badge.svg)

A production-grade RAG pipeline with full MLOps loop.

## Stack
- FastAPI — model serving
- FAISS + sentence-transformers — retrieval
- Docker — containerisation
- GitHub Actions — CI/CD with ML performance gate
- MLflow + DagsHub — experiment tracking

## Run locally
docker build -t rag-mlops .
docker run -p 8000:8000 rag-mlops

## Run tests
docker run rag-mlops pytest tests/ -v