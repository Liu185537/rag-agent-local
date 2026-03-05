# RAG-Agent Local (Interview-Oriented)
[![CI](https://github.com/Liu185537/rag-agent-local/actions/workflows/ci.yml/badge.svg)](https://github.com/Liu185537/rag-agent-local/actions/workflows/ci.yml)

This project is a local-first RAG + Agent demo designed for interview scenarios.
It intentionally removes all company-specific IPs, ports, and API keys.

## 1. What this project demonstrates

- End-to-end RAG pipeline:
  - Ingestion -> chunking -> embedding -> vector index
  - Hybrid retrieval (Vector + BM25 with RRF fusion)
  - Lightweight reranker (semantic + lexical + coverage)
  - Citation-aware answer generation
- Agent workflow:
  - Planner -> tool execution -> response synthesis
  - Session memory (SQLite)
  - Profile memory tool (read/update user profile)
- Engineering basics:
  - Config by `.env`
  - Local persistence with SQLite + Chroma
  - Upload parser for txt/md/csv/json/pdf/docx
  - Request metrics + lightweight dashboard
  - Evaluation script for quick metric checks

## 2. Stack

- Backend: FastAPI
- Vector store: Chroma (local persistent)
- Memory and metadata: SQLite
- LLM/Embedding: Ollama (local), with fallback behavior if unavailable

## 3. Quick start

### 3.1 Prerequisites

- Python 3.10+
- Optional but recommended: Ollama running locally (`http://127.0.0.1:11434`)

### 3.2 Install

```bash
cd Rag-Agent
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### 3.3 Configure

```bash
copy .env.example .env
```

All defaults are local-only. You can run without changing anything.

### 3.4 Run API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8008 --reload
```

Swagger UI: `http://127.0.0.1:8008/docs`

## 4. Demo flow

### 4.1 Ingest demo knowledge

```bash
python scripts/ingest_demo.py --base-url http://127.0.0.1:8008 --namespace demo
```

### 4.2 Ask questions

```bash
curl -X POST "http://127.0.0.1:8008/api/v1/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"namespace\":\"demo\",\"message\":\"How should I handle price objections in first chat?\"}"
```

### 4.3 Upload files directly

```bash
curl -X POST "http://127.0.0.1:8008/api/v1/knowledge/upload" ^
  -F "namespace=demo" ^
  -F "file=@data\\sample_knowledge.md"
```

### 4.4 View metrics/dashboard

- JSON metrics: `http://127.0.0.1:8008/api/v1/metrics`
- Dashboard: `http://127.0.0.1:8008/dashboard`
- Playground: `http://127.0.0.1:8008/playground`
  - Layout style is adapted from the `Knowleadge.vue` interaction pattern (left knowledge panel + right chat/config tabs).
  - Agent config tab now persists to backend API (`/api/v1/agent-config`) instead of browser-only storage.

### 4.5 Run lightweight evaluation

```bash
python scripts/run_eval.py --base-url http://127.0.0.1:8008 --namespace demo
```

The script now outputs:
- Console summary (hit rates + latency p50/p90/p95)
- `eval/reports/*.json` detailed report
- `eval/reports/*.csv` per-sample result

### 4.6 Generate larger dataset (50+)

```bash
python scripts/generate_eval_dataset.py --size 60 --output eval/dataset_v2.jsonl
python scripts/run_eval.py --dataset eval/dataset_v2.jsonl --namespace demo
```

## 5. Project layout

```text
Rag-Agent/
  app/
    main.py
    api/schemas.py
    core/{config.py,database.py,logging.py,observability.py}
    llm/client.py
    rag/{chunker.py,document_parser.py,embedding.py,indexer.py,retriever.py,reranker.py}
    agent/{tools.py,orchestrator.py}
  scripts/{ingest_demo.py,run_eval.py,generate_eval_dataset.py}
  tests/
  eval/dataset.jsonl
  eval/dataset_v2.jsonl
  docker-compose.yml
  Dockerfile
  data/sample_knowledge.md
  .env.example
  pyproject.toml
```

## 6. Interview talking points

- Why hybrid retrieval is better than pure vector in noisy business corpora.
- How planning + tools improve controllability compared with single-shot prompting.
- How to measure quality:
  - retrieval hit@k
  - citation coverage
  - task success / latency / cost
- What you would do next:
  - add reranker model
  - add async ingestion pipeline
  - add guardrails and observability (e.g., tracing)

## 7. Security and privacy

- No company endpoints in code defaults.
- No API keys committed.
- Local-only defaults in `.env.example`.

## 8. Test and docker

Run tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI:
- GitHub Actions workflow: `.github/workflows/ci.yml`
- Pipeline stages:
  - `ruff check app tests scripts`
  - `pytest -q tests`
  - `docker build`

Run with Docker:

```bash
docker compose up --build
```

Then open:
- `http://127.0.0.1:8008/docs`
- `http://127.0.0.1:8008/playground`
