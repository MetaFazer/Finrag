# FinRAG

> A production-grade, citation-enforced financial research assistant over SEC filings and earnings call transcripts.

[![Quality Gate](https://github.com/MetaFazer/Finrag/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/MetaFazer/Finrag/actions/workflows/quality-gate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## What This Does

FinRAG answers questions about SEC filings (10-K, 10-Q, 8-K) and earnings call transcripts. Every answer is grounded in a specific paragraph from a specific filing, with company, period, section, and page attached. When evidence doesn't support a claim, the system **refuses to answer** rather than hallucinate.

### Key Capabilities

- **Citation-enforced answers** — every claim maps to a source chunk with filing reference, section, and page
- **Hybrid retrieval** — BM25 sparse + dense vector search fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — precision-focused second-stage reranking
- **Multi-turn conversations** — entity tracking, reference resolution, session memory
- **Guardrails** — prompt injection detection, PII filtering, output validation
- **Streaming API** — Server-Sent Events for progressive UI rendering
- **Distributed tracing** — Langfuse integration with per-request cost tracking
- **Automated evaluation** — 50-item golden dataset, RAGAS metrics, LLM-as-Judge citation scoring
- **CI quality gates** — builds fail if faithfulness < 0.85 or citation coverage < 0.90

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Layer                            │
│  POST /query  │  POST /query/stream  │  GET /metrics           │
└──────┬────────┴──────────┬───────────┴──────────┬──────────────┘
       │                   │                      │
       ▼                   ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                    │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Input    │──▶│ Retrieve │──▶│ Rerank   │──▶│ Route    │ │
│  │  Guard    │   │  (Hybrid)│   │ (Cross-  │   │ (Keyword │ │
│  │          │   │          │   │  Encoder) │   │  Router) │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                      │       │
│                    ┌─────────────────┬───────────────┘       │
│                    ▼                 ▼                        │
│              ┌──────────┐     ┌──────────┐                   │
│              │ Generate │     │ Calculate │                   │
│              │ (Gemini) │     │ (Gemini)  │                   │
│              └────┬─────┘     └────┬─────┘                   │
│                   │                │                          │
│                   ▼                ▼                          │
│              ┌──────────┐   ┌──────────┐                     │
│              │ Validate │──▶│ Output   │                     │
│              │ Citations│   │ Guard    │                     │
│              └──────────┘   └──────────┘                     │
└──────────────────────────────────────────────────────────────┘
       │                                          │
       ▼                                          ▼
┌──────────────┐                        ┌──────────────────┐
│  ChromaDB    │                        │    Langfuse       │
│  + BM25      │                        │    Tracing        │
│  Vector Store│                        │    + Metrics      │
└──────────────┘                        └──────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (state machine with conditional routing) |
| Vector Store | ChromaDB (persistent, metadata-filtered) |
| Sparse Retrieval | BM25 via `rank-bm25` |
| Dense Retrieval | `sentence-transformers` (all-MiniLM-L6-v2) |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | Google Gemini 2.0 Flash via `langchain-google-genai` |
| API | FastAPI + SSE (`sse-starlette`) |
| Guardrails | Custom regex + policy-based input/output guards |
| Observability | Langfuse (traces, spans, token costs) |
| Evaluation | RAGAS metrics + LLM-as-Judge citation scorer |
| Config | `pydantic-settings` + versioned YAML prompts |
| CI | GitHub Actions (lint → test → eval gate) |

---

## Setup

### Prerequisites

- Python 3.11+
- Google API key (for Gemini LLM)

### Installation

```bash
# Clone the repo
git clone https://github.com/MetaFazer/Finrag.git
cd finrag

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Environment Configuration

```bash
# Copy example env file
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required: Google Gemini API key
GOOGLE_API_KEY=your_key_here

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: API authentication
FINRAG_API_KEY=your_api_secret
```

---

## Quick Start

### 1. Ingest a Filing

```bash
# Download and process Apple's latest 10-K
python scripts/ingest.py --ticker AAPL --filing-type 10-K --count 1
```

This downloads the filing from SEC EDGAR, parses sections, chunks with metadata, and indexes into ChromaDB + BM25.

### 2. Start the API Server

```bash
uvicorn finrag.api.app:app --reload --port 8000
```

### 3. Query the Pipeline

```bash
# Synchronous query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total net revenue for fiscal year 2024?"}'

# Streaming query (SSE)
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple total net revenue for fiscal year 2024?"}'
```

### 4. Check Metrics

```bash
curl http://localhost:8000/api/v1/metrics
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Synchronous JSON response |
| `/api/v1/query/stream` | POST | Server-Sent Events streaming |
| `/api/v1/sessions/{id}` | GET | Session state inspection |
| `/api/v1/sessions/{id}` | DELETE | Clear a session |
| `/api/v1/config/prompts` | GET | Active prompt versions |
| `/api/v1/metrics` | GET | Production metrics (p50/p95 latency, costs, rates) |

### Query Request

```json
{
  "query": "What was Apple's free cash flow in FY2024?",
  "session_id": "optional-session-id",
  "metadata_filter": {"ticker": "AAPL"}
}
```

### Query Response

```json
{
  "answer": "Apple's free cash flow in FY2024 was...",
  "citations": [
    {
      "chunk_id": "abc123",
      "filing_reference": "AAPL 10-K FY2024, Item 7 - MD&A",
      "section": "Item 7",
      "relevance_score": 0.92
    }
  ],
  "session_id": "auto-generated-uuid",
  "confidence": 0.87,
  "route": "retrieve",
  "prompt_version": "v2",
  "metadata": {
    "request_id": "uuid",
    "trace_id": "langfuse-trace-id",
    "total_latency_ms": 1250
  }
}
```

---

## Evaluation

### Golden Dataset

50 manually verified Q/A pairs across 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Numerical Extraction | 15 | Direct financial data queries |
| Multi-hop Comparison | 12 | Cross-document reasoning |
| Contradiction Detection | 11 | Narrative vs. data consistency |
| Out-of-scope | 12 | Should produce decline, not hallucination |

### Run Evaluations

```bash
# RAGAS metrics (faithfulness, relevancy, precision, coverage)
python -m finrag.evaluation.run_eval --mode ragas --threshold 0.85

# LLM-as-Judge citation scoring
python -m finrag.evaluation.run_eval --mode judge --threshold 0.90

# Full evaluation (both)
python -m finrag.evaluation.run_eval --mode full --output report.json

# Filter by category
python -m finrag.evaluation.run_eval --mode ragas --category numerical
```

### CI Quality Gates

Every PR triggers the [quality gate workflow](.github/workflows/quality-gate.yml):

```
lint → unit tests (60% coverage) → RAGAS eval (≥0.85) → Judge eval (≥0.90)
```

Builds fail if quality thresholds are not met.

---

## Project Structure

```
finrag/
├── .github/workflows/        # CI quality gate
│   └── quality-gate.yml
├── configs/                   # Versioned prompt configs (YAML)
├── scripts/
│   └── ingest.py              # EDGAR ingestion CLI
├── src/finrag/
│   ├── ingestion/             # EDGAR client, section chunker
│   ├── vectorstore/           # ChromaDB store
│   ├── retrieval/             # BM25, hybrid retriever
│   ├── orchestration/         # LangGraph, nodes, routing, memory
│   ├── guardrails/            # Input/output guards
│   ├── api/                   # FastAPI app, routes, middleware, MCP
│   ├── observability/         # Langfuse tracer, metrics
│   └── evaluation/            # Golden dataset, RAGAS, LLM-as-Judge
├── tests/                     # 16 test modules, 300+ tests
├── ROADMAP.md                 # 15-day build roadmap
├── DEBT_LEDGER.md             # Technical debt tracking
└── pyproject.toml             # Dependencies and tooling config
```

---

## Development

### Run Tests

```bash
# All tests
python -m pytest tests/ -v --tb=short

# Specific day/module
python -m pytest tests/test_integration.py -v

# With coverage
python -m pytest tests/ --cov=finrag --cov-report=term-missing
```

### Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse tracing (public key) |
| `LANGFUSE_SECRET_KEY` | No | Langfuse tracing (secret key) |
| `FINRAG_API_KEY` | No | API bearer token authentication |
| `FINRAG_INIT_PIPELINE` | No | Set `false` to skip pipeline init (testing) |

---

## Build Timeline

This project was built in 15 days following a structured roadmap:

| Phase | Days | Focus |
|-------|------|-------|
| Foundation | 1–3 | EDGAR ingestion, chunking, vector store |
| Retrieval | 4–6 | BM25, hybrid fusion, cross-encoder reranking |
| Generation & Safety | 7–10 | LangGraph, citations, guardrails, memory |
| API & Observability | 11–12 | FastAPI, SSE, Langfuse tracing |
| Evaluation & CI | 13–15 | Golden dataset, RAGAS, LLM-as-Judge, CI gates |

See [ROADMAP.md](ROADMAP.md) for full details and [DEBT_LEDGER.md](DEBT_LEDGER.md) for known technical debt.

---

## Deployment

This project uses a two-branch strategy:

| Branch   | Purpose              | Vector Store      | Reranker Model          |
|----------|----------------------|-------------------|-------------------------|
| `main`   | Local development    | ChromaDB local    | MiniLM-L-6-v2           |
| `deploy` | Cloud deployment     | ChromaDB Cloud    | MiniLM-L-2-v2           |

### Live Demo
- **Frontend:** [your-app.vercel.app](https://your-app.vercel.app)
- **Backend:** Deployed on Render free tier
- **Vector store:** ChromaDB Cloud free tier

> **Note on cold starts:** The backend is hosted on Render's free tier.
> If the service has been inactive, the first request may take 20–30
> seconds to wake up. Subsequent requests are fast. This is a known
> free-tier constraint managed with a keepalive ping every 14 minutes.

### Running Locally
Checkout `main` branch and follow the setup instructions above.
Local setup uses a persistent ChromaDB instance and the full
L-6 reranker model with no cold start constraints.

---

## License

MIT
