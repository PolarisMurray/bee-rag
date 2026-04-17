# Beekeeper Research Intelligence Platform

Production-oriented backend for beekeeper need discovery and grounded research workflows.

This project is not a generic chatbot. It is designed to help research and innovation teams analyze beekeeper pain points, workflows, unmet needs, and product opportunities with traceable evidence and citations.

## What This Project Does

- Ingests and normalizes research evidence into typed domain models
- Processes user queries (intent classification, rewrite, expansion, HyDE-style enrichment)
- Manages multi-turn context for follow-up questions
- Orchestrates retrieval, extraction, critique, and synthesis
- Returns grounded answers and structured research reports with citations
- Exposes a FastAPI interface and a lightweight web UI for interactive debugging

## Current State

The repository is currently set up for end-to-end debugging with a default `DemoRetriever`.

- `/query` and `/research/report` work out of the box for local testing
- Returned evidence is synthetic debug data unless you inject a real retriever
- API and architecture are ready for plugging in production retrieval and LLM generation

## Project Structure

```text
RAG/
├── beekeeper_intel/
│   ├── api/                 # FastAPI app, routes, schemas, middleware, debug UI
│   ├── orchestration/       # End-to-end orchestration pipeline
│   ├── query_processing/    # Intent, rewrite, expansion, HyDE, follow-up handling
│   ├── memory/              # Multi-turn context tracking and retrieval hints
│   ├── agents/              # Need extraction and aggregation
│   ├── citations/           # Citation rendering and explainability bundles
│   ├── evaluation/          # Retrieval/grounding/report evaluation utilities
│   ├── models.py            # Core Pydantic domain models
│   └── __init__.py
├── tests/                   # Unit tests for modules
├── requirements.txt
├── pytest.ini
└── README.md
```

## Architecture Overview

1. API receives request (`/query` or `/research/report`)
2. Session context is loaded/updated
3. Query is resolved for follow-up ambiguity and processed for retrieval
4. Planner builds a retrieval plan
5. Retriever fetches evidence
6. Extraction agent derives structured need signals
7. Critic validates evidence sufficiency, can trigger one more iteration
8. Synthesis creates final answer/report
9. Citation module renders user-facing citations and provenance metadata

## Requirements

- Python 3.11+ recommended
- macOS/Linux/WSL supported

## Quick Start

### 1) Create virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run API server

```bash
uvicorn beekeeper_intel.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Open interactive UI

- Debug UI: [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
- OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Endpoints

### Health and Metrics

- `GET /` - service info and entry links
- `GET /health` - liveness info
- `GET /metrics` - in-memory request/latency counters

### Query and Report

- `POST /query` - conversational grounded Q&A mode
- `POST /research/report` - structured research synthesis mode
- `POST /documents/ingest` - ingestion adapter endpoint (fallbacks if no service injected)

## Request Examples

### POST /query

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do hobbyist beekeepers currently deal with varroa monitoring burden?",
    "session_id": "demo-session-1",
    "user_id": "alex",
    "include_trace": true
  }'
```

### POST /research/report

```bash
curl -X POST http://127.0.0.1:8000/research/report \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare hobbyist vs commercial pain points in varroa monitoring and unmet needs.",
    "session_id": "demo-session-1",
    "include_trace": true
  }'
```

## Response Characteristics

`/query` responses include:

- `answer` (rendered with inline citations)
- `citations` (structured citation metadata)
- `evidence` (retrieved evidence payload)
- `trace` (optional orchestration steps)

`/research/report` responses include:

- `executive_summary`
- `needs_count`
- `citations`
- `evidence_map` by section
- `trace` (optional)

## Debug UI Usage

1. Choose mode:
   - `Conversational Q&A`
   - `Research Report`
2. Enter query text
3. Optionally set `Session ID` (for multi-turn continuity)
4. Click `Run`
5. Inspect JSON output in the response panel

## Testing

Run all tests:

```bash
pytest -q
```

Run a specific module test:

```bash
pytest tests/test_orchestrator.py -q
```

## Configuration Notes

### Environment Variables

The current debug setup does not require API keys. If you integrate a real LLM provider, store secrets in environment variables or a `.env` file and never commit them.

Suggested key names for provider-based generation:

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini

DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

GEMINI_API_KEY=your_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GEMINI_MODEL=gemini-2.5-flash
```

### Important

There is currently no LLM provider adapter module in this repo. Adding one requires:

- LLM client module (`beekeeper_intel/llm/...`)
- settings/env loading
- orchestrator synthesis integration

## Moving From Demo to Production

To productionize this stack:

1. Replace `DemoRetriever` with your real hybrid retriever adapter
2. Inject ingestion service implementation for `/documents/ingest`
3. Add LLM provider client for generation/reporting
4. Persist session state and metrics externally (Redis/DB/Prometheus)
5. Add auth, rate limits, and structured logging sinks
6. Add CI tests and deployment configuration

## Known Limitations

- Default retrieval is synthetic debug data unless replaced
- Metrics are in-memory (reset on restart)
- Session state is in-memory (not persisted across process restarts)
- No authentication middleware yet

## License

Internal project scaffold. Add your preferred license before external distribution.
