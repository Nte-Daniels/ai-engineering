# AI Engineering Roadmap

Personal learning repository — building production-grade AI engineering skills across 8 phases.

---

## Phase 1 — Raw API Fluency (Complete)
- `session_01.py` — Live API calls, response object inspection, token tracking
- `session_02.py` — Pydantic structured outputs, JSON validation, error handling

**Stack:** Python, OpenAI API, Pydantic

---

## Phase 2 — FastAPI AI Backend (Complete)
- `main.py` — Async FastAPI service with health check, input validation, rate limiting, token tracking per request

**Stack:** FastAPI, uvicorn, slowapi

---

## Phase 3 — RAG From Scratch (Complete)
- `session_03.py` — Ingestion pipeline: loads PDFs, chunks text, embeds chunks using `all-MiniLM-L6-v2`, stores vectors in pgvector
- `session_03_retrieve.py` — Retrieval pipeline: embeds user query, runs cosine similarity search, retrieves top chunks, constructs grounded prompt, calls GPT-4o-mini

**Stack:** pgvector, PostgreSQL, sentence-transformers, pypdf, OpenAI

### How to run
1. Start pgvector container: `docker start pgvector-db`
2. Add PDFs to the `data/` folder
3. Run ingestion: `python session_03.py`
4. Run retrieval: `python .\session_03_retrieve.py`

### Known limitations
- Image-based PDFs produce zero chunks — OCR required for full coverage
- Chunking is character-based — semantic chunking planned as extension