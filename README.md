# Palm Mind RAG Service

A modular FastAPI backend with two REST APIs:

1. **Document Ingestion API** — upload `.pdf`/`.txt`, chunk (two selectable
   strategies), embed, and store in Qdrant + SQL metadata store.
2. **Conversational RAG API** — custom-built retrieval + generation (no
   `RetrievalQAChain`), Redis-backed multi-turn memory, and an in-band
   LLM-driven interview booking flow.

## Architecture

```
app/
├── main.py                 # FastAPI app, startup hooks, router registration
├── config.py                # Typed settings loaded from environment
├── db/
│   ├── database.py           # SQLAlchemy engine/session (SQLserver)
│   └── models.py              # Document + Booking ORM models
├── schemas/                  # Pydantic request/response models
│   ├── ingestion.py
│   ├── chat.py
│   └── booking.py
├── services/                 # Business logic, framework-agnostic
│   ├── text_extraction.py     # PDF/TXT -> raw text
│   ├── chunking.py             # fixed-size + recursive chunking strategies
│   ├── embeddings.py            # sentence-transformers embedding generation
│   ├── vector_store.py           # Qdrant wrapper (swap-in point for Weaviate/Milvus)
│   ├── llm_client.py              # OpenAI/Anthropic-agnostic chat + JSON extraction
│   ├── memory.py                   # Redis-backed chat history + booking slots
│   ├── rag.py                       # Custom retrieval -> prompt -> generation pipeline
│   └── booking.py                    # Booking intent detection, slot-filling, persistence
└── routers/
    ├── ingestion.py            # POST /api/v1/documents/upload
    └── chat.py                  # POST /api/v1/chat
```

### Design notes

- **No FAISS/Chroma/RetrievalQAChain**: `services/vector_store.py` talks to
  Qdrant directly, and `services/rag.py` hand-rolls the retrieve → build
  prompt → inject history → generate flow instead of using a prebuilt chain.
- **Embeddings are local** (`sentence-transformers/all-MiniLM-L6-v2`), so
  ingestion has no dependency on an LLM API key or per-request cost.
- **LLM provider is pluggable**: `LLM_PROVIDER=openai` or `anthropic` in
  `.env` — used only for the generation step and for booking slot
  extraction, not for embeddings.
- **Booking is conversational, not a form**: `services/booking.py` detects
  booking intent, asks the LLM to extract `{name, email, date, time}` from
  the whole conversation on every turn (so users can fill slots across
  multiple messages, in any order), asks follow-up questions for missing
  fields, validates format, and persists once complete.
- **Metadata vs. vectors**: chunk text/vectors live in Qdrant; document
  metadata (filename, chunk count, strategy used) and bookings live in
  SQL — keeps the relational store lightweight and query-friendly.

## Setup

### 1. Start infra (Qdrant + Redis)

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# then edit .env: set LLM_PROVIDER and the matching API key
# (OPENAI_API_KEY or ANTHROPIC_API_KEY)
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

## Usage examples

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@sample.pdf" \
  -F "chunk_strategy=recursive" \
  -F "chunk_size=800" \
  -F "chunk_overlap=100"
```

### Ask a question (RAG)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "What does the document say about pricing?"}'
```

### Book an interview (multi-turn)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "I would like to book an interview"}'

# assistant asks for missing details, e.g. name/email/date/time

curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user-123", "message": "I am Mandip, email mandip@example.com, next Monday at 3pm"}'
```

## Testing

Chunking logic is covered by pure-unit tests with no external services required:

```bash
pytest tests/ -v
```

## Notes / trade-offs

- SQL Server is used for zero-setup local development; `DATABASE_URL` can be
  pointed at Postgres/MySQL without code changes (SQLAlchemy handles the
  dialect).
- Booking date/time normalization relies on the LLM extraction step; for a
  production system this would be backed by a proper date-parsing library
  (e.g. `dateparser`) as a secondary validation layer.
- No UI is included per the task constraints — all interaction is via the
  two REST APIs (see `/docs` for interactive Swagger UI).
