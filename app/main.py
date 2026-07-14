"""FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""
import logging

from fastapi import FastAPI

from app.config import get_settings
from app.db.database import init_db
from app.routers import chat, ingestion
from app.services.vector_store import ensure_collection

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(settings.app_name)

app = FastAPI(
    title="Palm Mind RAG Service",
    description=(
        "Document ingestion + conversational RAG API with Redis-backed "
        "multi-turn memory and LLM-driven interview booking."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    ensure_collection()
    logger.info("Startup complete: SQL tables and Qdrant collection are ready.")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(ingestion.router)
app.include_router(chat.router)
