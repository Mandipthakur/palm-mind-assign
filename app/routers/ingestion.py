"""Document ingestion API: upload -> extract -> chunk -> embed -> store."""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Document
from app.schemas.ingestion import ChunkStrategy, DocumentIngestResponse
from app.services import embeddings, text_extraction, vector_store
from app.services.chunking import chunk_text

router = APIRouter(prefix="/api/v1/documents", tags=["ingestion"])
settings = get_settings()


@router.post("/upload", response_model=DocumentIngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: ChunkStrategy = Form(default=ChunkStrategy(settings.default_chunk_strategy)),
    chunk_size: int = Form(default=settings.fixed_chunk_size),
    chunk_overlap: int = Form(default=settings.fixed_chunk_overlap),
    db: Session = Depends(get_db),
) -> DocumentIngestResponse:
    """Upload a .pdf or .txt file, chunk it, embed it, and store it.

    - **chunk_strategy**: "fixed" or "recursive"
    - **chunk_size** / **chunk_overlap**: characters per chunk / overlap
    """
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text = text_extraction.extract_text(raw_bytes, file)
    chunks = chunk_text(text, chunk_strategy, chunk_size, chunk_overlap)

    if not chunks:
        raise HTTPException(status_code=400, detail="Chunking produced no chunks for this document.")

    document = Document(
        filename=file.filename or "unnamed",
        content_type=file.content_type or "unknown",
        chunk_strategy=chunk_strategy.value,
        num_chunks=len(chunks),
        char_count=len(text),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    chunk_texts = [c.text for c in chunks]
    vectors = embeddings.embed_texts(chunk_texts)
    vector_store.upsert_chunks(
        document_id=document.id,
        filename=document.filename,
        chunk_texts=chunk_texts,
        chunk_vectors=vectors,
    )

    return DocumentIngestResponse(
        document_id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        chunk_strategy=chunk_strategy,
        num_chunks=document.num_chunks,
        char_count=document.char_count,
        created_at=document.created_at,
    )
