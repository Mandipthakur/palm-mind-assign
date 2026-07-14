"""Request/response schemas for the document ingestion API."""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    chunk_strategy: ChunkStrategy
    num_chunks: int
    char_count: int
    created_at: datetime


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str
    chunk_strategy: ChunkStrategy
    num_chunks: int
    char_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
