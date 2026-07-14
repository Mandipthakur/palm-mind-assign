"""Request/response schemas for the conversational RAG API."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable identifier for the conversation thread.")
    message: str = Field(..., min_length=1)


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    booking_in_progress: bool = False
    booking_confirmed: bool = False
