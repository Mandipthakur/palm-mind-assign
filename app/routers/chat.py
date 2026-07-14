"""Conversational RAG API: multi-turn Q&A over ingested documents, plus an
in-band interview booking flow.

Routing logic per incoming message:
1. If the message (or an in-progress booking) signals booking intent,
   hand off to the booking slot-filling flow instead of RAG.
2. Otherwise, run the standard custom RAG pipeline.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedChunk
from app.services import booking
from app.services.memory import append_turn, get_history
from app.services.rag import answer_query

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def _conversation_text(session_id: str, latest_message: str) -> str:
    history = get_history(session_id)
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    lines.append(f"user: {latest_message}")
    return "\n".join(lines)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    return _handle_chat(request.session_id, request.message, db)


def _handle_chat(session_id: str, message: str, db: Session) -> ChatResponse:
    if booking.is_booking_intent(message, session_id):
        append_turn(session_id, "user", message)

        conversation = _conversation_text(session_id, message)
        slots = booking.extract_slots(session_id, conversation)

        if not slots.is_complete():
            reply = booking.next_prompt_for_missing_fields(slots)
            append_turn(session_id, "assistant", reply)
            return ChatResponse(session_id=session_id, answer=reply, booking_in_progress=True)

        errors = booking.validate_slots(slots)
        if errors:
            reply = "I found an issue with those details: " + "; ".join(errors) + ". Could you clarify?"
            append_turn(session_id, "assistant", reply)
            return ChatResponse(session_id=session_id, answer=reply, booking_in_progress=True)

        record = booking.persist_booking(db, session_id, slots)
        reply = (
            f"You're all set, {record.name}! Your interview is booked for "
            f"{record.interview_date} at {record.interview_time}. "
            f"A confirmation will be sent to {record.email}."
        )
        append_turn(session_id, "assistant", reply)
        return ChatResponse(session_id=session_id, answer=reply, booking_confirmed=True)

    result = answer_query(session_id, message)
    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        retrieved_chunks=[
            RetrievedChunk(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
            )
            for c in result.sources
        ],
    )
