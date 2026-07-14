"""Interview booking capture via LLM-based slot extraction.

Flow:
1. Detect whether the user's message signals booking intent (or we're
   already mid-booking for this session).
2. Ask the LLM to extract {name, email, interview_date, interview_time}
   from the *entire* conversation so far (merged with anything already
   captured), so users can supply details across multiple turns.
3. If any slot is still missing, generate a natural follow-up question.
4. Once all slots are present, validate and persist to the SQL DB.
"""
import re
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Booking
from app.schemas.booking import BookingSlots
from app.services.llm_client import get_llm_client
from app.services.memory import (
    clear_booking_slots,
    get_booking_slots,
    set_booking_slots,
)

_BOOKING_INTENT_PATTERN = re.compile(
    r"\b(book|schedule|set up|arrange).{0,20}(interview|call|meeting|appointment)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_EXTRACTION_SYSTEM_PROMPT = """
You extract interview booking details from a conversation.
Return a JSON object with exactly these keys: "name", "email", "interview_date", "interview_time".
- Use null for any field not yet mentioned anywhere in the conversation.
- Normalize interview_date to YYYY-MM-DD if a specific date is given (resolve relative dates
  like "tomorrow" or "next Monday" using the provided current_date).
- Normalize interview_time to 24-hour HH:MM.
- Do not invent values that were not stated.
"""


def is_booking_intent(message: str, session_id: str) -> bool:
    existing = get_booking_slots(session_id)
    already_mid_booking = existing is not None and not existing.is_complete()
    return already_mid_booking or bool(_BOOKING_INTENT_PATTERN.search(message))


def extract_slots(session_id: str, conversation_text: str) -> BookingSlots:
    existing = get_booking_slots(session_id) or BookingSlots()
    llm = get_llm_client()

    user_content = (
        f"current_date: {datetime.utcnow().date().isoformat()}\n\n"
        f"conversation:\n{conversation_text}\n\n"
        f"already_captured: {existing.model_dump_json()}"
    )
    result = llm.extract_json(_EXTRACTION_SYSTEM_PROMPT, user_content)

    merged = BookingSlots(
        name=result.get("name") or existing.name,
        email=result.get("email") or existing.email,
        interview_date=result.get("interview_date") or existing.interview_date,
        interview_time=result.get("interview_time") or existing.interview_time,
    )
    set_booking_slots(session_id, merged)
    return merged


def validate_slots(slots: BookingSlots) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    errors: list[str] = []
    if slots.email and not _EMAIL_RE.match(slots.email):
        errors.append(f"'{slots.email}' doesn't look like a valid email address")
    if slots.interview_date:
        try:
            datetime.strptime(slots.interview_date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"'{slots.interview_date}' isn't a recognizable date (expected YYYY-MM-DD)")
    if slots.interview_time:
        try:
            datetime.strptime(slots.interview_time, "%H:%M")
        except ValueError:
            errors.append(f"'{slots.interview_time}' isn't a recognizable time (expected HH:MM)")
    return errors


def next_prompt_for_missing_fields(slots: BookingSlots) -> str:
    missing = slots.missing_fields()
    friendly = {
        "name": "your full name",
        "email": "your email address",
        "interview_date": "your preferred interview date",
        "interview_time": "your preferred interview time",
    }
    asks = ", ".join(friendly[f] for f in missing)
    return f"Sure, I can help set up your interview. Could you share {asks}?"


def persist_booking(db: Session, session_id: str, slots: BookingSlots) -> Booking:
    booking = Booking(
        session_id=session_id,
        name=slots.name,
        email=slots.email,
        interview_date=slots.interview_date,
        interview_time=slots.interview_time,
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)
    clear_booking_slots(session_id)
    return booking
