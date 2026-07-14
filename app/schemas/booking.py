"""Schemas for interview booking capture and storage."""
from datetime import datetime

from pydantic import BaseModel, EmailStr


class BookingSlots(BaseModel):
    """Slots the LLM is asked to extract from the conversation.

    Any field that hasn't been mentioned yet stays None so we know what
    still needs to be asked for.
    """

    name: str | None = None
    email: str | None = None
    interview_date: str | None = None
    interview_time: str | None = None

    def is_complete(self) -> bool:
        return all([self.name, self.email, self.interview_date, self.interview_time])

    def missing_fields(self) -> list[str]:
        fields = {
            "name": self.name,
            "email": self.email,
            "interview_date": self.interview_date,
            "interview_time": self.interview_time,
        }
        return [k for k, v in fields.items() if not v]


class BookingRecord(BaseModel):
    id: str
    session_id: str
    name: str
    email: EmailStr
    interview_date: str
    interview_time: str
    created_at: datetime

    model_config = {"from_attributes": True}
