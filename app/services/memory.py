"""Redis-backed conversational memory.

Each session's turn history and in-progress booking slots are stored as
JSON in Redis under session-scoped keys, with a TTL so idle conversations
naturally expire instead of growing Redis usage forever.
"""
import json
from functools import lru_cache

import redis

from app.config import get_settings
from app.schemas.booking import BookingSlots

settings = get_settings()

_HISTORY_KEY = "chat:history:{session_id}"
_BOOKING_KEY = "chat:booking:{session_id}"


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def get_history(session_id: str) -> list[dict[str, str]]:
    """Return the stored [{role, content}, ...] turns for a session."""
    raw = get_redis().get(_HISTORY_KEY.format(session_id=session_id))
    return json.loads(raw) if raw else []


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append one turn and trim history to the configured max length."""
    r = get_redis()
    key = _HISTORY_KEY.format(session_id=session_id)
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    history = history[-settings.chat_memory_max_turns :]
    r.set(key, json.dumps(history), ex=settings.chat_memory_ttl_seconds)


def get_booking_slots(session_id: str) -> BookingSlots | None:
    """Return the in-progress booking slots for a session, or None if no
    booking flow has been started yet for this session.
    """
    raw = get_redis().get(_BOOKING_KEY.format(session_id=session_id))
    return BookingSlots.model_validate_json(raw) if raw else None


def set_booking_slots(session_id: str, slots: BookingSlots) -> None:
    get_redis().set(
        _BOOKING_KEY.format(session_id=session_id),
        slots.model_dump_json(),
        ex=settings.chat_memory_ttl_seconds,
    )


def clear_booking_slots(session_id: str) -> None:
    get_redis().delete(_BOOKING_KEY.format(session_id=session_id))
