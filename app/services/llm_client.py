"""Provider-agnostic LLM client.

Wraps either OpenAI or Anthropic behind one interface so the RAG and
booking-extraction services don't care which provider is configured.
Selected via LLM_PROVIDER in .env.
"""
import json
import logging
from typing import Protocol

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("palm-mind-rag")


class ChatMessage(Protocol):
    role: str
    content: str


def _messages_as_dicts(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


class LLMClient:
    """Unified chat-completion + structured-JSON-extraction interface."""

    def __init__(self) -> None:
        self.provider = settings.llm_provider
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = settings.openai_model
        elif self.provider == "openai_compatible":
            # Any OpenAI-compatible endpoint (Groq, Google AI Studio/Gemini,
            # etc.) - useful when you want a no-payment-method-required
            # provider for local dev/testing.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.openai_compatible_api_key,
                base_url=settings.openai_compatible_base_url,
            )
            self._model = settings.openai_compatible_model
        elif self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self._model = settings.anthropic_model
        else:  # pragma: no cover - guarded by pydantic Literal
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 700) -> str:
        """Standard chat completion. `messages` is a list of {role, content}."""
        if self.provider in ("openai", "openai_compatible"):
            extra_body = {}
            if self.provider == "openai_compatible":
                # Gemini 2.5/3.5 models "think" by default, consuming part of
                # max_tokens on invisible reasoning before writing the actual
                # answer - which can silently truncate short JSON responses.
                # We don't need reasoning for these tasks, so disable it.
                extra_body["reasoning_effort"] = "none"

            response = self._client.chat.completions.create(
                model=self._model,
                messages=_messages_as_dicts(messages),
                max_tokens=max_tokens,
                temperature=0.3,
                extra_body=extra_body or None,
            )
            return response.choices[0].message.content or ""

        # anthropic
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        non_system = [m for m in messages if m["role"] != "system"]
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=_messages_as_dicts(non_system),
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def extract_json(self, system_prompt: str, user_content: str) -> dict:
        """Ask the model to return strict JSON and parse it.

        Used for slot-filling (booking extraction) where we need a
        structured result rather than free text.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Respond with ONLY valid JSON. No markdown fences, no preamble."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        raw = self.chat(messages, max_tokens=1024)
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(cleaned)
            logger.info("extract_json: parsed result: %r", parsed)
            return parsed
        except json.JSONDecodeError:
            logger.warning("extract_json: failed to parse model output as JSON. Raw output: %r", raw)
            return {}


_client_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton