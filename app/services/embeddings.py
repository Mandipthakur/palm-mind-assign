"""Embedding generation using a local sentence-transformers model.

Kept local (no external API call) so the ingestion pipeline has no hard
dependency on an LLM provider API key, and so embedding cost/latency is
predictable. Swap EMBEDDING_MODEL_NAME in .env for a different model if
needed - dimension must be updated to match.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

settings = get_settings()


@lru_cache
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input string."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string (e.g. a user chat message)."""
    return embed_texts([text])[0]
