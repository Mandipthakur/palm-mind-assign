"""Centralized application configuration.

All runtime configuration is sourced from environment variables (see
.env.example). Using pydantic-settings gives us validated, typed config
objects instead of scattering os.environ calls across the codebase.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "palm-mind-rag"
    log_level: str = "INFO"

    # SQL metadata store
    database_url: str = "sqlite:///./app_data.db"

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_memory_ttl_seconds: int = 86400
    chat_memory_max_turns: int = 20

    # Qdrant vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM provider
    llm_provider: Literal["openai", "anthropic", "openai_compatible"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    # Any OpenAI-compatible endpoint (Groq, Google AI Studio/Gemini, etc.) -
    # set llm_provider=openai_compatible to use this instead.
    openai_compatible_api_key: str | None = None
    openai_compatible_base_url: str = "https://api.groq.com/openai/v1"
    openai_compatible_model: str = "llama-3.3-70b-versatile"

    # Chunking
    default_chunk_strategy: Literal["fixed", "recursive"] = "recursive"
    fixed_chunk_size: int = 800
    fixed_chunk_overlap: int = 100

    # Retrieval
    retrieval_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton for the process)."""
    return Settings()