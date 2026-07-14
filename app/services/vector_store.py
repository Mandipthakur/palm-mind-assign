"""Thin wrapper around the Qdrant client.

Keeping all Qdrant-specific calls behind this module means the rest of the
app (routers, RAG service) never imports qdrant_client directly. Swapping
to Weaviate/Milvus later only touches this file.
"""
from dataclasses import dataclass
from functools import lru_cache
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    score: float


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def ensure_collection() -> None:
    """Create the collection at startup if it doesn't already exist."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qmodels.VectorParams(
            size=settings.embedding_dimension,
            distance=qmodels.Distance.COSINE,
        ),
    )


def upsert_chunks(
    document_id: str,
    filename: str,
    chunk_texts: list[str],
    chunk_vectors: list[list[float]],
) -> list[str]:
    """Store chunk vectors + payload (text, document_id, filename) in Qdrant.

    Returns the generated chunk ids in the same order as the inputs.
    """
    client = get_client()
    chunk_ids = [str(uuid4()) for _ in chunk_texts]

    points = [
        qmodels.PointStruct(
            id=chunk_ids[i],
            vector=chunk_vectors[i],
            payload={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk_texts[i],
            },
        )
        for i in range(len(chunk_texts))
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return chunk_ids


def search(query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
    """Return the top_k most similar chunks to the given query vector."""
    client = get_client()
    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        VectorSearchResult(
            chunk_id=str(hit.id),
            document_id=hit.payload["document_id"],
            text=hit.payload["text"],
            chunk_index=hit.payload["chunk_index"],
            score=hit.score,
        )
        for hit in hits
    ]
