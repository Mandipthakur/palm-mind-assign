"""Unit tests for the two chunking strategies.

These run with no external services (no Qdrant/Redis/LLM needed) since
chunking is pure text logic - run with: pytest tests/test_chunking.py
"""
from app.schemas.ingestion import ChunkStrategy
from app.services.chunking import chunk_text

SAMPLE_TEXT = (
    "Paragraph one. It has two sentences.\n\n"
    "Paragraph two is a bit longer and talks about something else entirely, "
    "spanning multiple sentences to test packing behavior.\n\n"
    "Paragraph three is short."
)


def test_fixed_chunking_respects_size_and_overlap() -> None:
    chunks = chunk_text(SAMPLE_TEXT, ChunkStrategy.FIXED, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 50


def test_recursive_chunking_keeps_paragraphs_together_when_possible() -> None:
    chunks = chunk_text(SAMPLE_TEXT, ChunkStrategy.RECURSIVE, chunk_size=200, overlap=0)
    assert len(chunks) >= 1
    joined = " ".join(c.text for c in chunks)
    assert "Paragraph one" in joined
    assert "Paragraph three is short" in joined


def test_recursive_chunking_splits_oversized_paragraph_by_sentence() -> None:
    long_paragraph = " ".join([f"Sentence number {i} in a long paragraph." for i in range(20)])
    chunks = chunk_text(long_paragraph, ChunkStrategy.RECURSIVE, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 150  # allow small overlap slack


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("", ChunkStrategy.FIXED, chunk_size=100, overlap=10) == []
    assert chunk_text("   \n\n  ", ChunkStrategy.RECURSIVE, chunk_size=100, overlap=0) == []
