"""Two selectable text-chunking strategies.

1. Fixed-size chunking: splits on a raw character count with configurable
   overlap. Simple, predictable, cheap - a reasonable default for uniform
   text like logs or plain reports.

2. Recursive chunking: tries to split on the "largest" natural boundary
   first (paragraphs), and only falls back to smaller boundaries
   (sentences, then words) for pieces that are still too large. This
   keeps semantically related sentences together, which tends to improve
   retrieval quality for prose-heavy documents.
"""
import re
from dataclasses import dataclass

from app.schemas.ingestion import ChunkStrategy


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def _fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    step = chunk_size - overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == text_len:
            break
        start += step

    return chunks


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _recursive_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Paragraph doesn't fit as-is alongside current buffer; flush and
        # decide whether the paragraph itself needs sentence-level splitting.
        flush()

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        # Paragraph itself too big -> split into sentences and pack greedily.
        for sentence in _split_sentences(paragraph):
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                flush()
                # Extremely long single sentence: hard-wrap as a last resort.
                if len(sentence) > chunk_size:
                    chunks.extend(_fixed_size_chunks(sentence, chunk_size, overlap))
                else:
                    current = sentence

    flush()

    # Apply a light overlap by prepending the tail of the previous chunk,
    # so retrieval doesn't lose context at chunk boundaries.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail} {curr}".strip())
        return overlapped

    return chunks


def chunk_text(
    text: str,
    strategy: ChunkStrategy,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split text into chunks using the requested strategy."""
    if strategy == ChunkStrategy.FIXED:
        raw_chunks = _fixed_size_chunks(text, chunk_size, overlap)
    elif strategy == ChunkStrategy.RECURSIVE:
        raw_chunks = _recursive_chunks(text, chunk_size, overlap)
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unknown chunk strategy: {strategy}")

    return [Chunk(index=i, text=t) for i, t in enumerate(raw_chunks)]
