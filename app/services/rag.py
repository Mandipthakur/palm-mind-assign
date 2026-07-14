"""Custom Retrieval-Augmented Generation pipeline.

Deliberately hand-rolled (no RetrievalQAChain / langchain chains) so the
retrieval, prompt assembly, and history injection steps are all explicit
and inspectable:

    query -> embed -> vector search -> build grounded prompt
          -> inject chat history -> call LLM -> return answer + sources
"""
from dataclasses import dataclass

from app.config import get_settings
from app.services.embeddings import embed_query
from app.services.llm_client import get_llm_client
from app.services.memory import append_turn, get_history
from app.services.vector_store import VectorSearchResult, search

settings = get_settings()

_SYSTEM_PROMPT = """
You are a helpful assistant answering questions using only the provided context
excerpts from the user's uploaded documents. Follow these rules:

- Answer using the context below. If the context doesn't contain the answer,
  say you don't have enough information in the uploaded documents, rather
  than guessing.
- Be concise and direct.
- You may use the conversation history for follow-up questions (e.g. "what
  about the second one?"), but ground factual claims in the context.
"""


@dataclass(frozen=True)
class RagResult:
    answer: str
    sources: list[VectorSearchResult]


def _build_context_block(chunks: list[VectorSearchResult]) -> str:
    if not chunks:
        return "(no relevant context found)"
    parts = [
        f"[Source {i + 1} | document={c.document_id} | chunk={c.chunk_index}]\n{c.text}"
        for i, c in enumerate(chunks)
    ]
    return "\n\n".join(parts)


def answer_query(session_id: str, user_message: str) -> RagResult:
    """Run one turn of the RAG pipeline and update session memory."""
    query_vector = embed_query(user_message)
    retrieved = search(query_vector, top_k=settings.retrieval_top_k)

    history = get_history(session_id)
    context_block = _build_context_block(retrieved)

    messages = [{"role": "system", "content": f"{_SYSTEM_PROMPT}\n\nContext:\n{context_block}"}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    llm = get_llm_client()
    answer = llm.chat(messages)

    append_turn(session_id, "user", user_message)
    append_turn(session_id, "assistant", answer)

    return RagResult(answer=answer, sources=retrieved)
