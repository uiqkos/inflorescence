"""RAG configuration settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Tuning knobs for the RAG chunking and embedding pipeline."""

    code_embedding_model: str = "mistralai/codestral-embed-2505"
    summary_embedding_model: str = "openai/text-embedding-3-small"
    chunk_size: int = 3000
    chunk_overlap: int = 1000
    # Documents (.md/.txt/...) get their own, smaller window: their chunks are the *only*
    # representation of the file's content (there are no per-symbol chunks to fall back
    # on), so precision matters more than context. Sizes are in tokens — llama_index's
    # SentenceSplitter counts tokens, not characters.
    document_chunk_size: int = 800
    document_chunk_overlap: int = 200
    top_k: int = 10
    embedding_batch_size: int = 50
    embedding_batch_max_chars: int = 80_000
    max_file_size_bytes: int = 262144
