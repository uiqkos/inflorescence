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
    top_k: int = 10
    embedding_batch_size: int = 50
    embedding_batch_max_chars: int = 80_000
    max_file_size_bytes: int = 262144
