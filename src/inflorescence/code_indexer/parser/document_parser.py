"""Whole-file indexing for prose documents (.md, .rst, .txt, ...).

A document has no structure this graph can model: a heading is not a symbol, nothing
CALLS a paragraph, and an ``##`` section boundary is a typographic choice, not a
containment relation. Parsing one into pseudo-symbols would produce nodes no edge ever
points at and a containment tree that says nothing true. So a document becomes exactly
**one** node — the file — and retrieval happens at the chunk level: ``rag/chunker.py``
splits the body into overlapping sentence windows, which is what makes a passage in the
middle of a long README findable without inventing a symbol tree above it.

The parser therefore reads the file only to know how many lines it has; the text itself
is read again by the summarizer (``node_source``) and the chunker, both of which work
from ``line_start``/``line_end``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from inflorescence.code_indexer.models import CodeNode, Edge, NodeType
from inflorescence.code_indexer.parser.base_parser import BaseParser
from inflorescence.config import DEFAULT_DOCUMENT_EXTENSIONS

logger = logging.getLogger(__name__)


def _normalize_extension(extension: str) -> str:
    """``md`` and ``.MD`` both mean ``.md`` — the scan compares raw ``Path.suffix``."""
    normalized = extension.strip().lower()
    return normalized if normalized.startswith(".") else f".{normalized}"


class DocumentParser(BaseParser):
    """Emits one node per prose file, with no structure extraction."""

    def __init__(self, extensions: Sequence[str] | None = None) -> None:
        source = DEFAULT_DOCUMENT_EXTENSIONS if extensions is None else extensions
        self._extensions = sorted({_normalize_extension(e) for e in source if e.strip()})

    def get_supported_extensions(self) -> list[str]:
        return list(self._extensions)

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        rel = str(file_path.relative_to(root_path))
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            # Same contract as every other parser: a read failure yields no nodes, and the
            # reconcile reads "scanned but produced nothing" as failure — keeping the file's
            # stored data stale rather than deleting it (INV-1/INV-3).
            logger.debug("Cannot read %s (%s), skipping", file_path, exc)
            return [], []

        lines = text.splitlines()
        node = CodeNode(
            id=rel,
            name=file_path.name,  # with the extension: README.md and README.rst are two docs
            node_type=NodeType.DOCUMENT,
            file_path=rel,
            # An empty file still spans line 1..1, so node_source and the chunker stay in
            # range; they simply find nothing to summarize or chunk.
            line_start=1,
            line_end=max(1, len(lines)),
        )
        return [node], []
