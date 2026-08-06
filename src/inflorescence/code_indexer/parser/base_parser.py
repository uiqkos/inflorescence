"""Abstract base class for language parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from inflorescence.code_indexer.models import CodeNode, Edge, FileFacts


class BaseParser(ABC):
    """Base class that all language parsers must implement."""

    @abstractmethod
    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        """Parse a source file and return extracted nodes and edges."""

    def parse_file_with_facts(
        self, file_path: Path, root_path: Path
    ) -> tuple[list[CodeNode], list[Edge], FileFacts | None]:
        """Parse a file returning nodes, edges and name-binding facts for the resolver.

        Parsers that don't extract facts inherit this default; their CALLS targets then
        only resolve at the ``exact``/``same-module`` rungs of the ladder.
        """
        nodes, edges = self.parse_file(file_path, root_path)
        return nodes, edges, None

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Return list of file extensions this parser supports (e.g. ['.py'])."""
