"""Parser registry: routes files to the appropriate language parser.

Routing is by extension for code and by extension *or* exact filename for documents. The
filename route exists because the files that describe how a repository is built and run
frequently have no extension at all (``Makefile``, ``Dockerfile``) or one that says nothing
(``.env.example`` → ``.example``), and an unrouted file is not merely ranked low — it is
never opened by the scan, so nothing about it reaches the index.

A filename match wins over an extension match: a name is the more specific claim.
"""

from __future__ import annotations

from pathlib import Path

from inflorescence.code_indexer.parser.base_parser import BaseParser


class ParserRegistry:
    """Registry that maps file extensions and exact filenames to parsers."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}  # ext -> parser
        self._by_filename: dict[str, BaseParser] = {}  # exact name -> parser

    def register(self, parser: BaseParser) -> None:
        for ext in parser.get_supported_extensions():
            self._parsers[ext] = parser
        for name in parser.get_supported_filenames():
            self._by_filename[name] = parser

    def get_parser(self, file_path: Path | str) -> BaseParser | None:
        path = Path(file_path)
        by_name = self._by_filename.get(path.name)
        if by_name is not None:
            return by_name
        return self._parsers.get(path.suffix)

    def get_all_extensions(self) -> list[str]:
        return list(self._parsers.keys())

    def get_all_filenames(self) -> list[str]:
        return list(self._by_filename.keys())

    def supports(self, file_path: Path | str) -> bool:
        """True when this registry would route *file_path* to a parser.

        The single predicate every stage must ask — scan, watcher, reconcile, cost
        estimate. Asking ``suffix in extensions`` directly is what made a name-routed file
        invisible to some of them and, worse, made the reconcile misread an indexed
        ``Makefile`` missing from disk as "its parser vanished, keep it stale" instead of
        "deleted" (INV-1).
        """
        return self.get_parser(file_path) is not None
