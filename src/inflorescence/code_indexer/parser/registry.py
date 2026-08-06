"""Parser registry: routes files to the appropriate language parser."""

from __future__ import annotations

from pathlib import Path

from inflorescence.code_indexer.parser.base_parser import BaseParser


class ParserRegistry:
    """Registry that maps file extensions to parsers."""

    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {}  # ext -> parser

    def register(self, parser: BaseParser) -> None:
        for ext in parser.get_supported_extensions():
            self._parsers[ext] = parser

    def get_parser(self, file_path: Path | str) -> BaseParser | None:
        suffix = Path(file_path).suffix
        return self._parsers.get(suffix)

    def get_all_extensions(self) -> list[str]:
        return list(self._parsers.keys())
