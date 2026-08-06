"""File chunking using llama_index splitters with AST-aware code splitting."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from inflorescence.code_indexer.models import CodeNode, NodeType
from inflorescence.rag.config import RAGConfig

logger = logging.getLogger(__name__)

# Extension -> llama_index CodeSplitter language identifier.
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".scala": "scala",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".pl": "perl",
    ".pm": "perl",
    ".dart": "dart",
    ".jl": "julia",
    ".proto": "proto",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".zig": "zig",
}

_TEXT_EXTENSIONS: frozenset[str] = frozenset({".md", ".rst", ".txt", ".adoc", ".tex"})

_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".zst",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac", ".ogg",
    ".pyc", ".pyo", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".bin", ".dat", ".iso", ".img",
})


@dataclass
class Chunk:
    """A single embeddable chunk linked to a canonical Code node."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_index: int
    node_id: str = ""
    chunk_kind: str = "fallback_text"


@dataclass
class Chunker:
    """Splits repository files into embeddable chunks.

    Uses ``CodeSplitter`` (tree-sitter AST) for recognised code languages and
    ``SentenceSplitter`` for prose / config files.
    """

    config: RAGConfig = field(default_factory=RAGConfig)

    def chunk_file(self, file_path: Path, root: Path) -> list[Chunk]:
        """Chunk a single file into pieces relative to *root*."""
        if self._is_binary(file_path):
            return []

        if self._too_large(file_path):
            logger.debug("Skipping large file %s", file_path)
            return []

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.debug("Cannot read %s", file_path)
            return []

        if not text.strip():
            return []

        rel = str(file_path.relative_to(root))
        ext = file_path.suffix.lower()
        language = _EXT_TO_LANGUAGE.get(ext, "")

        if language:
            return self._chunk_code(text, rel, language)
        if ext in _TEXT_EXTENSIONS:
            return self._chunk_text(text, rel)
        return self._chunk_text(text, rel)

    def chunk_repository(
        self,
        root: Path,
        nodes: Sequence[CodeNode],
        unreadable_out: set[str] | None = None,
    ) -> list[Chunk]:
        """Chunk the files owning *nodes*, relative to *root*.

        Files that vanished or turned unreadable since the scan (a ``git checkout`` racing
        a flush) are collected into *unreadable_out* (when provided) and skipped instead of
        raising — one bad file must not abort the whole run. The caller keeps such files'
        stored chunks stale rather than pruning them (see ``RAGIndexer.index_code``), so a
        transient read failure never empties good data (INV-1/INV-3, audit finding: OSError in
        chunk_repository crashes the run). A file that is merely binary/oversize is a
        legitimate "no chunks", not unreadable, so it is not reported.
        """
        chunks: list[Chunk] = []
        by_file: dict[str, list[CodeNode]] = {}
        for node in nodes:
            if node.file_path:
                by_file.setdefault(node.file_path, []).append(node)

        for rel_path, file_nodes in sorted(by_file.items()):
            full_path = root / rel_path
            if not full_path.exists():
                # Existed at parse time, gone now — transient; keep prior chunks stale.
                if unreadable_out is not None:
                    unreadable_out.add(rel_path)
                continue
            try:
                if not full_path.is_file() or self._is_binary_strict(full_path) or self._too_large_strict(full_path):
                    continue
            except OSError:
                # Readable at parse time, unreadable now (dropped permissions, I/O error).
                # Binary-vs-unreadable matters here: "binary" would stamp the file covered
                # and prune its stored chunks; "unreadable" keeps them stale for retry.
                logger.debug("Cannot probe %s during chunking, keeping prior chunks stale", full_path)
                if unreadable_out is not None:
                    unreadable_out.add(rel_path)
                continue
            try:
                text = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                logger.debug("Cannot read %s during chunking, keeping prior chunks stale", full_path)
                if unreadable_out is not None:
                    unreadable_out.add(rel_path)
                continue
            lines = text.splitlines()
            language = _EXT_TO_LANGUAGE.get(full_path.suffix.lower(), "text")
            module = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
            if module is not None:
                chunks.append(self._module_context_chunk(module, file_nodes, lines, language))
            for node in sorted(file_nodes, key=lambda n: (n.line_start, n.line_end, n.id)):
                if node.node_type == NodeType.CLASS:
                    chunks.append(self._class_context_chunk(node, file_nodes, lines, language))
                elif node.node_type in {NodeType.FUNCTION, NodeType.METHOD}:
                    chunks.extend(self._body_chunks(node, lines, language, "function_body"))
        return [chunk for chunk in chunks if chunk.content.strip()]

    def collect_files(self, root: Path, exclude_patterns: Sequence[str]) -> list[Path]:
        """Collect all text files under *root*, respecting exclude patterns."""
        logger.info("Scanning files in %s...", root)
        files: list[Path] = []
        skipped_pattern = 0
        skipped_binary = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if self._should_skip(path, exclude_patterns):
                skipped_pattern += 1
                continue
            if self._is_binary(path):
                skipped_binary += 1
                continue
            files.append(path)
        logger.info(
            "Collected %d files (skipped: %d by pattern, %d binary)",
            len(files), skipped_pattern, skipped_binary,
        )
        return files

    def _too_large(self, path: Path) -> bool:
        try:
            return self._too_large_strict(path)
        except OSError:
            return True

    def _too_large_strict(self, path: Path) -> bool:
        return path.stat().st_size > self.config.max_file_size_bytes

    def _is_binary(self, path: Path) -> bool:
        """Heuristic: known binary extension or contains null bytes; unreadable counts as binary."""
        try:
            return self._is_binary_strict(path)
        except OSError:
            return True

    def _is_binary_strict(self, path: Path) -> bool:
        """Like ``_is_binary`` but lets OSError propagate, so callers that must
        distinguish "binary" (legitimate no-chunks) from "unreadable" (keep stale) can."""
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return True
        with open(path, "rb") as f:
            data = f.read(8192)
        return b"\x00" in data

    def _should_skip(self, path: Path, exclude_patterns: Sequence[str]) -> bool:
        s = str(path)
        for pat in exclude_patterns:
            if pat.startswith("*"):
                if s.endswith(pat[1:]):
                    return True
            elif pat in s.split("/") or pat in s.split("\\"):
                return True
        return False

    def _chunk_code(self, text: str, rel_path: str, language: str) -> list[Chunk]:
        """AST-aware splitting via CodeSplitter; falls back to text split on error."""
        try:
            from llama_index.core.node_parser import CodeSplitter  # type: ignore[import-untyped]
            from llama_index.core.schema import Document  # type: ignore[import-untyped]

            splitter = CodeSplitter(
                language=language,
                chunk_lines=self.config.chunk_size // 40,
                chunk_lines_overlap=self.config.chunk_overlap // 40,
                max_chars=self.config.chunk_size * 2,
            )
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
        except ImportError:
            # A missing grammar package downgrades every code file in the index to prose
            # chunking — the core of semantic search, silently gone. This was invisible at
            # DEBUG for the whole life of the feature, so it warns once per file and names
            # the cause rather than reporting a generic "failed".
            logger.warning(
                "AST chunking unavailable for %s (%s) — tree-sitter grammar not installed; "
                "falling back to prose chunking, which degrades semantic search quality. "
                "Install the 'tree-sitter-language-pack' dependency to restore it.",
                rel_path,
                language,
            )
            return self._chunk_text(text, rel_path, language=language)
        except Exception:
            # A parse failure on one file is expected and local (syntax errors, an unsupported
            # dialect), unlike a missing grammar which affects every file.
            logger.info(
                "AST chunking failed for %s (%s), falling back to prose chunking", rel_path, language
            )
            return self._chunk_text(text, rel_path, language=language)

        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            content: str = node.get_content()
            start_line = self._find_line_number(text, content)
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                chunk_index=idx,
                node_id=rel_path,
            ))
        return chunks

    def _chunk_text(self, text: str, rel_path: str, language: str = "") -> list[Chunk]:
        """Sentence-boundary splitting via SentenceSplitter; falls back to naive split."""
        try:
            from llama_index.core.node_parser import SentenceSplitter  # type: ignore[import-untyped]
            from llama_index.core.schema import Document  # type: ignore[import-untyped]

            splitter = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception:
            logger.debug("SentenceSplitter failed for %s, using simple split", rel_path)
            return self._simple_split(text, rel_path, language)

        chunks: list[Chunk] = []
        for idx, node in enumerate(nodes):
            content: str = node.get_content()
            start_line = self._find_line_number(text, content)
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language or "text",
                chunk_index=idx,
                node_id=rel_path,
            ))
        return chunks

    def _simple_split(self, text: str, rel_path: str, language: str = "") -> list[Chunk]:
        """Last-resort fallback: split by character count with overlap."""
        chunks: list[Chunk] = []
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        idx = 0
        pos = 0
        while pos < len(text):
            end = min(pos + size, len(text))
            content = text[pos:end]
            start_line = text[:pos].count("\n") + 1
            end_line = start_line + content.count("\n")
            chunks.append(Chunk(
                content=content,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language or "text",
                chunk_index=idx,
                node_id=rel_path,
            ))
            idx += 1
            pos += size - overlap
            if pos >= len(text):
                break
        return chunks

    def _find_line_number(self, full_text: str, chunk_text: str) -> int:
        """Find the 1-based line number where *chunk_text* starts in *full_text*."""
        idx = full_text.find(chunk_text[:200])
        if idx == -1:
            return 1
        return full_text[:idx].count("\n") + 1

    def _module_context_chunk(self, module: CodeNode, nodes: list[CodeNode], lines: list[str], language: str) -> Chunk:
        symbols = [n for n in nodes if n.parent_id == module.id and n.id != module.id]
        content_lines = [f"module {module.file_path}", (module.docstring or "").strip()]
        content_lines.extend(
            f"{symbol.node_type.value} {symbol.name} lines {symbol.line_start}-{symbol.line_end}"
            for symbol in symbols
        )
        return Chunk(
            "\n".join(line for line in content_lines if line),
            module.file_path,
            1,
            max(1, min(len(lines), 80)),
            language,
            0,
            module.id,
            "module_context",
        )

    def _class_context_chunk(self, node: CodeNode, nodes: list[CodeNode], lines: list[str], language: str) -> Chunk:
        children = [n for n in nodes if n.parent_id == node.id]
        header = lines[node.line_start - 1].strip() if 0 < node.line_start <= len(lines) else node.signature or node.name
        content_lines = [header, (node.docstring or "").strip()]
        content_lines.extend(f"{child.node_type.value} {child.name} lines {child.line_start}-{child.line_end}" for child in children)
        return Chunk(
            "\n".join(line for line in content_lines if line),
            node.file_path,
            node.line_start,
            max(node.line_start, min(node.line_end, node.line_start + 80)),
            language,
            0,
            node.id,
            "class_context",
        )

    def _body_chunks(self, node: CodeNode, lines: list[str], language: str, chunk_kind: str) -> list[Chunk]:
        start = max(node.line_start, 1)
        end = min(node.line_end, len(lines))
        window = 80
        overlap = 12
        chunks: list[Chunk] = []
        index = 0
        current = start
        while current <= end:
            chunk_end = min(current + window - 1, end)
            content = "\n".join(lines[current - 1 : chunk_end])
            chunks.append(Chunk(content, node.file_path, current, chunk_end, language, index, node.id, chunk_kind))
            if chunk_end == end:
                break
            current = max(chunk_end - overlap + 1, current + 1)
            index += 1
        return chunks
