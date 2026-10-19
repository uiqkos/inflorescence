"""File chunking using llama_index splitters with AST-aware code splitting."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from inflorescence.code_indexer.models import DECLARATION_NODE_TYPES, CodeNode, NodeType
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

# Extension -> language stamped on a document chunk. Cosmetic (it labels search results
# and lets a Cypher query filter by it); the chunking itself is identical for all of them.
_DOCUMENT_LANGUAGES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".org": "org",
    ".tex": "latex",
    ".txt": "text",
    ".text": "text",
    # Styles, markup and configuration are documents too (see config.DEFAULT_DOCUMENT_
    # EXTENSIONS): one node per file, body chunked in overlapping windows. Listed here so a
    # chunk of index.css is labelled "css" rather than falling through to "text" — and so
    # chunk_file takes the document branch for them instead of asking a tree-sitter
    # CodeSplitter to build an AST of a stylesheet.
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".styl": "stylus",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".proto": "proto",
    ".sql": "sql",
    ".graphql": "graphql",
    ".gql": "graphql",
    # Shell scripts are documents here even though _EXT_TO_LANGUAGE has a bash grammar for
    # them: the pipeline gives a .sh file a Document node (it owns no symbols this graph
    # models), so its body is windowed as text. An AST split at this entry point would chunk
    # the same script a second, different way — and the thing worth retrieving from a runbook
    # script is usually a comment above the command, which no bash AST node contains.
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
}

# Documents routed by exact filename (config.DEFAULT_DOCUMENT_FILENAMES): same treatment,
# but ``Path.suffix`` cannot find them — a Makefile has none and ``.env.example``'s is
# ``.example``.
_DOCUMENT_LANGUAGES_BY_NAME: dict[str, str] = {
    "Makefile": "make",
    "makefile": "make",
    "GNUmakefile": "make",
    "Dockerfile": "docker",
    "Containerfile": "docker",
    "Justfile": "just",
    "justfile": "just",
    "Procfile": "procfile",
    "Rakefile": "ruby",
    "Brewfile": "ruby",
    ".env.example": "dotenv",
    ".env.sample": "dotenv",
    ".env.template": "dotenv",
    ".env.dist": "dotenv",
}


def _document_language(file_path: str | Path) -> str | None:
    """The chunk language label for a document, or None when the path is not one.

    By exact filename first, then by suffix — the same precedence the parser registry
    applies, so the two never disagree about what a file is.
    """
    path = Path(file_path)
    by_name = _DOCUMENT_LANGUAGES_BY_NAME.get(path.name)
    if by_name:
        return by_name
    suffix = path.suffix.lower()
    if suffix in _DOCUMENT_LANGUAGES:
        return _DOCUMENT_LANGUAGES[suffix]
    return "text" if suffix in _TEXT_EXTENSIONS else None


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


# Chunk kinds whose content is the file's own text, line for line, so a match inside one can
# be pointed at by file line number. The two *_context kinds are generated listings (a module's
# docstring and member table, a class header and its members) — their lines are not file lines.
VERBATIM_CHUNK_KINDS: frozenset[str] = frozenset({
    "function_body", "module_body", "type_body", "document", "fallback_text",
})

# The version of what chunk_repository produces for unchanged content. The coverage marker a
# file carries (``chunks_checksum``) is its content checksum *plus* this, so bumping it marks
# every file's chunks stale once: the next run re-chunks them, reuses every vector whose chunk
# is unchanged, embeds only the chunks the new format adds — and prices that before spending
# (ProjectManager's cost gates read the same stale set). Bump it whenever a chunker change
# alters the chunks of a file whose bytes did not move; nothing else can reach such a file.
#   2 — module-level code and type bodies became verbatim chunks (module_body / type_body).
CHUNK_FORMAT = 2
CHUNK_MARKER_SUFFIX = f"@chunks-v{CHUNK_FORMAT}"

# Node types whose own lines (those outside every member with a chunk of its own) are chunked
# as ``type_body``: fields, enum members, interface members, class attributes.
_TYPE_NODE_TYPES = frozenset({
    NodeType.CLASS, NodeType.INTERFACE, NodeType.ENUM, NodeType.STRUCT, NodeType.TRAIT,
})

# Verbatim window for a function body or a run of module/type lines, in lines. Small enough that
# a match names a narrow span, with an overlap so a statement cut by the boundary is whole in
# one of the two windows.
_BODY_WINDOW_LINES = 80
_BODY_WINDOW_OVERLAP = 12


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

        document_language = _document_language(file_path)
        if document_language is not None:
            # Checked *before* _EXT_TO_LANGUAGE: several document formats (.css, .html,
            # .json, .yaml, .toml, .sh) also have a CodeSplitter grammar, and the pipeline
            # gives them a Document node, whose body is windowed as text. Preferring the AST
            # here would make this entry point chunk the same stylesheet a second, different way.
            return self._chunk_text(
                text, rel,
                language=document_language,
                chunk_size=self.config.document_chunk_size,
                chunk_overlap=self.config.document_chunk_overlap,
            )
        if language:
            return self._chunk_code(text, rel, language)
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
            document = next((n for n in file_nodes if n.node_type == NodeType.DOCUMENT), None)
            if document is not None:
                # A document owns no symbols, so there is nothing to hang per-node chunks
                # off: the file body itself is chunked, with overlap (see _document_chunks).
                chunks.extend(self._document_chunks(document, text))
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
            chunks.extend(self._residual_chunks(file_nodes, lines, language))
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

    def _chunk_text(
        self,
        text: str,
        rel_path: str,
        language: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """Sentence-boundary splitting via SentenceSplitter; falls back to naive split.

        *chunk_size* / *chunk_overlap* override the configured code sizes — documents use
        their own, smaller window (see :meth:`_document_chunks`).
        """
        size = self.config.chunk_size if chunk_size is None else chunk_size
        overlap = self.config.chunk_overlap if chunk_overlap is None else chunk_overlap
        try:
            from llama_index.core.node_parser import SentenceSplitter  # type: ignore[import-untyped]
            from llama_index.core.schema import Document  # type: ignore[import-untyped]

            splitter = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)
            doc = Document(text=text)
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception:
            logger.debug("SentenceSplitter failed for %s, using simple split", rel_path)
            return self._simple_split(text, rel_path, language, size, overlap)

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

    def _simple_split(
        self,
        text: str,
        rel_path: str,
        language: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Chunk]:
        """Last-resort fallback: split by character count with overlap."""
        chunks: list[Chunk] = []
        size = max(1, self.config.chunk_size if chunk_size is None else chunk_size)
        # The step must advance: an overlap >= size (a mis-set knob) would otherwise
        # rewind `pos` forever and hang the indexer on a single file.
        overlap = min(self.config.chunk_overlap if chunk_overlap is None else chunk_overlap, size - 1)
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

    def _document_chunks(self, node: CodeNode, text: str) -> list[Chunk]:
        """Chunk a whole prose file into overlapping windows, attached to its document node.

        Deliberately structure-blind: no heading parsing, no section tree. Markdown headings
        are a typographic convention, not a reliable outline (a README may have one ``#`` and
        2000 words under it, or forty), so splitting on them yields chunks whose size is set
        by the author's formatting habits. Sentence-boundary windows of a fixed size are
        predictable, and the *overlap* is what makes them safe: a passage the window boundary
        cuts in half still appears whole inside its neighbour, so it stays retrievable.
        """
        language = _document_language(node.file_path) or "text"
        chunks = self._chunk_text(
            text,
            node.file_path,
            language=language,
            chunk_size=self.config.document_chunk_size,
            chunk_overlap=self.config.document_chunk_overlap,
        )
        for chunk in chunks:
            # _chunk_text keys chunks by path; bind them to the document node so search
            # results resolve to a real entity (the Code node the chunk HAS_CHUNK hangs off).
            chunk.node_id = node.id
            chunk.chunk_kind = "document"
        return chunks

    def _body_chunks(self, node: CodeNode, lines: list[str], language: str, chunk_kind: str) -> list[Chunk]:
        return self._window_chunks(node, node.line_start, node.line_end, lines, language, chunk_kind)

    def _window_chunks(
        self,
        node: CodeNode,
        first: int,
        last: int,
        lines: list[str],
        language: str,
        chunk_kind: str,
        index: int = 0,
    ) -> list[Chunk]:
        """Verbatim, overlapping windows over file lines *first*..*last* (1-based, inclusive)."""
        start = max(first, 1)
        end = min(last, len(lines))
        chunks: list[Chunk] = []
        current = start
        while current <= end:
            chunk_end = min(current + _BODY_WINDOW_LINES - 1, end)
            content = "\n".join(lines[current - 1 : chunk_end])
            chunks.append(Chunk(content, node.file_path, current, chunk_end, language, index, node.id, chunk_kind))
            if chunk_end == end:
                break
            current = max(chunk_end - _BODY_WINDOW_OVERLAP + 1, current + 1)
            index += 1
        return chunks

    def _residual_chunks(self, file_nodes: list[CodeNode], lines: list[str], language: str) -> list[Chunk]:
        """Verbatim chunks for every line no function or method body chunk already holds.

        A code file used to reach the text index only through its function and method bodies
        plus two generated listings (the module's member table, a class's header). Everything
        else in the file — imports, module-level constants, the rule table a whole subsystem
        is written as, an exported config object, a struct's fields, an enum's members — was
        in no chunk at all, so no search could find a literal written there: three agents of
        one field report (11.09.2026) independently failed to find ``Rule("cache.stale_no_etag",
        …)`` in a module-level tuple that ``grep`` found instantly, and later reports hit the
        same wall with ``export const NO_REFUND = {…}`` and ``var checkers = map[…]{…}``.

        Each line of the file goes to the innermost node whose span contains it. Lines owned
        by a function or method are already in that body's chunks; lines owned by a class,
        interface, enum, struct or trait become ``type_body`` chunks of that node; the rest
        become ``module_body`` chunks of the module. Contiguous runs are windowed like a body,
        and a run with no text on it is dropped — the blank lines between two functions are
        not content.
        """
        total = len(lines)
        module = next((n for n in file_nodes if n.node_type == NodeType.MODULE), None)
        if module is None or total == 0:
            return []
        owners: list[CodeNode | None] = [module] * (total + 1)  # 1-based; index 0 unused
        # Widest first, so a nested node overwrites its container's claim on the lines it spans.
        # Declarations (a const, a type alias) are transparent here: their lines stay with the
        # module or type around them, so a rule table is chunked with its neighbours instead of
        # as hundreds of one-line chunks, and the declaration itself is found by name.
        members = sorted(
            (
                n for n in file_nodes
                if n.node_type != NodeType.MODULE
                and n.node_type not in DECLARATION_NODE_TYPES
                and n.line_end >= n.line_start
            ),
            key=lambda n: (-(n.line_end - n.line_start), n.line_start, n.id),
        )
        for node in members:
            for line in range(max(node.line_start, 1), min(node.line_end, total) + 1):
                owners[line] = node
        chunks: list[Chunk] = []
        run_owner: CodeNode | None = None
        run_start = 0
        per_owner_index: dict[str, int] = {}

        def flush(end: int) -> None:
            if run_owner is None or run_start == 0:
                return
            if not any(lines[i - 1].strip() for i in range(run_start, end + 1)):
                return
            if run_owner.node_type in _TYPE_NODE_TYPES:
                kind = "type_body"
            elif run_owner.node_type == NodeType.MODULE:
                kind = "module_body"
            else:
                return  # a function's or method's lines are its body chunks already
            index = per_owner_index.get(run_owner.id, 0)
            produced = self._window_chunks(run_owner, run_start, end, lines, language, kind, index)
            per_owner_index[run_owner.id] = index + len(produced)
            chunks.extend(produced)

        for line in range(1, total + 1):
            owner = owners[line]
            if owner is not run_owner:
                flush(line - 1)
                run_owner, run_start = owner, line
        flush(total)
        return chunks
