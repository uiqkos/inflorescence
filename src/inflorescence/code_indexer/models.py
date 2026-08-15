"""Data models for the code indexer."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from inflorescence.config import (
    DEFAULT_DOCUMENT_EXTENSIONS,
    DEFAULT_EXCLUDE_PATTERNS,
    GENERATED_FILE_PATTERNS,
    Settings,
)


class NodeType(StrEnum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    DIRECTORY = "directory"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    TRAIT = "trait"
    # A prose file (README, docs/*.md, notes.txt) indexed whole: it has no symbols to
    # contain and no calls to resolve, so it is a leaf of the containment tree and its
    # body is chunked with overlap instead of being parsed. Kept apart from MODULE so a
    # search for modules doesn't return documentation and vice versa.
    DOCUMENT = "document"


class EdgeType(StrEnum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    CONTAINS = "contains"  # module->class, class->method
    IMPLEMENTS = "implements"


class CallResolution(StrEnum):
    """How a CALLS edge target was bound, ordered by decreasing confidence.

    SEMANTIC comes from a compiler-grade SCIP indexer; the rest from the syntactic
    ladder in resolver.py. There is deliberately no "unique short name" level: measured
    against SCIP ground truth it was wrong 61% of the time (docs/calls-resolution.md),
    so an unresolvable call is recorded on the caller node instead of guessing an edge.
    """

    SEMANTIC = "semantic"  # SCIP indexer (go/python/typescript compiler)
    EXACT = "exact"  # target was already a node id
    IMPORT = "import"  # first segment bound via the file's imports
    FIELD_TYPE = "field-type"  # recv.field.Method via the field's declared type (Go)
    SELF = "self"  # self./this./receiver method on the enclosing class
    SAME_MODULE = "same-module"  # bare name defined in the same file (py/js)
    SAME_PACKAGE = "same-package"  # bare name defined in the same directory/package (Go)


class CodeNode(BaseModel):
    id: str  # e.g. "src/foo.py::MyClass.method"
    name: str
    node_type: NodeType
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    summary: str = ""
    summary_input_hash: str = ""
    parent_id: str | None = None
    # Honest call accounting: "0 outgoing CALLS" must never read as "calls nothing"
    # (docs/calls-resolution.md). Calls bound to an import outside the project land in
    # external_calls; calls the ladder could not classify land in unresolved_calls.
    external_calls: list[str] = Field(default_factory=list)
    unresolved_calls: list[str] = Field(default_factory=list)
    # Module nodes only: which mechanism produced this file's CALLS edges
    # ("scip-go", "scip-python", "scip-typescript" or "heuristic").
    calls_provenance: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.node_type.value}) in {self.file_path}"


class Edge(BaseModel):
    source: str  # node id
    target: str  # node id
    edge_type: EdgeType
    # CALLS provenance: a CallResolution value ("" for non-CALLS edges and legacy data).
    resolution: str = ""
    # The call as written at the call site ("s.content.ForCheck") — kept for debugging
    # and for upgrading heuristic edges when a semantic pass later covers the file.
    callee_text: str = ""


class ExternalRef(BaseModel):
    """A module or class the project depends on but does not contain.

    ``import "net/http"`` and ``class Config(BaseModel)`` name real things that simply live
    outside the indexed tree. Before this existed, the edges to them were dropped at write
    time and the dependency vanished — measured at 29% of all edges on a Go service. They are
    stored under their own ``:External`` label rather than as ``:Code``, so every existing
    query, the entity listing, the dashboard counts, search and the summarizer keep seeing
    only the project's own code.
    """

    id: str  # "ext:module:net/http" — never collides with a Code id (file path / file::symbol)
    name: str  # the specifier as written: "net/http", "@electron-forge/maker-zip", "BaseModel"
    root: str  # the package it belongs to: "net/http", "@electron-forge/maker-zip", "BaseModel"
    kind: str  # "module" (an import) | "class" (a base class)
    language: str = ""


class FileFacts(BaseModel):
    """Per-file name-binding facts the CALLS resolver ladder consumes.

    ``imports`` maps a local binding to a language-native import target:
      python: ``np -> numpy``, ``g -> a.b.f`` (from-import; relative dots preserved)
      go:     alias -> import path (``content -> github.com/x/y/internal/content``)
      js/ts:  local -> ``./mod:orig`` for named/default imports, ``./mod:*`` for
              namespace imports and CJS ``require()``
    ``local_names`` lists identifiers (params, locals) declared inside each function
    node that shadow module-level names — a bare call to a shadowed name must NOT
    resolve to the module-level function (measured failure mode on cerberus).
    """

    file_path: str
    language: str = ""
    imports: dict[str, str] = Field(default_factory=dict)
    struct_fields: dict[str, dict[str, str]] = Field(default_factory=dict)  # Go: Type -> {field: type}
    receiver_names: dict[str, str] = Field(default_factory=dict)  # Go: 'Type.Method' -> receiver var
    local_names: dict[str, list[str]] = Field(default_factory=dict)  # node id -> shadowing names
    go_module: str = ""  # module path from the owning go.mod
    go_module_root: str = ""  # root-relative dir of that go.mod ("" = project root)


_CHECKSUM_BLOCK_SIZE = 1 << 20  # 1 MiB — bound peak memory instead of reading the whole file


class FileChecksum(BaseModel):
    file_path: str
    md5: str

    @staticmethod
    def compute(path: Path, root: Path | None = None) -> FileChecksum:
        digest = hashlib.md5()  # noqa: S324 — not used for security
        with path.open("rb") as f:
            for block in iter(lambda: f.read(_CHECKSUM_BLOCK_SIZE), b""):
                digest.update(block)
        rel = str(path.relative_to(root)) if root else str(path)
        return FileChecksum(file_path=rel, md5=digest.hexdigest())

    @staticmethod
    def try_compute(path: Path, root: Path | None = None) -> FileChecksum | None:
        """Like :meth:`compute`, but return ``None`` when the file can't be read.

        A file can vanish or turn unreadable between the directory scan and the checksum
        (a ``git checkout`` racing a watcher flush). The raising :meth:`compute` used at
        every call site would then abort the whole run; ``None`` lets the caller skip that
        one file, leaving its indexed data stale rather than crashing (INV-3, audit finding:
        OSError in checksum crashes the run).
        """
        try:
            return FileChecksum.compute(path, root)
        except OSError:
            return None


class ProjectGraph(BaseModel):
    root_path: str
    nodes: list[CodeNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    checksums: list[FileChecksum] = Field(default_factory=list)
    git_head: str = ""  # commit SHA at index time

    # Indexes -- rebuilt on load, not serialized
    _node_index: dict[str, CodeNode] = {}
    _file_index: dict[str, list[CodeNode]] = {}
    _adj: dict[str, list[Edge]] = {}
    _rev_adj: dict[str, list[Edge]] = {}

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _build_indexes(self) -> ProjectGraph:
        self._node_index = {n.id: n for n in self.nodes}
        self._file_index = {}
        for n in self.nodes:
            self._file_index.setdefault(n.file_path, []).append(n)
        self._adj = {}
        self._rev_adj = {}
        for e in self.edges:
            self._adj.setdefault(e.source, []).append(e)
            self._rev_adj.setdefault(e.target, []).append(e)
        return self

    def get_node(self, node_id: str) -> CodeNode | None:
        # Exact match
        if node_id in self._node_index:
            return self._node_index[node_id]
        # Suffix match: "provider.py::MyClass" matches "src/flask/json/provider.py::MyClass"
        normalized = node_id.replace("\\", "/").lstrip("./")
        for stored_id, node in self._node_index.items():
            if stored_id.endswith(normalized) or normalized.endswith(stored_id):
                return node
        return None

    def get_file_nodes(self, file_path: str) -> list[CodeNode]:
        # Exact match first
        if file_path in self._file_index:
            return self._file_index[file_path]
        # Suffix match: "provider.py" matches "src/flask/json/provider.py"
        normalized = file_path.replace("\\", "/").lstrip("./")
        for stored_path, nodes in self._file_index.items():
            if stored_path.endswith(normalized) or normalized.endswith(stored_path):
                return nodes
        # Basename match: "provider.py" matches any file named provider.py
        basename = normalized.rsplit("/", 1)[-1]
        for stored_path, nodes in self._file_index.items():
            if stored_path.rsplit("/", 1)[-1] == basename:
                return nodes
        return []

    def get_outgoing(self, node_id: str) -> list[Edge]:
        return self._adj.get(node_id, [])

    def get_incoming(self, node_id: str) -> list[Edge]:
        return self._rev_adj.get(node_id, [])

    def get_checksum(self, file_path: str) -> str | None:
        for c in self.checksums:
            if c.file_path == file_path:
                return c.md5
        return None


class IndexerConfig(BaseModel):
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    batch_size: int = 10
    respect_gitignore: bool = True
    max_file_size_bytes: int = 262144
    skip_generated: bool = True
    generated_patterns: list[str] = Field(default_factory=lambda: list(GENERATED_FILE_PATTERNS))
    use_llm_summaries: bool = True
    use_llm_fallback_parser: bool = True
    # Prose documents (.md/.txt/...) are scanned, stored as one node per file, and chunked
    # with overlap. Turning this off removes their extensions from the scan entirely, so
    # nothing about them is read, summarized, or embedded.
    index_documents: bool = True
    document_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_DOCUMENT_EXTENSIONS))
    enabled_languages: list[str] = Field(
        default_factory=lambda: ["python", "javascript", "typescript", "go", "rust"]
    )
    # Semantic CALLS enrichment via SCIP indexers (scip-go / scip-python /
    # scip-typescript). Preflight-gated: a missing binary, failing build, or timeout
    # silently degrades that language to the heuristic ladder — never fails the run.
    use_scip_semantic: bool = True
    scip_commands: dict[str, str] = Field(default_factory=dict)  # language -> command override
    scip_env: dict[str, str] = Field(default_factory=dict)  # extra env, e.g. {"GOFLAGS": "-tags=integration"}
    scip_timeout_seconds: int = 600
    scip_runner: str = "auto"  # auto | native | docker
    scip_images: dict[str, str] = Field(default_factory=dict)  # language -> container image override

    @classmethod
    def from_settings(cls, settings: Settings) -> IndexerConfig:
        """Build an indexer config from application settings.

        Every caller must go through this. There are three construction sites (the MCP server,
        the CLI index command, the CLI preview), and threading a new setting through two of
        them and forgetting the third is how `use_scip_semantic` and its timeout ended up
        unreachable from the environment while the README advertised them as a feature.
        """
        return cls(
            include_patterns=settings.include_patterns,
            exclude_patterns=settings.exclude_patterns,
            batch_size=settings.batch_size,
            respect_gitignore=settings.respect_gitignore,
            max_file_size_bytes=settings.max_file_size_bytes,
            skip_generated=settings.skip_generated,
            use_llm_summaries=settings.use_llm_summaries,
            index_documents=settings.index_documents,
            document_extensions=settings.document_extensions,
            use_scip_semantic=settings.use_scip_semantic,
            scip_timeout_seconds=settings.scip_timeout_seconds,
            scip_runner=settings.scip_runner,
            scip_images=settings.scip_images,
        )
