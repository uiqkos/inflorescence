"""Documents are indexed whole: one node per file, body chunked with overlap.

"Document" started as prose (.md/.txt/...) and now also covers styles, markup and
configuration — the files a frontend question is half about. The bug that forced the
widening: a real defect lived in ``@media (prefers-reduced-motion)`` inside ``index.css``
and was not merely ranked low but *absent*, because no scan ever opened the file.

The contract these guard:
  * such a file is *scanned* at all — outside the registry's extensions, no scan sees it,
    no chunker touches it, and it is invisible to every search tool;
  * it produces exactly ONE node — no invented heading/section/selector symbols, and no
    attempt to build an AST of a stylesheet;
  * its content reaches the index as overlapping chunks, so a passage cut by a window
    boundary survives whole in the neighbouring window;
  * the file-level bookkeeping (checksums, chunk-coverage) addresses it, so an unchanged
    document costs nothing on re-index;
  * the bulk that comes with admitting data formats stays out: lock files by name,
    oversized files into ``skipped_large_files``, ignored trees not at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import CodeNode, EdgeType, IndexerConfig, NodeType
from inflorescence.code_indexer.parser.document_parser import DocumentParser
from inflorescence.config import DEFAULT_DOCUMENT_EXTENSIONS, LOCKFILE_NAMES, Settings
from inflorescence.db.repository import _NODE_TYPE_LABEL_BY_VALUE
from inflorescence.rag.chunker import Chunker
from inflorescence.rag.config import RAGConfig

_NO_LLM = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)


class _RecordingRepository:
    """Minimal in-memory stand-in for GraphRepository — records what a build writes."""

    def __init__(self) -> None:
        self.upserted_nodes: list[CodeNode] = []
        self.upserted_edges: list = []
        self.checksums: set[str] = set()
        self.summaries_stored: list[dict] = []

    async def upsert_nodes(self, project: str, nodes: list) -> int:
        self.upserted_nodes = list(nodes)
        return len(nodes)

    async def upsert_edges(self, project: str, edges: list) -> int:
        self.upserted_edges = list(edges)
        return len(edges)

    async def upsert_external_nodes(self, project: str, refs: list) -> int:
        return len(refs)

    async def delete_orphan_external_nodes(self, project: str) -> int:
        return 0

    async def store_summaries(self, project: str, items: list) -> int:
        self.summaries_stored.extend(items)
        return len(items)

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        return None

    async def update_checksums(self, project: str, items: list) -> int:
        self.checksums.update(item["node_id"] for item in items)
        return len(items)

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        return {}

    async def get_stored_node_index(self, project: str) -> list[tuple[str, str, str]]:
        return []

    async def get_code_edges(self, project: str) -> list[tuple[str, str, str]]:
        return []

    async def delete_nodes_by_ids(self, project: str, ids: list) -> int:
        return len(ids)

    async def delete_edges_by_keys(self, project: str, rows: list) -> int:
        return len(rows)

    async def get_checksums(self, project: str) -> dict[str, str]:
        return {}


_GUIDE = "# Guide\n\n" + "\n\n".join(
    f"## Section {i}\n\nParagraph {i} explains the widget pipeline in detail. " * 12
    for i in range(12)
)


def test_document_parser_emits_one_node_per_file_and_no_structure(tmp_path: Path) -> None:
    """Headings, lists and fenced code inside a document must not become nodes or edges."""
    doc = tmp_path / "guide.md"
    doc.write_text(
        "# Title\n\n## Install\n\n```python\ndef setup():\n    return 1\n```\n\n## Usage\n\ntext\n",
        encoding="utf-8",
    )

    nodes, edges = DocumentParser().parse_file(doc, tmp_path)

    assert [n.id for n in nodes] == ["guide.md"]
    assert nodes[0].node_type == NodeType.DOCUMENT
    assert nodes[0].name == "guide.md"  # with the extension: README.md and README.rst differ
    assert (nodes[0].line_start, nodes[0].line_end) == (1, 12)  # the whole file, one span
    assert edges == []


def test_document_parser_spans_line_one_even_for_an_empty_file(tmp_path: Path) -> None:
    """line_end=0 would put node_source and the chunker out of range for an empty doc."""
    doc = tmp_path / "empty.md"
    doc.write_text("", encoding="utf-8")

    nodes, _ = DocumentParser().parse_file(doc, tmp_path)

    assert (nodes[0].line_start, nodes[0].line_end) == (1, 1)


def test_document_extensions_are_normalized() -> None:
    """A configured ``md``/``.MD`` must reach the registry as ``.md`` — the scan compares suffixes."""
    parser = DocumentParser(["md", ".TXT", " .rst "])

    assert parser.get_supported_extensions() == [".md", ".rst", ".txt"]


@pytest.mark.asyncio
async def test_build_indexes_documents_alongside_code(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# App\n\nHow to run the app.\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "design.txt").write_text("Why the app is built this way.\n", encoding="utf-8")

    repo = _RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=_NO_LLM)
    nodes, edges = await builder.build("fixture", tmp_path)

    documents = {n.file_path: n for n in nodes if n.node_type == NodeType.DOCUMENT}
    assert set(documents) == {"README.md", "docs/design.txt"}
    # Documents hang off their directory like modules do, so containment traversal and
    # directory summaries see them instead of leaving them orphaned outside the tree.
    contains = {(e.source, e.target) for e in edges if e.edge_type == EdgeType.CONTAINS}
    assert ("dir:.", "README.md") in contains
    assert ("dir:docs", "docs/design.txt") in contains
    # Checksums are what make an unchanged document free on the next run.
    assert {"README.md", "docs/design.txt"} <= repo.checksums


@pytest.mark.asyncio
async def test_documents_are_not_scanned_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# App\n", encoding="utf-8")

    builder = GraphBuilder(
        repo=_RecordingRepository(),
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False, index_documents=False),
    )
    nodes, _ = await builder.build("fixture", tmp_path)

    assert ".md" not in builder.extensions
    assert all(n.file_path != "README.md" for n in nodes)


class _RecordingLLM:
    """Captures the (prompt, system_prompt) pairs a summarization sends."""

    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.system_prompts.append(system_prompt)
        return "summary"


@pytest.mark.asyncio
async def test_documents_are_summarized_as_prose_not_as_code(tmp_path: Path) -> None:
    """The code prompts ask for call graphs, side effects and return values a README has none of."""
    from inflorescence.code_indexer import prompts
    from inflorescence.code_indexer.summarizer import summarize_single

    (tmp_path / "README.md").write_text("# App\n\nRun `app serve` to start it.\n", encoding="utf-8")
    node = CodeNode(
        id="README.md", name="README.md", node_type=NodeType.DOCUMENT,
        file_path="README.md", line_start=1, line_end=3,
    )
    llm = _RecordingLLM()

    await summarize_single(node, tmp_path, llm, max_source_chars=12000)  # type: ignore[arg-type]

    assert llm.system_prompts == [prompts.DOCUMENT_SYSTEM_PROMPT]


@pytest.mark.asyncio
async def test_an_oversized_document_map_reduces_with_the_document_prompts(tmp_path: Path) -> None:
    from inflorescence.code_indexer import prompts
    from inflorescence.code_indexer.summarizer import summarize_single

    (tmp_path / "guide.md").write_text(_GUIDE, encoding="utf-8")
    node = CodeNode(
        id="guide.md", name="guide.md", node_type=NodeType.DOCUMENT,
        file_path="guide.md", line_start=1, line_end=_GUIDE.count("\n") + 1,
    )
    llm = _RecordingLLM()

    await summarize_single(node, tmp_path, llm, max_source_chars=500)  # type: ignore[arg-type]

    used = set(llm.system_prompts)
    assert used == {prompts.DOCUMENT_MAP_SYSTEM_PROMPT, prompts.DOCUMENT_REDUCE_SYSTEM_PROMPT}
    assert prompts.MAP_SYSTEM_PROMPT not in used


def test_document_chunks_cover_the_body_in_overlapping_windows(tmp_path: Path) -> None:
    """The whole file is chunked with overlap — the point of indexing documents at all."""
    (tmp_path / "guide.md").write_text(_GUIDE, encoding="utf-8")
    nodes, _ = DocumentParser().parse_file(tmp_path / "guide.md", tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    assert len(chunks) > 1, "a long document must be split, not embedded as one vector"
    assert {c.chunk_kind for c in chunks} == {"document"}
    assert {c.node_id for c in chunks} == {"guide.md"}  # every chunk resolves to the doc node
    assert {c.language for c in chunks} == {"markdown"}
    assert chunks[0].start_line == 1
    for previous, following in zip(chunks, chunks[1:], strict=False):
        # Overlap, not adjacency: the start of each window is still inside the previous
        # one, so a sentence the boundary cuts stays retrievable whole.
        assert following.start_line <= previous.end_line
        assert following.content[:120] in previous.content


def test_document_chunking_uses_its_own_window_size(tmp_path: Path) -> None:
    """Documents have no per-symbol chunks to fall back on, so their window is tunable alone."""
    (tmp_path / "guide.md").write_text(_GUIDE, encoding="utf-8")
    nodes, _ = DocumentParser().parse_file(tmp_path / "guide.md", tmp_path)

    coarse = Chunker(config=RAGConfig(document_chunk_size=2000, document_chunk_overlap=200))
    fine = Chunker(config=RAGConfig(document_chunk_size=200, document_chunk_overlap=50))

    assert len(fine.chunk_repository(tmp_path, nodes)) > len(coarse.chunk_repository(tmp_path, nodes))


def test_chunk_sizes_reach_the_chunker_from_settings() -> None:
    """CHUNK_SIZE and friends are documented knobs; they must not stop at Settings."""
    from inflorescence.rag.indexer import RAGIndexer

    settings = Settings(
        _env_file=None, chunk_size=1234, chunk_overlap=321,
        document_chunk_size=456, document_chunk_overlap=78,
    )
    config = RAGIndexer(repo=None, settings=settings)._chunker.config  # type: ignore[arg-type]

    assert (config.chunk_size, config.chunk_overlap) == (1234, 321)
    assert (config.document_chunk_size, config.document_chunk_overlap) == (456, 78)


def test_simple_split_terminates_even_when_overlap_swallows_the_window() -> None:
    """A mis-set overlap >= size would rewind the cursor forever and hang the indexer."""
    chunker = Chunker(config=RAGConfig(document_chunk_size=50, document_chunk_overlap=500))

    chunks = chunker._simple_split("word " * 200, "notes.txt", "text", 50, 500)

    assert len(chunks) > 1
    assert "".join(dict.fromkeys(c.content for c in chunks))  # produced content, did not spin


def test_every_node_type_has_a_storage_label() -> None:
    """upsert_nodes indexes the label map by value — a missing entry is a KeyError in prod."""
    assert set(_NODE_TYPE_LABEL_BY_VALUE) == {t.value for t in NodeType}


@pytest.mark.parametrize(
    "query_name",
    ["GET_CHECKSUMS", "GET_FILES_WITH_STALE_CHUNKS", "MARK_CHUNKS_COVERED", "GET_STRUCTURE_ROOT_BY_FILE"],
)
def test_file_level_queries_address_documents_too(query_name: str) -> None:
    """These identify a file by its whole-file node, which for a document is not a :Module.

    Scoped to :Module alone, a document's checksum would be unreadable (so every update
    re-parses, re-summarizes and re-embeds it) and its chunk-coverage marker unwritable
    (so the repair pass re-chunks it forever).
    """
    from inflorescence.db import queries

    query = getattr(queries, query_name)

    # Case-insensitive on purpose: a query may address the file node via the label form
    # (:Module / :Document) or the property form (node_type IN ['module', 'document']) —
    # both satisfy the invariant; scoping to modules alone does not.
    lowered = query.lower()
    assert ":code:module" not in lowered
    assert "module" in lowered and "document" in lowered


# ===========================================================================
# Styles, markup and configuration are documents too
# ===========================================================================

INDEX_CSS = """:root {
  --accent: #2f6fed;
  --card-radius: 12px;
}

.card {
  border-radius: var(--card-radius);
  transition: transform 220ms ease-out;
}

@media (prefers-reduced-motion: reduce) {
  .card {
    transition: none;
    animation: none;
  }
}
"""

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Widget console</title>
    <link rel="stylesheet" href="/src/index.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
"""

TSCONFIG_JSON = """{
  "compilerOptions": {
    "target": "ES2022",
    "jsx": "react-jsx",
    "strict": true,
    "moduleResolution": "bundler"
  },
  "include": ["src"]
}
"""

COMPOSE_YAML = """services:
  memgraph:
    image: memgraph/memgraph-mage:latest
    ports:
      - "7687:7687"
"""

# One representative file per newly admitted family, keyed by the extension under test.
_NEW_DOCUMENT_FILES: dict[str, tuple[str, str]] = {
    ".css": ("src/index.css", INDEX_CSS),
    ".scss": ("src/theme.scss", "$accent: #2f6fed;\n.card { color: $accent; }\n"),
    ".sass": ("src/theme.sass", ".card\n  color: #2f6fed\n"),
    ".less": ("src/theme.less", "@accent: #2f6fed;\n.card { color: @accent; }\n"),
    ".styl": ("src/theme.styl", "accent = #2f6fed\n.card\n  color accent\n"),
    ".html": ("index.html", INDEX_HTML),
    ".htm": ("legacy.htm", "<html><body>legacy entry point</body></html>\n"),
    ".json": ("tsconfig.json", TSCONFIG_JSON),
    ".jsonc": ("tsconfig.build.jsonc", '// build overrides\n{ "extends": "./tsconfig.json" }\n'),
    ".yaml": ("docker-compose.yaml", COMPOSE_YAML),
    ".yml": ("docker-compose.yml", COMPOSE_YAML),
    ".toml": ("pyproject.toml", '[project]\nname = "widget"\nversion = "0.1.0"\n'),
    ".ini": ("pytest.ini", "[pytest]\nasyncio_mode = auto\n"),
    ".cfg": ("setup.cfg", "[flake8]\nmax-line-length = 100\n"),
    ".conf": ("nginx.conf", "server {\n  listen 80;\n}\n"),
}


@pytest.mark.parametrize("extension", sorted(_NEW_DOCUMENT_FILES))
def test_styles_markup_and_config_are_document_extensions(extension: str) -> None:
    """The registry is what every downstream stage keys off — scan, watcher, preview, reconcile.

    An extension missing here is not "ranked low", it is invisible: the walk never opens
    the file. That is exactly how a defect in ``@media (prefers-reduced-motion)`` inside
    index.css was unfindable through this server.
    """
    assert extension in DEFAULT_DOCUMENT_EXTENSIONS
    assert extension in DocumentParser().get_supported_extensions()
    assert extension in GraphBuilder(repo=None, llm=None, config=_NO_LLM).extensions


@pytest.mark.parametrize("extension", [".xml", ".vue", ".svelte", ".csv", ".env", ".lock", ".svg"])
def test_deliberately_excluded_extensions_stay_out(extension: str) -> None:
    """Pinned decisions, so widening one of them is a conscious edit rather than a drift.

    ``.xml`` is dominated by machine-written trees (IDE settings, JUnit/coverage reports,
    generated resources) whose tag soup chunks and embeds badly, and its hand-written uses
    belong to toolchains with no parser here. ``.vue``/``.svelte`` are *code* with no parser
    — indexing them as prose would claim coverage the graph does not have. ``.csv`` is data
    with no prose at all, ``.env`` holds secrets, ``.lock``/``.svg`` are generated bulk.
    """
    assert extension not in DEFAULT_DOCUMENT_EXTENSIONS


@pytest.mark.asyncio
async def test_build_stores_styles_markup_and_config_as_documents_with_checksums(
    tmp_path: Path,
) -> None:
    """One Document node per file, hung off its directory, with a checksum.

    The checksum is the whole freshness/reuse story: it is what ``GET_CHECKSUMS`` reads
    (``n:Module OR n:Document``), what makes an unchanged stylesheet free on the next run,
    and what stale-detection compares against disk. A new file kind that produced nodes but
    no checksums would be re-summarized and re-embedded on every single update.
    """
    (tmp_path / "src").mkdir()
    for rel, text in _NEW_DOCUMENT_FILES.values():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")

    repo = _RecordingRepository()
    nodes, edges = await GraphBuilder(repo=repo, llm=None, config=_NO_LLM).build("fixture", tmp_path)

    expected = {rel for rel, _ in _NEW_DOCUMENT_FILES.values()}
    documents = {n.file_path: n for n in nodes if n.node_type == NodeType.DOCUMENT}
    assert expected <= set(documents)
    # Exactly one node per file: no selectors, no `<head>`, no JSON keys became symbols.
    assert [n.file_path for n in nodes].count("src/index.css") == 1
    assert not any(n.id.startswith("src/index.css::") for n in nodes)
    contains = {(e.source, e.target) for e in edges if e.edge_type == EdgeType.CONTAINS}
    assert ("dir:src", "src/index.css") in contains
    assert ("dir:.", "index.html") in contains
    assert expected <= repo.checksums


@pytest.mark.asyncio
async def test_lock_files_are_never_indexed(tmp_path: Path) -> None:
    """A resolver transcript mentions every package in the tree and changes on every install.

    Dropped **by name**, before include patterns are consulted — an explicit
    ``include_patterns=["*"]`` must not pull one in either.
    """
    (tmp_path / "package.json").write_text('{"name": "widget"}\n', encoding="utf-8")
    lock_body = json.dumps({"packages": {f"node_modules/pkg{i}": {"version": "1.0.0"} for i in range(40)}})
    for name in ("package-lock.json", "npm-shrinkwrap.json", "packages.lock.json"):
        (tmp_path / name).write_text(lock_body, encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\npackages:\n  pkg: {}\n", encoding="utf-8")

    builder = GraphBuilder(repo=_RecordingRepository(), llm=None, config=_NO_LLM)
    nodes, _ = await builder.build("fixture", tmp_path, include_patterns=["*"])

    indexed = {n.file_path for n in nodes if n.file_path}
    assert "package.json" in indexed, "package.json is hand-written and belongs in the index"
    assert not (indexed & LOCKFILE_NAMES)
    # The preview must quote the same world it would index, or the estimate is a lie.
    preview = builder.preview(tmp_path)
    assert preview["files"] == 1
    assert not any(f["path"] in LOCKFILE_NAMES for f in preview["largest_files"])  # type: ignore[index]


@pytest.mark.asyncio
async def test_an_oversized_generated_config_is_reported_not_indexed(tmp_path: Path) -> None:
    """``skipped_large_files`` is the honest channel for "this exists and I did not read it".

    A generated 200-key JSON blob is exactly the bulk that admitting data formats risks
    paying for. It must be named in the response — both in the preview and in the build
    result — rather than silently dropped or silently summarized.
    """
    (tmp_path / "small.json").write_text('{"a": 1}\n', encoding="utf-8")
    (tmp_path / "generated.json").write_text(
        json.dumps({f"key{i}": "x" * 200 for i in range(200)}), encoding="utf-8"
    )
    (tmp_path / "huge.css").write_text(".a { color: red; }\n" * 4000, encoding="utf-8")

    config = IndexerConfig(
        use_llm_summaries=False, use_llm_fallback_parser=False, max_file_size_bytes=4096
    )
    builder = GraphBuilder(repo=_RecordingRepository(), llm=None, config=config)
    result = await builder.build("fixture", tmp_path)

    assert set(result.skipped_large_files) == {"generated.json", "huge.css"}
    assert {n.file_path for n in result.nodes if n.node_type == NodeType.DOCUMENT} == {"small.json"}
    assert set(builder.preview(tmp_path)["skipped_large_files"]) == {"generated.json", "huge.css"}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ignored_and_excluded_trees_do_not_leak_config_or_styles(tmp_path: Path) -> None:
    """node_modules and dist are where generated JSON/CSS lives by the megabyte.

    They were already kept out for code; the point here is that widening the extension set
    did not open a side door into them.
    """
    (tmp_path / ".gitignore").write_text("dist/\ncoverage.json\n", encoding="utf-8")
    for rel in (
        "node_modules/pkg/package.json",
        "node_modules/pkg/style.css",
        "dist/assets/index.css",
        "dist/index.html",
        ".next/build-manifest.json",
    ):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    (tmp_path / "coverage.json").write_text('{"total": 1}\n', encoding="utf-8")
    (tmp_path / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    nodes, _ = await GraphBuilder(repo=_RecordingRepository(), llm=None, config=_NO_LLM).build(
        "fixture", tmp_path
    )

    assert {n.file_path for n in nodes if n.node_type == NodeType.DOCUMENT} == {"index.html"}


def test_preview_prices_the_new_document_types(tmp_path: Path) -> None:
    """Widening the scan raises the estimate for existing projects — the preview must say so.

    ``index_directory(preview=true)`` is the only thing between a user and an unexpected
    bill, so the new files have to be inside its file count, its byte total and its
    per-language breakdown, not appear for the first time when the run is already paying.
    """
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    (tmp_path / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (tmp_path / "index.css").write_text(INDEX_CSS, encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text(TSCONFIG_JSON, encoding="utf-8")
    (tmp_path / "docker-compose.yml").write_text(COMPOSE_YAML, encoding="utf-8")

    with_documents = GraphBuilder(repo=None, llm=None, config=_NO_LLM).preview(tmp_path)
    code_only = GraphBuilder(
        repo=None, llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False, index_documents=False),
    ).preview(tmp_path)

    assert with_documents["files"] == 5 and code_only["files"] == 1
    assert set(with_documents["languages"]) == {"python", "html", "css", "json", "yaml"}  # type: ignore[arg-type]
    assert with_documents["estimated_tokens"] > code_only["estimated_tokens"]  # type: ignore[operator]
    assert with_documents["estimated_nodes"] > code_only["estimated_nodes"]  # type: ignore[operator]


def test_a_stylesheet_is_chunked_as_a_document_not_parsed_as_code(tmp_path: Path) -> None:
    """The media query must survive verbatim in a chunk — that is what BM25 will match.

    And the chunk must come from the document path: a CSS AST split would produce chunks
    bound to symbols this graph never created.
    """
    (tmp_path / "index.css").write_text(INDEX_CSS, encoding="utf-8")
    nodes, _ = DocumentParser().parse_file(tmp_path / "index.css", tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    assert {c.chunk_kind for c in chunks} == {"document"}
    assert {c.node_id for c in chunks} == {"index.css"}
    assert {c.language for c in chunks} == {"css"}
    assert any("prefers-reduced-motion" in c.content for c in chunks)


@pytest.mark.parametrize(
    ("name", "language"),
    [("index.css", "css"), ("index.html", "html"), ("tsconfig.json", "json"), ("app.yml", "yaml")],
)
def test_chunk_file_takes_the_document_path_for_document_formats(
    tmp_path: Path, name: str, language: str
) -> None:
    """``chunk_file`` and ``chunk_repository`` must not chunk the same stylesheet two ways.

    Several document formats (.css, .html, .json, .yaml, .toml) also have a tree-sitter
    CodeSplitter grammar, so the ordering of the two lookups decides whether this entry
    point builds an AST of a file the pipeline treats as text.
    """
    (tmp_path / name).write_text(INDEX_CSS if name.endswith(".css") else INDEX_HTML, encoding="utf-8")

    chunks = Chunker().chunk_file(tmp_path / name, tmp_path)

    assert chunks and {c.language for c in chunks} == {language}
    assert {c.chunk_kind for c in chunks} == {"fallback_text"}  # the text path, not _chunk_code


# ===========================================================================
# Live · the regression the widening exists for
# ===========================================================================

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536


class _LiveLLM:
    """Summarizes without spending: the claim under test is retrieval, not summary quality."""

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        return "a concise summary"

    async def close(self) -> None:
        return None


class _LiveEmbedder:
    """Deterministic 1536-dim vectors — BM25 needs the chunk rows, not their geometry."""

    def __init__(self) -> None:
        self.model = "fake-embed"

    def preflight(self) -> None:
        return None

    def embed(self, texts):
        return [[0.01 * ((i % 97) + 1)] * _DIM for i, _ in enumerate(list(texts))]


@live
@pytest.mark.asyncio
async def test_a_css_media_query_is_findable_by_search_text_live(tmp_path: Path) -> None:
    """The dogfooding miss, end to end against a real Memgraph and the real MCP tool.

    A frontend defect lived in ``@media (prefers-reduced-motion)`` inside ``index.css``.
    Asking this server for it returned nothing — not a bad ranking, an empty result, because
    the file was never scanned. This indexes a mini project the same way the server does and
    asks ``search_text`` the question that failed: the stylesheet must come back, as a
    ``document`` entity, with the media query among its matches.
    """
    import json as _json

    from mcp.server.fastmcp import FastMCP

    from inflorescence.code_indexer.graph_builder import GraphBuilder as _GraphBuilder
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.db.schema import setup_schema
    from inflorescence.project_manager import ProjectManager
    from inflorescence.rag.indexer import RAGIndexer
    from inflorescence.tools.search import register_search_tools

    project_root = tmp_path / "frontend"
    (project_root / "src").mkdir(parents=True)
    (project_root / "src" / "index.css").write_text(INDEX_CSS, encoding="utf-8")
    (project_root / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (project_root / "tsconfig.json").write_text(TSCONFIG_JSON, encoding="utf-8")
    (project_root / "package-lock.json").write_text(
        _json.dumps({"packages": {"node_modules/motion": {"version": "1.0.0"}}}), encoding="utf-8"
    )
    (project_root / "src" / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL, embedding_dimension=_DIM)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    repo = GraphRepository(conn)
    builder = _GraphBuilder(
        repo=repo, llm=_LiveLLM(), settings=settings,
        config=IndexerConfig.from_settings(settings).model_copy(update={"use_scip_semantic": False}),
    )
    rag = RAGIndexer(repo=repo, settings=settings)
    rag._code_embedder = _LiveEmbedder()  # type: ignore[assignment]
    rag._summary_embedder = _LiveEmbedder()  # type: ignore[assignment]
    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn
    )
    project = manager.get_project(str(project_root))
    try:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await manager.index_directory(str(project_root))

        mcp = FastMCP("t")
        register_search_tools(mcp, manager, repo, settings)

        def _payload(res):
            content = res[0] if isinstance(res, tuple) else res
            return _json.loads(content[0].text)

        payload = _payload(await mcp.call_tool(
            "search_text", {"directory": str(project_root), "query": "prefers-reduced-motion"}
        ))

        assert "error" not in payload, payload
        hits = {item["entity"]["file_path"]: item for item in payload["items"]}
        assert "src/index.css" in hits, (
            "the stylesheet holding @media (prefers-reduced-motion) was not findable — "
            f"got {sorted(hits)}"
        )
        assert hits["src/index.css"]["entity"]["node_type"] == "document"
        # The hit came from a chunk of the file body — and that chunk really holds the media
        # query, so the match is the rule itself and not the file name catching a token.
        chunk_ids = [m["chunk_id"] for m in hits["src/index.css"].get("matches", [])]
        assert chunk_ids, hits["src/index.css"]
        chunk_rows = await conn.execute_query(
            "MATCH (c:CodeChunk {project: $p}) WHERE c.id IN $ids RETURN c.content AS content",
            {"p": project, "ids": chunk_ids},
        )
        assert any("prefers-reduced-motion" in str(r["content"]) for r in chunk_rows)

        # The other newly admitted kinds landed too, and the lock file did not.
        indexed = {
            str(r["p"]) for r in await conn.execute_query(
                "MATCH (n:Document {project: $p}) RETURN n.file_path AS p", {"p": project}
            )
        }
        assert indexed == {"src/index.css", "index.html", "tsconfig.json"}

        # The bookkeeping the recent features hang off, read back from the live graph for a
        # file kind that did not exist when they were written:
        #   * a checksum, so freshness/stale-detection can compare it against disk and so a
        #     re-index reuses it instead of paying again (GET_CHECKSUMS: n:Module OR n:Document);
        #   * no stale-chunk entry, so MARK_CHUNKS_COVERED stamped it and the repair pass will
        #     not re-chunk and re-embed the stylesheet on every run.
        checksums = await repo.get_checksums(project)
        assert {"src/index.css", "index.html", "tsconfig.json"} <= set(checksums)
        assert await repo.get_files_with_stale_chunks(project) == []
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()
