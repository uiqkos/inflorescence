"""Documents (.md/.txt/...) are indexed whole: one node per file, body chunked with overlap.

The contract these guard:
  * a prose file is *scanned* at all — before this it fell outside the registry's
    extensions, so no scan saw it, no chunker touched it, and a repository's entire
    documentation was invisible to every search tool;
  * it produces exactly ONE node — no invented heading/section symbols;
  * its content reaches the index as overlapping chunks, so a passage cut by a window
    boundary survives whole in the neighbouring window;
  * the file-level bookkeeping (checksums, chunk-coverage) addresses it, so an unchanged
    document costs nothing on re-index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import CodeNode, EdgeType, IndexerConfig, NodeType
from inflorescence.code_indexer.parser.document_parser import DocumentParser
from inflorescence.config import Settings
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

    assert ":Code:Module" not in query
    assert "Module" in query and "Document" in query
