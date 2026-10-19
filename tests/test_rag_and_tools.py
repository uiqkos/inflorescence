from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.code_indexer.models import CodeNode, NodeType
from inflorescence.config import Settings
from inflorescence.db import queries
from inflorescence.db.repository import GraphRepository, ProjectHits
from inflorescence.db.schema import setup_schema
from inflorescence.rag.chunker import Chunk
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.rag.indexer import RAGIndexer
from inflorescence.tools import freshness
from inflorescence.tools.cypher import register_cypher_tools
from inflorescence.tools.freshness import FreshnessChecker
from inflorescence.tools.graph import register_graph_tools
from inflorescence.tools.indexing import register_indexing_tools
from inflorescence.tools.path_filters import PathFilterConfig, is_test_path, path_is_allowed
from inflorescence.tools.responses import (
    FULL_SUMMARY_MAX_LIMIT,
    NAV_SUMMARY_MAX_CHARS,
    entity_ref,
    error_response,
    paginated_response,
    truncate_summary,
)
from inflorescence.tools.search import _code_hits, register_search_tools
from tests.test_worktree import git, make_repo


def test_embedding_client_splits_batches_by_count_and_character_limits() -> None:
    client = EmbeddingClient(Settings(_env_file=None), model="test-model", batch_size=2, batch_max_chars=5)

    assert client._split_batches(["aa", "bb", "cccc", "d"]) == [["aa", "bb"], ["cccc", "d"]]


def test_default_embedding_models_match_schema_dimension() -> None:
    settings = Settings(_env_file=None)

    assert settings.code_embedding_model == "openai/text-embedding-3-small"
    assert settings.summary_embedding_model == "openai/text-embedding-3-small"
    assert settings.embedding_dimension == 1536


def test_embedding_client_preserves_alignment_when_batch_fails(monkeypatch) -> None:
    client = EmbeddingClient(Settings(_env_file=None), model="test-model", batch_size=2)

    def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        if "bad" in texts:
            raise RuntimeError("provider failed")
        return [[1.0] for _ in texts]

    monkeypatch.setattr(client, "_embed_batch", fake_embed_batch)

    assert client.embed(["ok1", "bad", "ok2"]) == [None, None, [1.0]]


def test_entity_ref_strips_internal_fields() -> None:
    raw = {
        "id": "validator.py::BareValidator",
        "name": "BareValidator",
        "node_type": "class",
        "file_path": "validator.py",
        "line_start": 10,
        "line_end": 99,
        "summary": "Validates documents.",
        "embedding": [0.1, 0.2],
        "summary_embedding": [0.3, 0.4],
        "source_code": "class BareValidator: ...",
        "content": "internal chunk",
    }

    assert entity_ref(raw) == {
        "id": "validator.py::BareValidator",
        "name": "BareValidator",
        "node_type": "class",
        "file_path": "validator.py",
        "line_start": 10,
        "line_end": 99,
        "summary": "Validates documents.",
    }


def test_truncate_summary_bounds_long_text_with_ellipsis() -> None:
    long_text = "word " * 100

    truncated = truncate_summary(long_text, 200)

    assert len(truncated) <= 201  # bound plus the ellipsis
    assert truncated.endswith("…")
    assert truncate_summary("short", 200) == "short"
    assert truncate_summary(long_text, 0) == long_text  # non-positive bound disables truncation


def test_entity_ref_truncates_summary_only_when_bounded() -> None:
    raw = entity(summary="s" * 400)

    bounded = entity_ref(raw, summary_max_chars=NAV_SUMMARY_MAX_CHARS)
    unbounded = entity_ref(raw)

    assert bounded["summary"] == "s" * NAV_SUMMARY_MAX_CHARS + "…"
    assert unbounded["summary"] == "s" * 400


def test_paginated_response_uses_limit_plus_one_item() -> None:
    payload = paginated_response(
        items=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
        limit=2,
        offset=4,
        meta={"project": "project-id"},
    )

    assert payload == {
        "items": [{"id": "a"}, {"id": "b"}],
        "page": {"limit": 2, "offset": 4, "next_offset": 6, "has_more": True},
        "meta": {"project": "project-id"},
    }


def test_error_response_shape() -> None:
    assert error_response("not_found", "Entity not found: missing") == {
        "error": {"code": "not_found", "message": "Entity not found: missing"}
    }


def test_path_filter_config_search_defaults_include_tests() -> None:
    config = PathFilterConfig.from_paths(include_paths=None, exclude_paths=None)

    assert config.include == []
    assert config.include_all is True
    assert config.exclude == []
    assert path_is_allowed("src/app.py", config) is True
    assert path_is_allowed("tests/test_app.py", config) is True
    assert path_is_allowed("src/app.test.ts", config) is True


def test_path_filter_config_empty_exclude_disables_defaults() -> None:
    config = PathFilterConfig.from_paths(include_paths=None, exclude_paths=[])

    assert config.exclude == []
    assert path_is_allowed("tests/test_app.py", config) is True


def test_path_filter_include_and_exclude_patterns() -> None:
    config = PathFilterConfig(include=["**/*.py"], exclude=["**/fixtures/**"], include_all=False)

    assert path_is_allowed("src/app.py", config) is True
    assert path_is_allowed("src/app.ts", config) is False
    assert path_is_allowed("src/fixtures/app.py", config) is False


def test_path_filter_empty_include_matches_no_paths() -> None:
    config = PathFilterConfig.from_paths(include_paths=[], exclude_paths=None)

    assert config.include == []
    assert config.include_all is False
    assert path_is_allowed("src/app.py", config) is False


@pytest.mark.asyncio
async def test_rag_indexer_skips_failed_code_embeddings(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], None])

    stored = await indexer.index_code(
        "fixture",
        tmp_path,
        [CodeNode(id="module.py", name="module", node_type=NodeType.MODULE, file_path="module.py", line_start=1, line_end=2)],
    )

    assert stored == 1
    assert len(repo.code_chunks) == 1
    assert repo.code_chunks[0]["content"] == "first chunk"


@pytest.mark.asyncio
async def test_rag_indexer_stores_entity_linked_chunk_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], None])

    stored = await indexer.index_code(
        "fixture",
        tmp_path,
        [CodeNode(id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION, file_path="module.py", line_start=1, line_end=2)],
    )

    assert stored == 1
    assert repo.code_chunks == [
        {
            "chunk_id": repo.code_chunks[0]["chunk_id"],
            "node_id": "module.py::alpha",
            "file_path": "module.py",
            "content": "first chunk",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
            "chunk_kind": "function_body",
            "embedding": [0.1, 0.2],
        }
    ]


@pytest.mark.asyncio
async def test_rag_indexer_uses_stable_chunk_ids_for_identical_chunks(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    file_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], None])

    node = CodeNode(
        id="module.py::alpha",
        name="alpha",
        node_type=NodeType.FUNCTION,
        file_path="module.py",
        line_start=1,
        line_end=2,
    )
    await indexer.index_code("fixture", tmp_path, [node])
    await indexer.index_code("fixture", tmp_path, [node])

    assert repo.code_chunks[0]["chunk_id"] == repo.code_chunks[1]["chunk_id"]


def test_store_code_chunk_query_upserts_chunks_and_relationships() -> None:
    assert "MERGE (c:CodeChunk" in queries.STORE_CODE_CHUNK
    assert "MERGE (n)-[:HAS_CHUNK]->(c)" in queries.STORE_CODE_CHUNK


@pytest.mark.asyncio
async def test_rag_indexer_prunes_stale_chunks_only_when_no_embeddings_failed(tmp_path: Path) -> None:
    """prune_stale removes chunks absent from this run — but never after a failed embed,
    so an API outage can only leave stale chunks behind, not shrink the stored set."""
    file_path = tmp_path / "module.py"
    file_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    node = CodeNode(
        id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION,
        file_path="module.py", line_start=1, line_end=2,
    )

    # All embeds succeed -> the stale stored chunk is pruned.
    repo = FakeVectorRepository(chunk_ids={"chunk:stale-id"})
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], [0.3, 0.4]])
    await indexer.index_code("fixture", tmp_path, [node], prune_stale=True)
    assert repo.deleted_chunk_ids == ["chunk:stale-id"]

    # One embed fails -> pruning is skipped entirely.
    repo = FakeVectorRepository(chunk_ids={"chunk:stale-id"})
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], None])
    await indexer.index_code("fixture", tmp_path, [node], prune_stale=True)
    assert repo.deleted_chunk_ids == []


@pytest.mark.asyncio
async def test_rag_indexer_scopes_stale_prune_to_reconcilable_files(tmp_path: Path) -> None:
    """INV-1/INV-3 (finding 4): a file a read/parse failure dropped from the build keeps its
    chunks. With reconcilable_files set, prune removes only chunks of files this run
    actually rebuilt (module.py), never a kept file's chunks (other.py)."""
    file_path = tmp_path / "module.py"
    file_path.write_text("def alpha():\n    return 1\n", encoding="utf-8")
    node = CodeNode(
        id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION,
        file_path="module.py", line_start=1, line_end=2,
    )

    repo = FakeVectorRepository()
    # module.py has a leftover stale chunk; other.py was dropped by a failure this run.
    repo.chunk_index = {"chunk:stale-module": "module.py", "chunk:kept-other": "other.py"}
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2], [0.3, 0.4]])

    await indexer.index_code(
        "fixture", tmp_path, [node], prune_stale=True, reconcilable_files={"module.py"}
    )

    # Only module.py's stale chunk is pruned; the kept file's chunk survives.
    assert repo.deleted_chunk_ids == ["chunk:stale-module"]


class _EmptyChunker:
    """A run whose files legitimately produce zero chunks (e.g. the file was emptied)."""

    def chunk_repository(
        self, root: Path, nodes: list[CodeNode], unreadable_out: set[str] | None = None
    ) -> list[Chunk]:
        return []


@pytest.mark.asyncio
async def test_index_code_prunes_emptied_files_before_stamping_coverage(tmp_path: Path) -> None:
    """INV-8 (finding 8b): a file emptied of code must not keep its old chunks invisibly.

    The zero-chunks early exit used to stamp chunk coverage without pruning, so the
    emptied file's stale chunks stayed stored, findable by search, and undetectable by
    the staleness query (chunks_checksum matched). Pruning must run before the stamp."""
    file_path = tmp_path / "module.py"
    file_path.write_text("# emptied: all code removed\n", encoding="utf-8")
    node = CodeNode(
        id="module.py", name="module", node_type=NodeType.MODULE,
        file_path="module.py", line_start=1, line_end=1,
    )

    repo = FakeVectorRepository(checksums={"module.py": "new-checksum"})
    repo.chunk_index = {"chunk:old-module": "module.py", "chunk:kept-other": "other.py"}
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=_EmptyChunker())
    indexer._code_embedder = FakeEmbedder([])

    await indexer.index_code(
        "fixture", tmp_path, [node], prune_stale=True, reconcilable_files={"module.py"}
    )

    # The emptied file's old chunks are gone; an unrelated kept file's chunk survives;
    # coverage is still stamped for the (now legitimately chunk-less) file.
    assert repo.deleted_chunk_ids == ["chunk:old-module"]
    assert {item["file_path"] for item in repo.chunks_covered} == {"module.py"}


@pytest.mark.asyncio
async def test_index_code_zero_chunks_without_scope_does_not_wipe(tmp_path: Path) -> None:
    """Without reconcilable_files, zero chunks is indistinguishable from total failure —
    pruning must be skipped rather than wiping every stored chunk of the project (INV-1)."""
    node = CodeNode(
        id="module.py", name="module", node_type=NodeType.MODULE,
        file_path="module.py", line_start=1, line_end=1,
    )
    repo = FakeVectorRepository(chunk_ids={"chunk:a", "chunk:b"})
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=_EmptyChunker())
    indexer._code_embedder = FakeEmbedder([])

    await indexer.index_code("fixture", tmp_path, [node], prune_stale=True)

    assert repo.deleted_chunk_ids == []


@pytest.mark.asyncio
async def test_setup_schema_creates_text_indexes_for_symbols_and_chunk_content() -> None:
    conn = RecordingSchemaConnection()

    await setup_schema(conn)

    text_index_writes = [query for query, _params in conn.writes if "CREATE TEXT INDEX" in query]
    assert any("code_symbols_text ON :Code(name, signature)" in query for query in text_index_writes)
    assert any("chunk_content_text ON :CodeChunk(content)" in query for query in text_index_writes)


def test_property_indexes_include_node_type() -> None:
    from inflorescence.db.schema import PROPERTY_INDEXES

    assert ("Code", "node_type") in PROPERTY_INDEXES


class _BackfillConnection:
    """Records writes and answers the backfill probe with a configurable result."""

    def __init__(self, probe_rows: list[dict[str, object]]) -> None:
        self._probe_rows = probe_rows
        self.writes: list[str] = []

    async def execute_query(self, query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
        assert "n.node_type IS NULL" in query
        return self._probe_rows

    async def execute_write(self, query: str, params: dict[str, object] | None = None) -> None:
        self.writes.append(query)


@pytest.mark.asyncio
async def test_backfill_node_type_sets_property_from_label_when_missing() -> None:
    from inflorescence.code_indexer.models import NodeType
    from inflorescence.db.schema import backfill_node_type

    conn = _BackfillConnection([{"id": "pkg/a.py::run"}])
    updated = await backfill_node_type(conn)

    # One SET per node-type label group — derived, so adding a NodeType extends the
    # backfill instead of failing this assertion on a hardcoded count.
    assert updated == len(NodeType)
    assert any("MATCH (n:Function) WHERE n.node_type IS NULL SET n.node_type = 'function'" in q for q in conn.writes)
    assert any("MATCH (n:Module) WHERE n.node_type IS NULL SET n.node_type = 'module'" in q for q in conn.writes)
    assert any("MATCH (n:Document) WHERE n.node_type IS NULL SET n.node_type = 'document'" in q for q in conn.writes)
    assert all("SET n.node_type = '" in q for q in conn.writes)


@pytest.mark.asyncio
async def test_backfill_node_type_is_a_noop_when_probe_finds_nothing() -> None:
    from inflorescence.db.schema import backfill_node_type

    conn = _BackfillConnection([])
    updated = await backfill_node_type(conn)

    assert updated == 0
    assert conn.writes == []


@pytest.mark.asyncio
async def test_graph_schema_describes_labels_types_and_rules() -> None:
    mcp = FastMCP("test")
    register_cypher_tools(mcp, FakeManager(), FakeConnection())

    payload = _payload(await mcp.call_tool("graph_schema", {"directory": "/tmp/project"}))

    assert payload["project"] == "project-id"
    assert "method" in payload["node_types"] and "function" in payload["node_types"]
    assert "Method" in payload["node_labels"]["by_type"]
    assert "CALLS" in payload["relationship_types"] and "HAS_CHUNK" in payload["relationship_types"]
    assert "node_type" in payload["entity_properties"]
    assert any("project: $project" in rule for rule in payload["query_rules"])
    assert "code_embeddings" in payload["indexes"]["vector"]


def test_text_search_queries_use_search_all_and_join_chunks_to_entities() -> None:
    assert "text_search.search_all('summary_text'" in queries.TEXT_SEARCH_SUMMARY
    assert "text_search.search_all('code_symbols_text'" in queries.TEXT_SEARCH_SYMBOL
    assert "text_search.search_all('chunk_content_text'" in queries.TEXT_SEARCH_CODE
    assert "MATCH (code:Code {id: node.node_id, project: $project})" in queries.TEXT_SEARCH_CODE
    assert "node.id AS chunk_id" in queries.TEXT_SEARCH_CODE
    for query in (queries.TEXT_SEARCH_SUMMARY, queries.TEXT_SEARCH_SYMBOL, queries.TEXT_SEARCH_CODE):
        assert "score AS score" in query
        assert "WHERE node.project = $project" in query
        # The window is always explicit: left out, the procedure stops at 1000 rows across
        # every project, before the project filter runs (INV-9).
        assert "{limit: $fetch}" in query
        assert "ORDER BY score DESC" in query


def test_rag_indexer_default_chunker_inherits_settings_size_limit() -> None:
    settings = Settings(_env_file=None, max_file_size_bytes=524288)

    indexer = RAGIndexer(FakeVectorRepository(), settings)

    assert indexer._chunker.config.max_file_size_bytes == 524288


@pytest.mark.asyncio
async def test_rag_indexer_skips_empty_summaries_and_failed_summary_embeddings() -> None:
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._summary_embedder = FakeEmbedder([None, [0.3, 0.4]])

    stored = await indexer.index_summaries(
        "fixture",
        [
            CodeNode(id="a.py", name="a", node_type=NodeType.MODULE, file_path="a.py", line_start=1, line_end=1, summary=""),
            CodeNode(id="a.py::first", name="first", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=1, summary="first summary"),
            CodeNode(id="a.py::second", name="second", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=1, summary="second summary"),
        ],
    )

    assert stored == 1
    # Each stored embedding now carries the freshness marker (the node's summary_input_hash
    # at embed time); these nodes have no hash set, so it is "" (audit finding 8a).
    assert repo.summary_embeddings == [{"node_id": "a.py::second", "embedding": [0.3, 0.4], "embedding_hash": ""}]


@pytest.mark.asyncio
async def test_index_summaries_reuses_embeddings_for_non_dirty_nodes() -> None:
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._summary_embedder = FakeEmbedder([[9.0, 9.0]])  # only ONE fresh embed allowed

    nodes = [
        CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=1, summary="new foo"),
        CodeNode(id="a.py::bar", name="bar", node_type=NodeType.FUNCTION, file_path="a.py", line_start=2, line_end=2, summary="old bar"),
    ]
    stored = {"a.py::bar": {"summary": "old bar", "summary_input_hash": "h", "summary_embedding": [1.0, 2.0]}}

    stored_count = await indexer.index_summaries("fixture", nodes, stored_summaries=stored, dirty_ids={"a.py::foo"})

    assert stored_count == 2
    by_id = {item["node_id"]: item["embedding"] for item in repo.summary_embeddings}
    assert by_id["a.py::foo"] == [9.0, 9.0]   # freshly embedded
    assert by_id["a.py::bar"] == [1.0, 2.0]   # reused, no embed call


@pytest.mark.asyncio
async def test_index_summaries_reembeds_stale_vector_reuses_fresh_and_legacy() -> None:
    """INV-3/INV-8 oracle (audit finding 8a): a stored summary vector is reused only when it is
    still fresh for the current summary.

    A node not in ``dirty_ids`` used to unconditionally reuse its stored vector — so a
    summary regenerated with a dropped embedding kept the old vector forever. Now reuse is
    gated on the freshness marker: a mismatched hash is re-embedded, a matching hash is
    reused, and a legacy (unmarked) vector is left alone so an upgrade doesn't re-embed
    the world. Fails on the old code, which reuses the stale vector instead of re-embedding.
    """
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._summary_embedder = FakeEmbedder([[7.0]])  # exactly ONE fresh embed is allowed

    def node(nid: str) -> CodeNode:
        n = CodeNode(id=nid, name=nid, node_type=NodeType.FUNCTION,
                     file_path="a.py", line_start=1, line_end=1, summary="s")
        n.summary_input_hash = "H1"
        return n

    nodes = [node("stale"), node("fresh"), node("legacy")]
    stored = {
        "stale":  {"summary": "s", "summary_input_hash": "H1", "summary_embedding": [1.0], "summary_embedding_hash": "H0"},
        "fresh":  {"summary": "s", "summary_input_hash": "H1", "summary_embedding": [2.0], "summary_embedding_hash": "H1"},
        "legacy": {"summary": "s", "summary_input_hash": "H1", "summary_embedding": [3.0], "summary_embedding_hash": None},
    }

    await indexer.index_summaries("fixture", nodes, stored_summaries=stored, dirty_ids=set())

    items = {i["node_id"]: i for i in repo.summary_embeddings}
    assert items["stale"]["embedding"] == [7.0]        # re-embedded: recorded hash != current
    assert items["stale"]["embedding_hash"] == "H1"    # ...and stamped fresh
    assert items["fresh"]["embedding"] == [2.0]        # reused: hash matches
    assert items["legacy"]["embedding"] == [3.0]       # reused: no marker -> assumed fresh
    assert items["legacy"]["embedding_hash"] is None   # legacy hash preserved, not stamped


@pytest.mark.asyncio
async def test_index_summaries_skips_nodes_whose_fresh_vector_is_already_in_this_project() -> None:
    """INV-5 oracle: a same-project update must not re-write a vector that is already there.

    ``get_stored_summaries_light`` hands over ``has_embedding`` instead of the vector — the
    node's embedding never left the database, because it does not need to move. Writing it
    back would produce byte-identical properties at the price of a Bolt round trip and a
    vector-index update per node; on an incremental run that was the most expensive thing
    this method did. So the node is skipped outright, and the count reports what was
    actually written.

    Fails on the old code, which re-stored every reused vector.
    """
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    # Exactly TWO fresh embeds are allowed: the embedder is handed every node the run
    # decided to re-embed, and a mismatched length raises rather than passing quietly.
    indexer._summary_embedder = FakeEmbedder([[7.0], [8.0]])

    def node(nid: str) -> CodeNode:
        n = CodeNode(id=nid, name=nid, node_type=NodeType.FUNCTION,
                     file_path="a.py", line_start=1, line_end=1, summary="s")
        n.summary_input_hash = "H1"
        return n

    nodes = [node("fresh"), node("legacy"), node("stale"), node("never-embedded")]
    light = {
        "fresh":  {"summary": "s", "summary_input_hash": "H1", "summary_embedding_hash": "H1", "has_embedding": True},
        # No marker: a pre-marker vector is assumed fresh (same rule the repair query
        # applies), so it is left exactly as it is — not re-stamped, not re-embedded.
        "legacy": {"summary": "s", "summary_input_hash": "H1", "summary_embedding_hash": None, "has_embedding": True},
        # The summary was regenerated and the vector never caught up: the skip must not
        # hide it from re-embedding (audit finding 8a).
        "stale":  {"summary": "s", "summary_input_hash": "H1", "summary_embedding_hash": "H0", "has_embedding": True},
        "never-embedded": {"summary": "s", "summary_input_hash": "H1", "summary_embedding_hash": None, "has_embedding": False},
    }

    stored_count = await indexer.index_summaries(
        "fixture", nodes, stored_summaries=light, dirty_ids=set()
    )

    written = {i["node_id"]: i for i in repo.summary_embeddings}
    assert set(written) == {"stale", "never-embedded"}
    assert written["stale"]["embedding"] == [7.0] and written["stale"]["embedding_hash"] == "H1"
    assert written["never-embedded"]["embedding"] == [8.0]
    assert stored_count == len(written)


@pytest.mark.asyncio
async def test_index_summaries_still_copies_a_donors_vector_across_projects() -> None:
    """The other reuse shape must not be swept up by the skip: when the entry carries an
    actual ``summary_embedding`` the vector lives in *another* project (client-side worktree
    seeding) and this project has nothing — it has to be written, under its original hash."""
    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    indexer._summary_embedder = FakeEmbedder([])  # any embed call would blow up

    n = CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION,
                 file_path="a.py", line_start=1, line_end=1, summary="s")
    n.summary_input_hash = "H1"
    donor = {"a.py::foo": {"summary": "s", "summary_input_hash": "H1",
                           "summary_embedding": [1.0, 2.0], "summary_embedding_hash": "H1"}}

    stored_count = await indexer.index_summaries("fixture", [n], stored_summaries=donor, dirty_ids=set())

    assert stored_count == 1
    assert repo.summary_embeddings == [
        {"node_id": "a.py::foo", "embedding": [1.0, 2.0], "embedding_hash": "H1"}
    ]


@pytest.mark.asyncio
async def test_index_code_stamps_chunk_coverage_only_on_full_success() -> None:
    """INV-8 oracle (audit finding 8b): the chunk-coverage marker is stamped for a file only
    when every one of its chunks stored without an embedding failure.

    The marker (the file's checksum) is what lets a later run tell a fully-chunked file
    from one with stale/partial chunks. On an embedding failure nothing is stamped, so the
    file stays flagged for repair.
    """
    ok = FakeVectorRepository(checksums={"module.py": "cksum"})
    indexer_ok = RAGIndexer(ok, Settings(_env_file=None), chunker=FakeChunker())
    indexer_ok._code_embedder = FakeEmbedder([[1.0], [2.0]])  # both chunks embed
    node = CodeNode(id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION,
                    file_path="module.py", line_start=1, line_end=2)
    await indexer_ok.index_code("p", Path("."), [node])
    assert ok.chunks_covered == [{"file_path": "module.py", "checksum": "cksum"}]

    failed = FakeVectorRepository(checksums={"module.py": "cksum"})
    indexer_fail = RAGIndexer(failed, Settings(_env_file=None), chunker=FakeChunker())
    indexer_fail._code_embedder = FakeEmbedder([None, [2.0]])  # one chunk fails to embed
    await indexer_fail.index_code("p", Path("."), [node])
    assert failed.chunks_covered == []  # incomplete run -> file NOT marked covered


@pytest.mark.asyncio
async def test_repository_get_stored_chunk_embeddings_maps_by_chunk_id() -> None:
    conn = RecordingQueryConnection([
        {"chunk_id": "chunk:abc", "embedding": [0.1, 0.2]},
        {"chunk_id": "chunk:def", "embedding": None},
    ])
    repo = GraphRepository(conn)

    result = await repo.get_stored_chunk_embeddings("project-id")

    assert conn.params == {"project": "project-id"}
    assert result == {"chunk:abc": [0.1, 0.2]}


@pytest.mark.asyncio
async def test_index_code_reuses_stored_chunk_embeddings(tmp_path: Path) -> None:
    from inflorescence.rag.indexer import _chunk_id

    repo = FakeVectorRepository()
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=FakeChunker())
    chunks = FakeChunker().chunk_repository(tmp_path, [])
    stored_map = {_chunk_id(chunks[0]): [7.0, 7.0]}   # first chunk already embedded
    indexer._code_embedder = FakeEmbedder([[0.1, 0.2]])  # only the 2nd chunk may be embedded

    stored_count = await indexer.index_code(
        "fixture",
        tmp_path,
        [CodeNode(id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION, file_path="module.py", line_start=1, line_end=2)],
        stored_chunk_embeddings=stored_map,
    )

    assert stored_count == 2
    by_content = {c["content"]: c["embedding"] for c in repo.code_chunks}
    assert by_content["first chunk"] == [7.0, 7.0]    # reused
    assert by_content["second chunk"] == [0.1, 0.2]   # embedded


@pytest.mark.asyncio
async def test_search_code_returns_error_without_calling_repo_when_embedding_fails(monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [None])

    payload = _payload(await mcp.call_tool("search_code", {"directory": "/tmp/project", "query": "anything"}))

    assert payload["error"]["code"] == "embedding_failed"
    assert "Embedding returned None" in payload["error"]["message"]
    assert repo.code_searches == 0


@pytest.mark.asyncio
async def test_search_code_returns_entity_hits_without_content_or_embeddings(monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(
        await mcp.call_tool(
            "search_code",
            {
                "directory": "/tmp/project",
                "query": "readonly field",
                "include_paths": ["**/*.py"],
                "exclude_paths": ["tests/**"],
                "min_score": 0.2,
                "limit": 10,
                "offset": 0,
            },
        )
    )

    assert payload["items"] == [
        {
            "entity": entity("pkg/module.py::Worker.run", "Worker.run", "method", "pkg/module.py", 10, 20, "Runs the worker."),
            "score": 0.7,
            "source_type": "code",
            "matches": [
                {
                    "chunk_id": "chunk:1",
                    "chunk_kind": "function_body",
                    "line_start": 12,
                    "line_end": 18,
                    "score": 0.7,
                }
            ],
        }
    ]
    assert "content" not in str(payload)
    assert "embedding" not in str(payload)
    assert payload["meta"]["path_filters"]["include"] == ["**/*.py"]


@pytest.mark.asyncio
async def test_search_semantic_applies_min_score_without_default_path_filters(monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_semantic")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(
        await mcp.call_tool(
            "search_semantic",
            {"directory": "/tmp/project", "query": "worker", "min_score": 0.2, "limit": 10},
        )
    )

    assert [item["entity"]["id"] for item in payload["items"]] == [
        "tests/test_module.py::test_run",
        "pkg/module.py::Worker.run",
    ]
    assert payload["meta"]["path_filters"] == {
        "include": [],
        "exclude": [],
    }


@pytest.mark.asyncio
async def test_cypher_query_blocks_writes_limits_skip_and_unscoped_matches() -> None:
    mcp = FastMCP("test")
    conn = FakeConnection()
    register_cypher_tools(mcp, FakeManager(), conn)

    blocked = _payload(await mcp.call_tool("cypher_query", {"directory": "/tmp/project", "query": "CREATE (n)"}))
    unscoped = _payload(await mcp.call_tool("cypher_query", {"directory": "/tmp/project", "query": "MATCH (n) RETURN count(n) AS count"}))
    limit = _payload(
        await mcp.call_tool(
            "cypher_query",
            {"directory": "/tmp/project", "query": "MATCH (n:Code {project: $project}) RETURN n LIMIT 1"},
        )
    )
    skip = _payload(
        await mcp.call_tool(
            "cypher_query",
            {"directory": "/tmp/project", "query": "MATCH (n:Code {project: $project}) RETURN n SKIP 1"},
        )
    )

    assert blocked == {"error": {"code": "unsafe_query", "message": "Write operations are not allowed."}}
    assert unscoped["error"]["code"] == "unsafe_query"
    assert limit == {"error": {"code": "invalid_argument", "message": "Use tool-level limit/offset instead of Cypher LIMIT or SKIP."}}
    assert skip == {"error": {"code": "invalid_argument", "message": "Use tool-level limit/offset instead of Cypher LIMIT or SKIP."}}


@pytest.mark.asyncio
async def test_cypher_query_returns_paginated_envelope() -> None:
    mcp = FastMCP("test")
    conn = FakeConnection()
    conn.records = [{"count": 1}, {"count": 2}, {"count": 3}]
    register_cypher_tools(mcp, FakeManager(), conn)

    payload = _payload(
        await mcp.call_tool(
            "cypher_query",
            {
                "directory": "/tmp/project",
                "query": "MATCH (n:Code {project: $project}) RETURN n.id AS id",
                "limit": 2,
                "offset": 4,
            },
        )
    )

    assert payload == {
        "items": [{"count": 1}, {"count": 2}],
        "page": {"limit": 2, "offset": 4, "next_offset": 6, "has_more": True},
        "meta": {"project": "project-id", "query_type": "cypher"},
    }
    assert conn.params == {"project": "project-id", "limit": 3, "offset": 4}
    assert "SKIP $offset" in conn.query
    assert "LIMIT $limit" in conn.query
    # The row cap is also handed to the driver, so memory stays bounded even if the appended
    # LIMIT clause never takes effect.
    assert conn.max_records == 3


@pytest.mark.asyncio
async def test_index_directory_starts_watcher_after_successful_index() -> None:
    mcp = FastMCP("test")
    manager = FakeIndexManager()
    watcher = FakeWatcher()
    register_indexing_tools(mcp, manager, watcher)

    payload = _payload(await mcp.call_tool("index_directory", {"path": "/tmp/project"}))

    assert payload["project"] == "project-id"
    assert watcher.started == [("project-id", "/tmp/project")]


@pytest.mark.asyncio
async def test_index_directory_preview_returns_estimate_without_starting_watcher() -> None:
    mcp = FastMCP("test")
    manager = FakeIndexManager()
    watcher = FakeWatcher()
    register_indexing_tools(mcp, manager, watcher)

    payload = _payload(await mcp.call_tool("index_directory", {"path": "/tmp/project", "preview": True}))

    assert payload["preview"] is True
    assert payload["files"] == 2
    assert watcher.started == []


@pytest.mark.asyncio
async def test_list_entities_returns_paginated_envelope_with_path_filters() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "list_entities",
            {
                "directory": "/tmp/project",
                "node_types": ["function"],
                "include_paths": ["**/*.py"],
                "exclude_paths": ["**/tests/**"],
                "limit": 1,
                "offset": 0,
            },
        )
    )

    assert payload["items"] == [entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs.")]
    assert payload["page"] == page_dict(1, 0, None, False)
    assert payload["meta"]["project"] == "project-id"
    assert payload["meta"]["path_filters"] == {
        "include": ["**/*.py"],
        "exclude": ["**/tests/**"],
    }


@pytest.mark.asyncio
async def test_list_entities_applies_path_filters_before_pagination() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.pages = [
        [entity("tests/test_a.py::first", "first", "function", "tests/test_a.py", 1, 3, "Tests.")],
        [entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs.")],
    ]
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "list_entities",
            {
                "directory": "/tmp/project",
                "include_paths": ["**/*.py"],
                "exclude_paths": ["tests/**"],
                "limit": 1,
                "offset": 0,
            },
        )
    )

    assert payload["items"] == [entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs.")]
    assert repo.list_calls == [(501, 0), (501, 500)]


@pytest.mark.asyncio
async def test_list_entities_empty_include_returns_empty_page_without_fetching() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "list_entities",
            {
                "directory": "/tmp/project",
                "include_paths": [],
                "limit": 1,
                "offset": 0,
            },
        )
    )

    assert payload["items"] == []
    assert payload["page"] == page_dict(1, 0, None, False)
    assert repo.list_calls == []


@pytest.mark.asyncio
async def test_get_entity_structure_resolves_file_root_and_paginates_children() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_entity_structure",
            {"directory": "/tmp/project", "file_path": "pkg/a.py", "depth": 1, "limit": 1, "offset": 0},
        )
    )

    assert payload["root"]["id"] == "pkg/a.py"
    assert payload["items"][0]["id"] == "pkg/a.py::Worker"
    assert payload["page"] == page_dict(1, 0, 1, True)


@pytest.mark.asyncio
async def test_get_entity_structure_returns_structured_not_found_error(tmp_path: Path) -> None:
    """A path that is in neither the index nor the filesystem says so — and says both halves.

    "Structure root not found" was true and useless: it read identically for a typo and for a
    real file the index had never seen, which is the ambiguity INV-7 exists to remove.
    """
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.root = None
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_entity_structure", {"directory": str(tmp_path), "file_path": "missing.py"}
        )
    )

    assert payload["error"]["code"] == "not_found"
    message = payload["error"]["message"]
    assert "missing.py" in message and "no such file or directory on disk" in message


@pytest.mark.asyncio
async def test_get_entity_structure_says_when_the_file_exists_but_was_never_indexed(
    tmp_path: Path,
) -> None:
    """The field-report case: a real file, absent from the index, reported as plain "not found".

    An agent read that as "this file does not exist" over a 174-line module sitting on disk
    and was one step from writing it a second time. The error must name the gap and the fix.
    """
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.root = None
    register_graph_tools(mcp, FakeManager(), repo)
    (tmp_path / "wave_queue.py").write_text("QUEUE = 1\n", encoding="utf-8")

    payload = _payload(
        await mcp.call_tool(
            "get_entity_structure",
            {"directory": str(tmp_path), "file_path": "wave_queue.py"},
        )
    )

    message = payload["error"]["message"]
    assert "exists on disk but is NOT in this index" in message
    assert "index_directory" in message


@pytest.mark.asyncio
async def test_repository_upsert_nodes_stores_type_as_label_and_node_type_property() -> None:
    conn = RecordingQueryConnection([])
    repo = GraphRepository(conn)

    await repo.upsert_nodes(
        "project-id",
        [
            CodeNode(
                id="pkg/a.py::run",
                name="run",
                node_type=NodeType.FUNCTION,
                file_path="pkg/a.py",
                line_start=2,
                line_end=4,
            )
        ],
    )

    assert "UNWIND $rows AS row" in conn.write_query
    assert "MERGE (n:Code {id: row.id, project: $project})" in conn.write_query
    # Type is written both as a label (for index-backed :Method matching) and as a
    # node_type property (so raw Cypher can `RETURN n.node_type` / filter on it).
    assert "SET n:Function" in conn.write_query
    assert "REMOVE n:Function" in conn.write_query
    assert "n.node_type = row.node_type" in conn.write_query
    row = conn.write_params["rows"][0]
    assert row["node_type"] == "function"


@pytest.mark.asyncio
async def test_repository_upsert_nodes_persists_summary_input_hash() -> None:
    conn = RecordingQueryConnection([])
    repo = GraphRepository(conn)

    await repo.upsert_nodes(
        "project-id",
        [
            CodeNode(
                id="pkg/a.py::run",
                name="run",
                node_type=NodeType.FUNCTION,
                file_path="pkg/a.py",
                line_start=2,
                line_end=4,
                summary_input_hash="abc123",
            )
        ],
    )

    # The hash is written through the summary-preserving CASE (INV-1/INV-3): an empty incoming
    # summary keeps the stored value, a non-empty one lands the fresh hash.
    assert "n.summary_input_hash = CASE WHEN keep_stored THEN n.summary_input_hash ELSE row.summary_input_hash END" in conn.write_query
    assert conn.write_params["rows"][0]["summary_input_hash"] == "abc123"


@pytest.mark.asyncio
async def test_fetch_filtered_page_bounds_scan_and_reports_truncation() -> None:
    # Finding 15: when the scan hits MAX_FILTER_SCAN_ROWS with rows still unscanned,
    # the empty page must be reported as truncated, not silently as "no matches".
    # Rewritten from the old single-return version, which could not observe truncation.
    from inflorescence.tools.graph import MAX_FILTER_SCAN_ROWS, _fetch_filtered_page

    calls = {"n": 0}

    async def fetch_rows(page_limit: int, page_offset: int) -> list[dict[str, object]]:
        calls["n"] += 1
        # A full page of rows that never match the include filter -> forces scanning,
        # and always returns page_limit rows so raw rows are never "exhausted".
        return [{"file_path": f"other/x{page_offset + i}.py", "id": str(page_offset + i)} for i in range(page_limit)]

    filters = PathFilterConfig.from_paths(["wanted/**"], None)
    result, truncated = await _fetch_filtered_page(fetch_rows, filters, limit=10, offset=0)

    assert result == []  # nothing matched
    assert truncated is True  # scan stopped at the cap with more rows to go
    # Scan is capped, not unbounded: at most ceil(cap / 500) + 1 pages.
    assert calls["n"] <= (MAX_FILTER_SCAN_ROWS // 500) + 1


@pytest.mark.asyncio
async def test_fetch_filtered_page_not_truncated_when_rows_exhausted() -> None:
    # The complement: when the underlying scan runs out of rows before the cap, the
    # page is complete and truncated must be False (finding 15 — no false alarms).
    from inflorescence.tools.graph import _fetch_filtered_page

    async def fetch_rows(page_limit: int, page_offset: int) -> list[dict[str, object]]:
        # One short page -> raw rows exhausted immediately.
        if page_offset == 0:
            return [{"file_path": "wanted/a.py", "id": "a"}]
        return []

    filters = PathFilterConfig.from_paths(["wanted/**"], None)
    result, truncated = await _fetch_filtered_page(fetch_rows, filters, limit=10, offset=0)

    assert [row["id"] for row in result] == ["a"]
    assert truncated is False


class AccumulatingWriteConnection:
    """Records every execute_write call so batching can be asserted."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, query: str, params: dict[str, object]) -> list[dict[str, object]]:
        return []

    async def execute_write(self, query: str, params: dict[str, object]) -> None:
        self.writes.append((query, params))

    async def execute_write_query(self, query: str, params: dict[str, object]) -> list[dict[str, object]]:
        # Counted writes (RETURN count(*) AS written) report every submitted row as matched,
        # so batching is still asserted via self.writes while counts equal the submission.
        self.writes.append((query, params))
        return [{"written": len(params.get("rows", []))}]


@pytest.mark.asyncio
async def test_upsert_nodes_splits_into_unwind_batches_grouped_by_label() -> None:
    conn = AccumulatingWriteConnection()
    repo = GraphRepository(conn, batch_size=2)
    nodes = [
        CodeNode(id=f"m{i}.py", name=f"m{i}", node_type=NodeType.MODULE, file_path=f"m{i}.py", line_start=1, line_end=1)
        for i in range(3)
    ] + [
        CodeNode(id=f"f{i}", name=f"f{i}", node_type=NodeType.FUNCTION, file_path="m0.py", line_start=1, line_end=1)
        for i in range(2)
    ]

    count = await repo.upsert_nodes("proj", nodes)

    assert count == 5
    # 3 modules -> batches of 2 + 1; 2 functions -> one batch => 3 write calls total.
    assert len(conn.writes) == 3
    assert all(len(params["rows"]) <= 2 for _query, params in conn.writes)
    total_rows = sum(len(params["rows"]) for _query, params in conn.writes)
    assert total_rows == 5
    # Every row is a plain dict addressed by UNWIND's `row.` prefix, one query per label group.
    assert all("UNWIND $rows AS row" in query for query, _params in conn.writes)


@pytest.mark.asyncio
async def test_upsert_edges_batches_by_type_and_update_checksums_batches() -> None:
    from inflorescence.code_indexer.models import Edge, EdgeType

    conn = AccumulatingWriteConnection()
    repo = GraphRepository(conn, batch_size=2)
    edges = [Edge(source=f"a{i}", target=f"b{i}", edge_type=EdgeType.CALLS) for i in range(3)]

    edge_count = await repo.upsert_edges("proj", edges)
    checksum_count = await repo.update_checksums(
        "proj", [{"node_id": f"m{i}.py", "checksum": f"h{i}"} for i in range(3)]
    )

    assert edge_count == 3
    assert checksum_count == 3
    # 3 CALLS edges -> batches of 2 + 1; 3 checksum rows -> batches of 2 + 1.
    assert len(conn.writes) == 4
    assert all(len(params["rows"]) <= 2 for _query, params in conn.writes)


class ShortWriteConnection:
    """A write connection whose counted writes report one fewer row than submitted.

    Stands in for the DB dropping a row because an id didn't resolve to a node (an
    unresolved edge endpoint, a missing chunk owner) — the exact case audit finding 14
    says the old code reported as full success.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def execute_query(self, query: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
        return []

    async def execute_write(self, query: str, params: dict[str, object] | None = None) -> None:
        return None

    async def execute_write_query(self, query: str, params: dict[str, object]) -> list[dict[str, object]]:
        self.calls += 1
        rows = params.get("rows", [])
        return [{"written": max(0, len(rows) - 1)}]


@pytest.mark.asyncio
async def test_write_counters_report_rows_the_db_wrote_not_submitted(caplog) -> None:
    """INV-7 oracle (audit finding 14): batched writes return what the DB actually wrote and
    log ERROR on a shortfall, instead of returning ``len(batch)`` regardless.

    Fails on the old code, which returned the submitted count for all four methods.
    """
    import logging

    from inflorescence.code_indexer.models import Edge, EdgeType

    repo = GraphRepository(ShortWriteConnection(), batch_size=100)
    with caplog.at_level(logging.ERROR):
        edges = await repo.upsert_edges("p", [Edge(source=f"a{i}", target=f"b{i}", edge_type=EdgeType.CALLS) for i in range(3)])
        chunks = await repo.store_code_chunks("p", [{"chunk_id": f"c{i}", "node_id": f"n{i}"} for i in range(3)])
        summaries = await repo.store_summaries("p", [{"node_id": f"n{i}", "summary": "s", "summary_input_hash": "h"} for i in range(3)])
        checks = await repo.update_checksums("p", [{"node_id": f"n{i}", "checksum": "h"} for i in range(3)])
        embeds = await repo.store_summary_embeddings("p", [{"node_id": f"n{i}", "embedding": [0.1], "embedding_hash": "h"} for i in range(3)])

    assert (edges, chunks, summaries, checks, embeds) == (2, 2, 2, 2, 2)  # 3 submitted, 1 dropped each
    shortfalls = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(shortfalls) == 5  # every method surfaced the shortfall as ERROR


@pytest.mark.asyncio
async def test_repository_get_stored_summaries_maps_by_node_id() -> None:
    conn = RecordingQueryConnection([
        {"id": "a.py::foo", "summary": "does foo", "summary_input_hash": "h1", "summary_prompt_version": "2.ab",
         "summary_embedding": [0.1, 0.2], "summary_embedding_hash": "h1"},
        {"id": "a.py", "summary": "module a", "summary_input_hash": "h2",
         "summary_embedding": None, "summary_embedding_hash": None},
    ])
    repo = GraphRepository(conn)

    stored = await repo.get_stored_summaries("project-id")

    assert conn.params == {"project": "project-id"}
    # summary_embedding_hash comes along so the caller can tell a fresh vector from a stale
    # one (matching vs mismatched hash) when deciding whether to reuse it (finding 8a).
    assert stored["a.py::foo"] == {
        "summary": "does foo", "summary_input_hash": "h1", "summary_prompt_version": "2.ab",
        "summary_embedding": [0.1, 0.2], "summary_embedding_hash": "h1",
    }
    assert stored["a.py"]["summary_embedding"] is None


@pytest.mark.asyncio
async def test_repository_light_summaries_carry_freshness_without_the_vectors() -> None:
    """The reuse map an incremental run reads: same decisions, none of the payload.

    A summary vector is 1536 floats; a project of a few thousand nodes moved tens of
    megabytes over Bolt just to answer "which summaries can I reuse?" — which needs the
    text, the input hash, and whether a *fresh* vector already exists. The light read
    replaces the vector with that last fact, and must never hand out a vector, or a caller
    could copy one somewhere it does not belong.
    """
    conn = RecordingQueryConnection([
        {"id": "a.py::foo", "summary": "does foo", "summary_input_hash": "h1",
         "summary_embedding_hash": "h1", "has_embedding": True},
        {"id": "a.py", "summary": "module a", "summary_input_hash": "h2",
         "summary_embedding_hash": None, "has_embedding": False},
    ])
    repo = GraphRepository(conn)

    stored = await repo.get_stored_summaries_light("project-id")

    assert conn.params == {"project": "project-id"}
    # A row written before prompt versions were recorded reads as "" — unrecorded, not absent.
    assert stored["a.py::foo"] == {
        "summary": "does foo", "summary_input_hash": "h1", "summary_prompt_version": "",
        "summary_embedding_hash": "h1", "has_embedding": True,
    }
    assert stored["a.py"]["has_embedding"] is False
    assert all("summary_embedding" not in entry for entry in stored.values())
    assert "n.summary_embedding AS" not in queries.GET_STORED_SUMMARIES_LIGHT


@pytest.mark.asyncio
async def test_repository_call_facts_are_scoped_and_normalized() -> None:
    """The cross-file half of a node row, for the files an update did not touch.

    Everything else in the row is a function of the file's own bytes, so an unchanged
    checksum settles it; these three come from the whole-project resolution and can move
    under an untouched file. Nulls (a legacy row, a cloned one) must read as the empty
    defaults a freshly parsed node carries, or every such node would look "changed" forever.
    """
    conn = RecordingQueryConnection([
        {"id": "a.py::foo", "file_path": "a.py", "external_calls": ["json.dumps"],
         "unresolved_calls": [], "calls_provenance": ""},
        {"id": "a.py", "file_path": "a.py", "external_calls": None,
         "unresolved_calls": None, "calls_provenance": None},
    ])
    repo = GraphRepository(conn)

    facts = await repo.get_stored_call_facts("project-id", ["a.py"])

    assert conn.params == {"project": "project-id", "file_paths": ["a.py"]}
    assert facts["a.py::foo"]["external_calls"] == ["json.dumps"]
    assert facts["a.py"] == {
        "file_path": "a.py", "external_calls": [], "unresolved_calls": [], "calls_provenance": "",
    }
    # No files, no query: an update where every file changed asks the database nothing.
    assert await repo.get_stored_call_facts("project-id", []) == {}


@pytest.mark.asyncio
async def test_repository_list_entities_passes_pagination_and_returns_compact_refs() -> None:
    conn = RecordingQueryConnection([
        {
            "n.id": "pkg/a.py::run",
            "n.name": "run",
            "n.labels": ["Code", "Function"],
            "n.file_path": "pkg/a.py",
            "n.line_start": 2,
            "n.line_end": 4,
            "n.summary": "Runs.",
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.list_entities("project-id", node_types=["function"], limit=11, offset=7)

    assert result == [entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs.")]
    assert conn.params["project"] == "project-id"
    assert conn.params["node_types"] == ["function"]
    assert conn.params["limit"] == 11
    assert conn.params["offset"] == 7
    # Filtered on the canonical node_type property (a label lowercased is "typealias", never
    # "type_alias"); the compact ref itself still carries no node_type column.
    assert "n.node_type IN $node_types" in conn.query
    assert "node_type" not in conn.query.split("RETURN", 1)[1]
    assert "labels(n) AS `n.labels`" in conn.query


@pytest.mark.asyncio
async def test_repository_get_project_stats_counts_existing_graph() -> None:
    conn = RecordingQueryConnection([
        {
            "nodes": 12,
            "edges": 34,
            "chunks": 5,
            "summaries": 6,
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.get_project_stats("project-id")

    assert result == {"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6}
    assert conn.params == {"project": "project-id"}
    assert "MATCH (n {project: $project})" in conn.query


@pytest.mark.asyncio
async def test_repository_get_entity_context_returns_bounded_sections() -> None:
    conn = RecordingQueryConnection([
        {
            "entity": entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 1, 20, "Worker class."),
            "parent": entity("pkg/a.py", "a", "module", "pkg/a.py", 1, 30, "Module."),
            "children": [entity("pkg/a.py::Worker.run", "Worker.run", "method", "pkg/a.py", 3, 8, "Runs.")],
            "incoming": [{"relation": "CALLS", "entity": entity("pkg/b.py::call", "call", "function", "pkg/b.py", 5, 7, "Calls.")}],
            "outgoing": [{"relation": "INHERITS", "entity": entity("pkg/base.py::Base", "Base", "class", "pkg/base.py", 1, 5, "Base.")}],
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.get_entity_context("project-id", "pkg/a.py::Worker", relation_types=["CALLS"], limit=5, offset=0)

    assert result is not None
    assert result["entity"]["id"] == "pkg/a.py::Worker"
    assert result["parent"]["id"] == "pkg/a.py"
    assert result["children"][0]["id"] == "pkg/a.py::Worker.run"
    assert result["incoming"][0]["relation"] == "CALLS"
    assert result["outgoing"][0]["relation"] == "INHERITS"
    assert conn.params["relation_types"] == ["CALLS"]


@pytest.mark.asyncio
async def test_repository_get_entity_structure_formats_depth_without_touching_cypher_params() -> None:
    conn = RecordingQueryConnection([
        {
            "n.id": "pkg/a.py::Worker",
            "n.name": "Worker",
            "n.labels": ["Code", "Class"],
            "n.file_path": "pkg/a.py",
            "n.line_start": 5,
            "n.line_end": 20,
            "n.summary": "Worker.",
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.get_entity_structure("project-id", "pkg/a.py", depth=2, limit=11, offset=7)

    assert result == [entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 5, 20, "Worker.")]
    assert "[:CONTAINS*1..2]" in conn.query
    assert "{project: $project, id: $root_id}" in conn.query
    # Filtered on the canonical node_type property (a label lowercased is "typealias", never
    # "type_alias"); the compact ref itself still carries no node_type column.
    assert "n.node_type IN $node_types" in conn.query
    assert "node_type" not in conn.query.split("RETURN", 1)[1]
    assert conn.params == {"project": "project-id", "root_id": "pkg/a.py", "node_types": None, "limit": 11, "offset": 7}


@pytest.mark.asyncio
async def test_repository_search_code_vectors_derives_node_type_from_labels_only() -> None:
    conn = RecordingQueryConnection([
        {
            "id": "pkg/a.py::run",
            "name": "run",
            "labels": ["Code", "Function"],
            "node_type": "STALE",
            "file_path": "pkg/a.py",
            "line_start": 2,
            "line_end": 4,
            "summary": "Runs.",
            "score": 0.9,
            "chunk_id": "chunk:1",
            "chunk_kind": "function_body",
            "match_start_line": 2,
            "match_end_line": 4,
        }
    ])
    repo = GraphRepository(conn)

    rows = await repo.search_code_vectors("project-id", [0.1, 0.2], top_k=5)

    assert rows[0]["node_type"] == "function"
    assert rows[0]["score"] == 0.9
    assert rows[0]["chunk_id"] == "chunk:1"
    assert rows[0]["match_start_line"] == 2


@pytest.mark.asyncio
async def test_repository_search_summary_text_returns_compact_refs_with_score() -> None:
    conn = RecordingQueryConnection([
        {
            "id": "pkg/a.py::run", "name": "run", "labels": ["Code", "Function"],
            "file_path": "pkg/a.py", "line_start": 2, "line_end": 4,
            "summary": "Runs.", "score": 3.5,
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.search_summary_text("project-id", "runs", top_k=25)

    assert result == [{**entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."), "score": 3.5}]
    assert conn.params == {"query": "runs", "fetch": 1000, "project": "project-id"}
    assert "text_search.search_all('summary_text'" in conn.query


@pytest.mark.asyncio
async def test_repository_search_symbol_text_returns_compact_refs_with_score() -> None:
    conn = RecordingQueryConnection([
        {
            "id": "pkg/a.py::run", "name": "run", "labels": ["Code", "Function"],
            "file_path": "pkg/a.py", "line_start": 2, "line_end": 4,
            "summary": "Runs.", "score": 8.0,
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.search_symbol_text("project-id", "run", top_k=25)

    assert result == [{**entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."), "score": 8.0}]
    assert conn.params == {"query": "run", "fetch": 1000, "project": "project-id"}
    assert "text_search.search_all('code_symbols_text'" in conn.query


@pytest.mark.asyncio
async def test_repository_search_code_text_joins_chunks_to_entities_with_match_lines() -> None:
    conn = RecordingQueryConnection([
        {
            "id": "pkg/a.py::run", "name": "run", "labels": ["Code", "Function"],
            "file_path": "pkg/a.py", "line_start": 2, "line_end": 4, "summary": "Runs.",
            "score": 2.0, "chunk_id": "chunk:1", "chunk_kind": "function_body",
            "match_start_line": 2, "match_end_line": 3,
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.search_code_text("project-id", "runs", top_k=25)

    assert result == [{
        **entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."),
        "score": 2.0, "chunk_id": "chunk:1", "chunk_kind": "function_body",
        "match_start_line": 2, "match_end_line": 3,
    }]
    assert conn.params == {"query": "runs", "fetch": 1000, "project": "project-id"}
    assert "MATCH (code:Code {id: node.node_id, project: $project})" in conn.query


@pytest.mark.asyncio
async def test_repository_get_structure_root_serializes_root_prefixed_record() -> None:
    conn = RecordingQueryConnection([
        {
            "root.id": "pkg/a.py",
            "root.name": "a",
            "root.labels": ["Code", "Module"],
            "root.node_type": "STALE",
            "root.file_path": "pkg/a.py",
            "root.line_start": 1,
            "root.line_end": 30,
            "root.summary": "Module a.",
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.get_structure_root("project-id", node_id="pkg/a.py")

    assert result == entity("pkg/a.py", "a", "module", "pkg/a.py", 1, 30, "Module a.")


@pytest.mark.asyncio
async def test_repository_get_related_entities_serializes_related_prefixed_record() -> None:
    conn = RecordingQueryConnection([
        {
            "relation": "CALLS",
            "direction": "outgoing",
            "related.id": "pkg/b.py::helper",
            "related.name": "helper",
            "related.labels": ["Code", "Function"],
            "related.node_type": "STALE",
            "related.file_path": "pkg/b.py",
            "related.line_start": 5,
            "related.line_end": 9,
            "related.summary": "Helper function.",
        }
    ])
    repo = GraphRepository(conn)

    result = await repo.get_related_entities("project-id", "pkg/a.py::Worker")

    assert result[0]["relation"] == "CALLS"
    assert result[0]["direction"] == "outgoing"
    assert result[0]["entity"] == entity(
        "pkg/b.py::helper", "helper", "function", "pkg/b.py", 5, 9, "Helper function."
    )


class FakeChunker:
    def chunk_repository(
        self, root: Path, nodes: list[CodeNode], unreadable_out: set[str] | None = None
    ) -> list[Chunk]:
        return [
            Chunk(
                content="first chunk",
                node_id="module.py::alpha",
                file_path="module.py",
                start_line=1,
                end_line=1,
                language="python",
                chunk_index=0,
                chunk_kind="function_body",
            ),
            Chunk(
                content="second chunk",
                node_id="module.py::alpha",
                file_path="module.py",
                start_line=2,
                end_line=2,
                language="python",
                chunk_index=1,
                chunk_kind="function_body",
            ),
        ]

    def chunk_file(self, file_path: Path, root: Path) -> list[Chunk]:
        return self.chunk_repository(root, [])


def _payload(result):
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def _closure_cell_with_attr(fn, attr: str):
    for cell in fn.__closure__ or []:
        value = cell.cell_contents
        if hasattr(value, attr):
            return value
    raise AssertionError(f"closure cell with {attr!r} not found")


def page_dict(limit: int, offset: int, next_offset: int | None, has_more: bool) -> dict[str, object]:
    return {"limit": limit, "offset": offset, "next_offset": next_offset, "has_more": has_more}


def entity(
    node_id: str = "pkg/module.py::Worker.run",
    name: str = "Worker.run",
    node_type: str = "method",
    file_path: str = "pkg/module.py",
    line_start: int = 10,
    line_end: int = 20,
    summary: str = "Runs the worker.",
) -> dict[str, object]:
    return {
        "id": node_id,
        "name": name,
        "node_type": node_type,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "summary": summary,
    }


# The `watcher` line every freshness block carries for an ordinary indexed project: nothing
# is following it, so the index only moves when someone re-indexes it. Its exact wording is
# pinned by test_freshness_block_names_who_keeps_the_index_current; the other tests just need
# the block to match.
UNWATCHED = (
    "none — this index changes only when something re-indexes it; follow this tree with "
    "`inflorescence watch add <path>`"
)


class StoredChecksums:
    """The ``get_checksums`` every entity-returning tool calls once, for freshness.

    Empty by default, which lands every path in ``index_freshness.unknown`` — the honest
    answer for a fake project whose files were never really indexed, and the one status
    that puts no flag on an item, so the tool tests above keep asserting bare entity refs.
    Tests that care set ``repo.checksums``; ``checksum_calls`` guards the one-call budget.

    Also serves the watch state the same block reports (whether a daemon is keeping this
    index current): unwatched by default, which is what an ordinary indexed project is.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.checksums: dict[str, str] = {}
        self.checksum_calls = 0
        self.watch_state: dict[str, object] = {"watched": False, "live": False, "holder": None}
        self.watch_state_calls = 0

    async def get_checksums(self, project: str) -> dict[str, str]:
        self.checksum_calls += 1
        return dict(self.checksums)

    async def get_watch_state(self, project: str) -> dict[str, object]:
        self.watch_state_calls += 1
        return dict(self.watch_state)

    async def get_indexed_head(self, project: str) -> str | None:
        return None

    # The tools ask the repository for a project's hits through the widening entry points;
    # these fakes answer them from the per-field methods each test fake already defines, so a
    # test states *what the index holds* once and every tool reads it the same way.
    # A fake cannot widen its window, so it reports what the real repository would report at
    # its ceiling: a window that came back full while the filters left the request unfilled is
    # an incomplete answer; anything else is the whole answer.
    async def vector_hits(self, kind, project, embedding, needed, keep=lambda rows: rows):  # noqa: ANN001, ANN201
        method = self.search_code_vectors if kind == "code" else self.search_summary_vectors
        raw = [dict(row) for row in await method(project, embedding, needed)]
        kept = keep(raw)
        return ProjectHits(kept[:needed], complete=not (len(raw) >= needed and len(kept) < needed))

    async def text_hits(self, kind, project, query, needed, keep=lambda rows: rows, fuzzy_distance=0):  # noqa: ANN001, ANN201
        if fuzzy_distance:
            return ProjectHits([], complete=True)
        method = {
            "summary": self.search_summary_text,
            "symbol": self.search_symbol_text,
            "code": self.search_code_text,
        }[kind]
        raw = [dict(row) for row in await method(project, query, needed)]
        kept = keep(raw)
        return ProjectHits(kept[:needed], complete=not (len(raw) >= needed and len(kept) < needed))


class FakeEmbedder:
    def __init__(self, embeddings: list[list[float] | None]) -> None:
        self.model = "fake-model"
        self.embeddings = embeddings

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        return self.embeddings


class FakeVectorRepository:
    def __init__(self, chunk_ids: set[str] | None = None, checksums: dict[str, str] | None = None) -> None:
        self.code_chunks: list[dict[str, object]] = []
        self.summary_embeddings: list[dict[str, object]] = []
        self.stored_chunk_ids = chunk_ids or set()
        self.deleted_chunk_ids: list[str] = []
        self.chunk_index: dict[str, str] = {}  # chunk_id -> file_path, for scoped pruning
        self.checksums = checksums or {}        # file_path -> md5, for chunk-coverage stamping
        self.chunks_covered: list[dict[str, object]] = []

    async def get_stored_chunk_facts(self, project: str, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        """This fake never reports a chunk as already stored: these tests are about embedding
        failure, pruning, coverage and windowing, none of which involve the skip. The skip is
        covered in tests/test_chunk_reuse.py against a store that models it."""
        return {}

    async def get_stored_chunk_embeddings_for_ids(self, project: str, chunk_ids: list[str]) -> dict[str, list[float]]:
        return {}

    async def store_code_chunks(self, project: str, chunks: list[dict[str, object]]) -> int:
        self.code_chunks.extend(chunks)
        return len(chunks)

    async def get_checksums(self, project: str) -> dict[str, str]:
        return dict(self.checksums)

    async def mark_files_chunks_covered(self, project: str, items: list[dict[str, object]]) -> int:
        self.chunks_covered.extend(items)
        return len(items)

    async def store_summary_embeddings(self, project: str, items: list[dict[str, object]]) -> int:
        self.summary_embeddings.extend(items)
        return len(items)

    async def get_chunk_ids(self, project: str) -> set[str]:
        return set(self.stored_chunk_ids)

    async def get_chunk_index(self, project: str) -> dict[str, str]:
        return dict(self.chunk_index)

    async def delete_chunks_by_ids(self, project: str, ids: list[str]) -> int:
        self.deleted_chunk_ids.extend(ids)
        return len(ids)


class PerTextEmbedder:
    """Embeds each text independently so multiple windowed calls stay aligned."""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        self.calls += 1
        return [[float(len(t))] for t in texts]


class CountingVectorRepository(FakeVectorRepository):
    def __init__(self) -> None:
        super().__init__()
        self.store_calls = 0

    async def store_code_chunks(self, project: str, chunks: list[dict[str, object]]) -> int:
        self.store_calls += 1
        return await super().store_code_chunks(project, chunks)


@pytest.mark.asyncio
async def test_index_code_streams_windows_persisting_each(tmp_path: Path) -> None:
    repo = CountingVectorRepository()
    settings = Settings(_env_file=None)
    settings.rag_index_batch_size = 1  # force one window per chunk
    indexer = RAGIndexer(repo, settings, chunker=FakeChunker())
    embedder = PerTextEmbedder()
    indexer._code_embedder = embedder

    stored_count = await indexer.index_code(
        "fixture",
        tmp_path,
        [CodeNode(id="module.py::alpha", name="alpha", node_type=NodeType.FUNCTION, file_path="module.py", line_start=1, line_end=2)],
    )

    # Two chunks, window=1 -> two independent embed calls and two incremental store calls.
    assert stored_count == 2
    assert embedder.calls == 2
    assert repo.store_calls == 2


class FakeManager:
    def get_project(self, directory: str) -> str:
        return "project-id"

    async def project_is_indexed(self, project: str) -> bool:
        # Fakes stand in for an indexed project with seeded data; the T5 guard only
        # short-circuits genuinely unindexed projects, which these tests never exercise.
        return True

    async def ensure_indexed_for_query(self, directory: str) -> tuple[str, str | None]:
        """What the read tools actually call: (project, reason-it-cannot-be-queried).

        Mirrors the real manager for the fakes' case — an indexed project, nothing to
        auto-index. Subclasses below stand in for the unindexed ones.
        """
        project = self.get_project(directory)
        if await self.project_is_indexed(project):
            return project, None
        return project, "index it with index_directory first"


class FakeIndexManager(FakeManager):
    async def index_directory(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_cost_usd: float | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, object]:
        return {"project": self.get_project(path), "directory": path}

    async def preview_directory(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, object]:
        return {"project": self.get_project(path), "directory": path, "preview": True, "files": 2, "estimated_tokens": 10}


class FakeWatcher:
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []

    def start(self, project: str, path: str) -> None:
        self.started.append((project, path))


class FakeGraphRepositoryV2(StoredChecksums):
    def __init__(self) -> None:
        super().__init__()
        self.root = entity("pkg/a.py", "a", "module", "pkg/a.py", 1, 30, "Module.")
        self.pages: list[list[dict[str, object]]] | None = None
        self.list_calls: list[tuple[int, int]] = []

    async def list_entities(self, project: str, node_types: list[str] | None, limit: int, offset: int) -> list[dict[str, object]]:
        self.list_calls.append((limit, offset))
        if self.pages is not None:
            page_index = offset // 500
            rows = self.pages[page_index] if page_index < len(self.pages) else []
            if page_index < len(self.pages) - 1:
                return [*rows, *({"__sentinel__": True, "file_path": ""} for _ in range(limit - len(rows)))]
            return rows
        return [
            entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."),
            entity("tests/test_a.py::test_run", "test_run", "function", "tests/test_a.py", 1, 3, "Tests."),
        ]

    async def get_structure_root(self, project: str, node_id: str | None = None, file_path: str | None = None) -> dict[str, object] | None:
        return self.root

    async def get_entity_structure(
        self,
        project: str,
        root_id: str,
        depth: int,
        node_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        return [
            entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 5, 20, "Worker."),
            entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."),
        ]

    async def get_entity_context(
        self,
        project: str,
        node_id: str,
        relation_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict[str, object] | None:
        return {
            "entity": entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 5, 20, "Worker."),
            "parent": self.root,
            "children": [entity("pkg/a.py::Worker.run", "Worker.run", "method", "pkg/a.py", 7, 9, "Runs.")],
            "incoming": [],
            "outgoing": [],
        }

    async def get_related_entities(
        self,
        project: str,
        node_id: str,
        direction: str,
        relation_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        return [{"relation": "CONTAINS", "direction": "outgoing", "entity": entity()}]

    async def search_entities(
        self,
        project: str,
        query: str,
        node_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        return [entity()]


class FakeContextRepository(StoredChecksums, FakeManager):
    async def get_entity_context(
        self,
        project: str,
        node_id: str,
        relation_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> dict[str, object] | None:
        return {
            "entity": entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 1, 20, "Worker."),
            "parent": entity("pkg/a.py", "a", "module", "pkg/a.py", 1, 30, "Module."),
            "children": [
                entity("pkg/a.py::Worker.run", "Worker.run", "method", "pkg/a.py", 3, 8, "Runs."),
                entity("pkg/a.py::Worker.stop", "Worker.stop", "method", "pkg/a.py", 9, 12, "Stops."),
            ],
            "incoming": [
                {"relation": "CALLS", "entity": entity("pkg/b.py::call", "call", "function", "pkg/b.py", 5, 7, "Calls.")}
            ],
            "outgoing": [],
        }


class FakeSearchRepository(StoredChecksums):
    def __init__(self) -> None:
        super().__init__()
        self.code_searches = 0

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        self.code_searches += 1
        return [
            {
                **entity(),
                "score": 0.7,
                "chunk_id": "chunk:1",
                "chunk_kind": "function_body",
                "match_start_line": 12,
                "match_end_line": 18,
                "content": "internal",
                "embedding": [0.1],
            },
            {
                **entity("tests/test_module.py::test_run", "test_run", "function", "tests/test_module.py", 1, 3, "Tests."),
                "score": 0.8,
                "chunk_id": "chunk:test",
                "chunk_kind": "function_body",
                "match_start_line": 1,
                "match_end_line": 3,
            },
            {
                **entity("pkg/low.py::low", "low", "function", "pkg/low.py", 1, 2, "Low."),
                "score": 0.1,
                "chunk_id": "chunk:low",
                "chunk_kind": "function_body",
                "match_start_line": 1,
                "match_end_line": 2,
            },
        ]

    async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return [
            {**entity(), "score": 0.6},
            {**entity("tests/test_module.py::test_run", "test_run", "function", "tests/test_module.py", 1, 3, "Tests."), "score": 0.9},
            {**entity("pkg/low.py::low", "low", "function", "pkg/low.py", 1, 2, "Low."), "score": 0.1},
        ]


class FakeTextSearchRepository(StoredChecksums):
    def __init__(self) -> None:
        super().__init__()
        self.summary_calls = 0
        self.symbol_calls = 0
        self.code_calls = 0
        self.received_queries: list[str] = []
        self.summary_rows: list[dict[str, object]] = []
        self.symbol_rows: list[dict[str, object]] = []
        self.code_rows: list[dict[str, object]] = []

    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        self.summary_calls += 1
        self.received_queries.append(query)
        return self.summary_rows

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        self.symbol_calls += 1
        self.received_queries.append(query)
        return self.symbol_rows

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        self.code_calls += 1
        self.received_queries.append(query)
        return self.code_rows


class FakeFailingTextRepository(StoredChecksums):
    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        raise RuntimeError("text index summary_text does not exist")

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []


class FakeConnectionErrorTextRepository(StoredChecksums):
    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        raise RuntimeError("Connection to Memgraph was lost")

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []


class RecordingQueryConnection:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.query = ""
        self.params: dict[str, object] = {}
        self.write_query = ""
        self.write_params: dict[str, object] = {}

    async def execute_query(self, query: str, params: dict[str, object]) -> list[dict[str, object]]:
        if "AS scanned" in query:
            # The search window probe: this fake's index holds exactly `records`, so a window
            # of any size comes back with that many rows and the search stops widening.
            return [{"scanned": len(self.records)}]
        self.query = query
        self.params = params
        return self.records

    async def execute_write(self, query: str, params: dict[str, object]) -> None:
        self.write_query = query
        self.write_params = params


class RecordingSchemaConnection:
    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, query: str, params: dict[str, object] | None = None) -> list[object]:
        # Serves both `SHOW INDEX INFO;` and the node_type backfill probe; empty means
        # "no existing indexes / nothing to backfill", so setup_schema does the full pass.
        assert query == "SHOW INDEX INFO;" or "n.node_type IS NULL" in query
        return []

    async def execute_write(self, query: str, params: dict[str, object] | None = None) -> None:
        self.writes.append((query, params or {}))


class FakeRecord:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()


class FakeConnection:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = [{"count": 1}]
        self.params: dict[str, object] = {}
        self.query = ""
        self.max_records: int | None = None

    async def execute_query(
        self,
        query: str,
        params: dict[str, object],
        max_records: int | None = None,
    ) -> list[dict[str, object] | FakeRecord]:
        self.query = query
        self.params = params
        self.max_records = max_records
        if self.records != [{"count": 1}]:
            # Honour the cap the way the real driver does, so a caller that relies on it
            # instead of on Cypher LIMIT is exercised rather than merely accepted.
            return self.records if max_records is None else self.records[:max_records]
        if "child.id" in query:
            return [
                FakeRecord(
                    {
                        "child.id": "validator.py",
                        "child.name": "validator",
                        "child.labels": ["Code", "Module"],
                        "child.summary": "summary",
                        "child.file_path": "validator.py",
                    }
                )
            ]
        return self.records


class RaisingConnection:
    async def execute_query(
        self,
        query: str,
        params: dict[str, object],
        max_records: int | None = None,
    ) -> list[dict[str, object]]:
        raise RuntimeError("boom")


def test_code_hits_aggregates_multiple_chunks_into_one_entity() -> None:
    rows = [
        {**entity(), "score": 0.5, "chunk_id": "chunk:1", "chunk_kind": "function_body", "match_start_line": 10, "match_end_line": 12},
        {**entity(), "score": 0.9, "chunk_id": "chunk:2", "chunk_kind": "function_body", "match_start_line": 14, "match_end_line": 18},
        {**entity("pkg/other.py::other", "other", "function", "pkg/other.py", 1, 2, "Other."), "score": 0.3, "chunk_id": "chunk:3", "chunk_kind": "function_body", "match_start_line": 1, "match_end_line": 2},
    ]

    hits = _code_hits(rows, min_score=0.0)

    assert [hit["entity"]["id"] for hit in hits] == ["pkg/module.py::Worker.run", "pkg/other.py::other"]
    assert hits[0]["score"] == 0.9
    assert hits[0]["source_type"] == "code"
    assert [match["chunk_id"] for match in hits[0]["matches"]] == ["chunk:1", "chunk:2"]
    assert hits[1]["matches"] == [{"chunk_id": "chunk:3", "chunk_kind": "function_body", "line_start": 1, "line_end": 2, "score": 0.3}]


@pytest.mark.asyncio
async def test_search_hybrid_merges_and_dedupes_code_and_summary_hits(monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_hybrid")
    for cell in tool.fn.__closure__ or []:
        value = cell.cell_contents
        if hasattr(value, "model") and hasattr(value, "embed"):
            monkeypatch.setattr(value, "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(
        await mcp.call_tool(
            "search_hybrid",
            {"directory": "/tmp/project", "query": "worker", "min_score": 0.2, "limit": 10},
        )
    )

    assert [item["entity"]["id"] for item in payload["items"]] == [
        "tests/test_module.py::test_run",
        "pkg/module.py::Worker.run",
    ]
    assert payload["items"][0]["score"] == 0.9
    assert payload["items"][0]["source_type"] == "hybrid"
    assert [match["chunk_id"] for match in payload["items"][0]["matches"]] == ["chunk:test"]
    assert payload["items"][1]["score"] == 0.7
    assert payload["items"][1]["source_type"] == "hybrid"
    assert [match["chunk_id"] for match in payload["items"][1]["matches"]] == ["chunk:1"]
    assert "content" not in str(payload)


@pytest.mark.asyncio
async def test_search_text_ranks_across_summary_symbol_and_code_fields() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.summary_rows = [{**entity("pkg/a.py::a", "a", "function", "pkg/a.py", 1, 2, "Alpha."), "score": 2.0}]
    repo.symbol_rows = [{**entity("pkg/b.py::b", "b", "function", "pkg/b.py", 1, 2, "Beta."), "score": 5.0}]
    repo.code_rows = [{
        **entity("pkg/c.py::c", "c", "function", "pkg/c.py", 1, 2, "Gamma."),
        "score": 3.0, "chunk_id": "chunk:c", "chunk_kind": "function_body",
        "match_start_line": 1, "match_end_line": 2,
    }]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run"}))

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/b.py::b", "pkg/c.py::c", "pkg/a.py::a"]
    assert [item["source_type"] for item in payload["items"]] == ["name", "code", "summary"]
    assert repo.summary_calls == 1 and repo.symbol_calls == 1 and repo.code_calls == 1
    assert payload["meta"]["query"] == "run"


@pytest.mark.asyncio
async def test_search_text_aggregates_multiple_chunks_to_one_entity() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.code_rows = [
        {
            **entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 20, "Runs."),
            "score": 1.5, "chunk_id": "chunk:1", "chunk_kind": "function_body",
            "match_start_line": 2, "match_end_line": 6,
        },
        {
            **entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 20, "Runs."),
            "score": 4.0, "chunk_id": "chunk:2", "chunk_kind": "function_body",
            "match_start_line": 10, "match_end_line": 14,
        },
    ]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run"}))

    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["entity"]["id"] == "pkg/a.py::run"
    assert item["score"] == 4.0
    assert item["source_type"] == "code"
    assert {m["chunk_id"] for m in item["matches"]} == {"chunk:1", "chunk:2"}


@pytest.mark.asyncio
async def test_search_text_dedups_entity_across_summary_and_code_fields() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.summary_rows = [{**entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 20, "Runs."), "score": 2.0}]
    repo.code_rows = [{
        **entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 20, "Runs."),
        "score": 6.0, "chunk_id": "chunk:1", "chunk_kind": "function_body",
        "match_start_line": 3, "match_end_line": 8,
    }]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run"}))

    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["entity"]["id"] == "pkg/a.py::run"
    assert item["score"] == 6.0
    assert item["source_type"] == "text"
    assert [m["chunk_id"] for m in item["matches"]] == ["chunk:1"]


@pytest.mark.asyncio
async def test_search_text_paginates_merged_hits() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.summary_rows = [
        {**entity("pkg/a.py::a", "a", "function", "pkg/a.py", 1, 2, "A."), "score": 3.0},
        {**entity("pkg/b.py::b", "b", "function", "pkg/b.py", 1, 2, "B."), "score": 2.0},
        {**entity("pkg/c.py::c", "c", "function", "pkg/c.py", 1, 2, "C."), "score": 1.0},
    ]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "x", "limit": 2, "offset": 0}))

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/a.py::a", "pkg/b.py::b"]
    assert payload["page"] == page_dict(2, 0, 2, True)


@pytest.mark.asyncio
async def test_search_text_returns_entity_refs_without_content_or_embeddings() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.code_rows = [{
        **entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 20, "Runs."),
        "score": 4.0, "chunk_id": "chunk:1", "chunk_kind": "function_body",
        "match_start_line": 3, "match_end_line": 8,
        "content": "def run(): raise ValueError('boom')", "embedding": [0.1, 0.2],
    }]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "boom"}))

    assert payload["items"][0]["entity"]["id"] == "pkg/a.py::run"
    assert "content" not in str(payload)
    assert "embedding" not in str(payload)


@pytest.mark.asyncio
async def test_search_text_degrades_gracefully_when_text_index_missing() -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeFailingTextRepository(), Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run"}))

    assert payload["error"]["code"] == "text_search_unavailable"
    assert "MAGE" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_search_text_sanitizes_tantivy_metacharacters_before_querying() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    _payload(await mcp.call_tool(
        "search_text",
        {"directory": "/tmp/project", "query": "ValueError: invalid literal (arr[0])"},
    ))

    # Lower-cased too: the index lower-cases every token anyway, and an upper-case AND/OR/NOT
    # in pasted text must not become a boolean operator. (Later entries are the per-word
    # probes that tell an unmatched word from a filtered one.)
    assert repo.received_queries[:3] == ["valueerror invalid literal arr 0"] * 3


@pytest.mark.asyncio
async def test_search_text_returns_empty_page_when_query_is_only_metacharacters() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": ":: [] ()"}))

    assert payload["items"] == []
    assert repo.summary_calls == 0 and repo.symbol_calls == 0 and repo.code_calls == 0
    assert payload["meta"]["query"] == ":: [] ()"


@pytest.mark.asyncio
async def test_search_text_reports_generic_failure_without_blaming_mage() -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeConnectionErrorTextRepository(), Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run"}))

    assert payload["error"]["code"] == "text_search_failed"
    assert "MAGE" not in payload["error"]["message"]


@pytest.mark.asyncio
async def test_search_code_paginates_with_offset_and_reports_has_more(monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(
        await mcp.call_tool(
            "search_code",
            {"directory": "/tmp/project", "query": "worker", "limit": 1, "offset": 1},
        )
    )

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/module.py::Worker.run"]
    assert payload["page"] == page_dict(1, 1, 2, True)


@pytest.mark.asyncio
async def test_get_entity_context_tool_paginates_and_trims_sections() -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeContextRepository(), FakeContextRepository())

    payload = _payload(
        await mcp.call_tool(
            "get_entity_context",
            {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker", "relation_types": ["CALLS"], "limit": 1, "offset": 0},
        )
    )

    assert payload["entity"]["id"] == "pkg/a.py::Worker"
    assert payload["parent"]["id"] == "pkg/a.py"
    assert payload["sections"]["children"]["items"] == [
        entity("pkg/a.py::Worker.run", "Worker.run", "method", "pkg/a.py", 3, 8, "Runs.")
    ]
    assert payload["sections"]["children"]["page"] == page_dict(1, 0, 1, True)
    assert payload["sections"]["incoming"]["items"] == [
        {"relation": "CALLS", "entity": entity("pkg/b.py::call", "call", "function", "pkg/b.py", 5, 7, "Calls.")}
    ]
    assert payload["sections"]["incoming"]["page"] == page_dict(1, 0, None, False)
    assert payload["sections"]["outgoing"]["items"] == []
    assert payload["meta"] == {
        "project": "project-id",
        "node_id": "pkg/a.py::Worker",
        "relation_types": ["CALLS"],
        "answered_from": str(Path("/tmp/project").resolve()),
        # /tmp/project holds no such files, and the fake stored no checksums: every path
        # on the page is 'unknown', which is meta-only and flags no item.
        "index_freshness": {
            "checked": 2,
            "stale": [],
            "missing": [],
            "unknown": ["pkg/a.py", "pkg/b.py"],
            "watcher": UNWATCHED,
        },
    }


@pytest.mark.asyncio
async def test_get_related_entities_tool_returns_compact_related_refs() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_related_entities",
            {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker", "direction": "both", "limit": 50, "offset": 0},
        )
    )

    assert payload["items"] == [{"relation": "CONTAINS", "direction": "outgoing", "entity": entity()}]
    assert payload["page"] == page_dict(50, 0, None, False)
    assert payload["meta"]["project"] == "project-id"
    assert payload["meta"]["direction"] == "both"
    assert "content" not in str(payload)


class RecordingDirectionRepository(FakeGraphRepositoryV2):
    """Records the direction value the tool actually passes to the repository."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_directions: list[str] = []

    async def get_related_entities(
        self,
        project: str,
        node_id: str,
        direction: str,
        relation_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        self.seen_directions.append(direction)
        return [{"relation": "CONTAINS", "direction": "outgoing", "entity": entity()}]


@pytest.mark.asyncio
async def test_get_related_entities_tool_normalizes_direction_aliases_case_insensitively() -> None:
    mcp = FastMCP("test")
    repo = RecordingDirectionRepository()
    register_graph_tools(mcp, FakeManager(), repo)

    cases = [
        ("both", "both"),
        ("BOTH", "both"),
        ("outgoing", "outgoing"),
        ("out", "outgoing"),
        ("Out", "outgoing"),
        ("incoming", "incoming"),
        ("in", "incoming"),
        ("IN", "incoming"),
        ("INCOMING", "incoming"),
    ]

    for raw_direction, expected in cases:
        payload = _payload(
            await mcp.call_tool(
                "get_related_entities",
                {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker", "direction": raw_direction},
            )
        )
        assert "error" not in payload, f"direction={raw_direction!r} unexpectedly errored: {payload}"
        assert payload["meta"]["direction"] == expected

    assert repo.seen_directions == [expected for _, expected in cases]


@pytest.mark.asyncio
async def test_get_related_entities_tool_rejects_invalid_direction_without_querying_db() -> None:
    mcp = FastMCP("test")
    repo = RecordingDirectionRepository()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_related_entities",
            {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker", "direction": "sideways"},
        )
    )

    assert payload["error"]["code"] == "invalid_direction"
    assert "sideways" in payload["error"]["message"]
    assert "both" in payload["error"]["message"]
    assert "incoming" in payload["error"]["message"]
    assert "outgoing" in payload["error"]["message"]
    # Bug regression: an invalid direction must not silently reach the DB and
    # come back as an indistinguishable empty page (INV-7).
    assert repo.seen_directions == []


@pytest.mark.asyncio
async def test_search_entities_tool_returns_paginated_entity_refs() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "search_entities",
            {"directory": "/tmp/project", "query": "Worker", "node_types": ["method"], "limit": 20, "offset": 0},
        )
    )

    assert payload["items"] == [entity()]
    assert payload["page"] == page_dict(20, 0, None, False)
    assert payload["meta"]["query"] == "Worker"
    assert payload["meta"]["node_types"] == ["method"]


class EmptySearchRepository(FakeGraphRepositoryV2):
    """Finds nothing — the shape of the field report this test is about."""

    async def search_entities(
        self,
        project: str,
        query: str,
        node_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        return []


class CoverageManager(FakeManager):
    """A manager that can answer "what is on disk that the index does not hold"."""

    async def scan_uncovered(
        self, directory: str, indexed_paths: object, scope: object = None
    ) -> dict[str, object]:
        from inflorescence.code_indexer.graph_builder import GraphBuilder
        from inflorescence.code_indexer.models import IndexerConfig

        builder = GraphBuilder(
            repo=None,
            llm=None,
            config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
        )
        return builder.scan_uncovered(Path(directory), list(indexed_paths), scope)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_empty_search_says_what_the_index_does_not_cover(tmp_path: Path) -> None:
    """End to end: the tool that reported "nothing named wave_queue exists" now names the file.

    The whole path is under test — the page comes back empty, the freshness block is clean,
    and the coverage block is what keeps the response from reading as proof of absence.
    """
    (tmp_path / "wave_queue.py").write_text("QUEUE = []\n", encoding="utf-8")
    mcp = FastMCP("test")
    register_graph_tools(mcp, CoverageManager(), EmptySearchRepository())

    payload = _payload(
        await mcp.call_tool(
            "search_entities", {"directory": str(tmp_path), "query": "wave_queue", "limit": 10}
        )
    )

    assert payload["items"] == []
    coverage = payload["meta"]["index_coverage"]
    assert coverage["not_indexed"] == ["wave_queue.py"]
    assert "may be a coverage gap" in coverage["note"]


LONG_SUMMARY = "A long multi-paragraph summary. " * 20  # 640 chars, over the preview bound


class LongSummaryGraphRepository(FakeGraphRepositoryV2):
    """Every returned entity carries a summary longer than the navigation preview."""

    def __init__(self) -> None:
        super().__init__()
        self.root = entity("pkg/a.py", "a", "module", "pkg/a.py", 1, 30, LONG_SUMMARY)

    async def list_entities(self, project, node_types, limit, offset):
        return [entity(summary=LONG_SUMMARY)]

    async def get_entity_structure(self, project, root_id, depth, node_types, limit, offset):
        return [entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 5, 20, LONG_SUMMARY)]

    async def get_related_entities(self, project, node_id, direction, relation_types, limit, offset):
        return [{"relation": "CONTAINS", "direction": "outgoing", "entity": entity(summary=LONG_SUMMARY)}]

    async def search_entities(self, project, query, node_types, limit, offset):
        return [entity(summary=LONG_SUMMARY)]

    async def get_entity_context(self, project, node_id, relation_types, limit, offset):
        return {
            "entity": entity("pkg/a.py::Worker", "Worker", "class", "pkg/a.py", 5, 20, LONG_SUMMARY),
            "parent": self.root,
            "children": [entity("pkg/a.py::Worker.run", "Worker.run", "method", "pkg/a.py", 7, 9, LONG_SUMMARY)],
            "incoming": [],
            "outgoing": [],
        }


@pytest.mark.asyncio
async def test_navigation_tools_truncate_summaries_to_preview() -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), LongSummaryGraphRepository())
    preview = truncate_summary(LONG_SUMMARY, NAV_SUMMARY_MAX_CHARS)

    listed = _payload(await mcp.call_tool("list_entities", {"directory": "/tmp/project"}))
    structure = _payload(await mcp.call_tool("get_entity_structure", {"directory": "/tmp/project", "file_path": "pkg/a.py"}))
    related = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker"}))

    assert listed["items"][0]["summary"] == preview
    assert structure["items"][0]["summary"] == preview
    assert structure["root"]["summary"] == LONG_SUMMARY  # single focus entity keeps full text
    assert related["items"][0]["entity"]["summary"] == preview


@pytest.mark.asyncio
async def test_search_and_context_tools_keep_full_summaries() -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), LongSummaryGraphRepository())

    searched = _payload(await mcp.call_tool("search_entities", {"directory": "/tmp/project", "query": "Worker"}))
    context = _payload(await mcp.call_tool("get_entity_context", {"directory": "/tmp/project", "node_id": "pkg/a.py::Worker"}))

    assert searched["items"][0]["summary"] == LONG_SUMMARY
    assert context["entity"]["summary"] == LONG_SUMMARY
    assert context["sections"]["children"]["items"][0]["summary"] == LONG_SUMMARY


@pytest.mark.asyncio
async def test_cypher_query_scopes_every_match_pattern_and_blocks_global_call() -> None:
    # Invariant INV-9 (finding 12): a project-scoped MATCH does not constrain what a
    # stored procedure reads. text_search.search_all / vector_search.search read GLOBAL
    # indexes, so a CALL leaks other projects' rows. The validator must block CALL —
    # rewritten from the previous `allows_text_search` test, which encoded the leak.
    mcp = FastMCP("test")
    conn = FakeConnection()
    register_cypher_tools(mcp, FakeManager(), conn)

    two_scoped = _payload(await mcp.call_tool("cypher_query", {
        "directory": "/tmp/project",
        "query": "MATCH (a:Code {project: $project}) MATCH (b:Code {project: $project}) RETURN a.id AS id",
    }))
    one_unscoped = _payload(await mcp.call_tool("cypher_query", {
        "directory": "/tmp/project",
        "query": "MATCH (a:Code {project: $project}) MATCH (b:Code) RETURN a.id AS id",
    }))
    comma_unscoped = _payload(await mcp.call_tool("cypher_query", {
        "directory": "/tmp/project",
        "query": "MATCH (a:Code {project: $project}), (b:Code) RETURN a.id AS id",
    }))
    global_call = _payload(await mcp.call_tool("cypher_query", {
        "directory": "/tmp/project",
        "query": "MATCH (n:Code {project: $project}) CALL text_search.search_all('summary_text', 'query') YIELD node RETURN node.name",
    }))

    assert two_scoped["items"] == [{"count": 1}]
    assert one_unscoped["error"]["code"] == "unsafe_query"
    assert comma_unscoped["error"]["code"] == "unsafe_query"
    assert global_call["error"]["code"] == "unsafe_query"
    assert "CALL" in global_call["error"]["message"]


@pytest.mark.asyncio
async def test_cypher_query_returns_query_failed_on_execution_error() -> None:
    mcp = FastMCP("test")
    register_cypher_tools(mcp, FakeManager(), RaisingConnection())

    payload = _payload(await mcp.call_tool("cypher_query", {
        "directory": "/tmp/project",
        "query": "MATCH (n:Code {project: $project}) RETURN n.id AS id",
    }))

    assert payload == {"error": {"code": "query_failed", "message": "boom"}}


# ---------------------------------------------------------------------------
# G5 · search & query correctness (findings 10, 11, 15)
# ---------------------------------------------------------------------------


def test_get_entity_context_query_uses_offset_and_relation_types() -> None:
    # Finding 10: the query must actually consume the $offset and $relation_types it is
    # passed. Before the fix it sliced [0..$limit] (offset ignored -> pagination loops)
    # and never referenced $relation_types (the relation filter was silently dropped).
    q = queries.GET_ENTITY_CONTEXT
    assert "$relation_types" in q
    assert "type(incoming) IN $relation_types" in q
    assert "type(outgoing) IN $relation_types" in q
    assert "[$offset..$offset + $limit]" in q
    assert "[0..$limit]" not in q


class _SaturatingSearchRepo(StoredChecksums):
    """A repo whose vector search fills the candidate pool with rows outside any filter."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def _rows(self, top_k: int) -> list[dict[str, object]]:
        n = min(self.count, top_k)
        return [
            {
                **entity(f"src/f{i}.py::x", "x", "function", f"src/f{i}.py", 1, 2, "s"),
                "score": 0.5,
                "chunk_id": f"c{i}",
                "chunk_kind": "function_body",
                "match_start_line": 1,
                "match_end_line": 2,
            }
            for i in range(n)
        ]

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._rows(top_k)

    async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._rows(top_k)


@pytest.mark.asyncio
async def test_search_code_signals_truncation_when_filter_starves_a_full_pool(monkeypatch) -> None:
    # Finding 11: a rare path filter over a saturated candidate pool must not report a
    # false empty. The pool is full (>= candidate_k) and none of it survives the filter,
    # so the empty page is flagged truncated — "the scan stopped early", not "no matches".
    mcp = FastMCP("test")
    repo = _SaturatingSearchRepo(count=500)  # >= candidate_k (100) -> saturated
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "q", "include_paths": ["rare/**"]},
    ))

    assert payload["items"] == []
    assert payload["page"]["truncated"] is True


@pytest.mark.asyncio
async def test_search_code_does_not_signal_truncation_when_pool_not_saturated(monkeypatch) -> None:
    # Complement: when the pool did not fill the ceiling, an empty filtered result is
    # genuine exhaustion, not truncation -> no false alarm.
    mcp = FastMCP("test")
    repo = _SaturatingSearchRepo(count=5)  # < candidate_k -> not saturated
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "q", "include_paths": ["rare/**"]},
    ))

    assert payload["items"] == []
    assert "truncated" not in payload["page"]


@pytest.mark.asyncio
async def test_search_code_without_filters_never_signals_truncation(monkeypatch) -> None:
    # An unfiltered vector search returns the similarity top-k by design; a full pool is
    # the answer, not truncation. The signal must fire only when a filter can starve it.
    mcp = FastMCP("test")
    repo = _SaturatingSearchRepo(count=500)
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    tool = mcp._tool_manager.get_tool("search_code")
    monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_code", {"directory": "/tmp/project", "query": "q"}))

    assert "truncated" not in payload["page"]


@pytest.mark.asyncio
async def test_list_entities_flags_truncation_when_scan_hits_the_cap() -> None:
    # Finding 15 at the tool level: a sparse filter over a project larger than
    # MAX_FILTER_SCAN_ROWS returns a short/empty page flagged truncated, so the client
    # can tell "scan capped" from "no such entities".
    from inflorescence.tools.graph import MAX_FILTER_SCAN_ROWS

    class _NeverMatchRepo(StoredChecksums, FakeManager):
        async def list_entities(self, project, node_types, limit, offset):
            # Always a full page of non-matching rows -> the scan never exhausts.
            return [{"id": str(offset + i), "file_path": f"other/x{offset + i}.py"} for i in range(limit)]

    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), _NeverMatchRepo())

    payload = _payload(await mcp.call_tool(
        "list_entities",
        {"directory": "/tmp/project", "include_paths": ["wanted/**"], "limit": 10, "offset": 0},
    ))

    assert payload["items"] == []
    assert payload["page"]["truncated"] is True
    assert MAX_FILTER_SCAN_ROWS == 5000


# ---------------------------------------------------------------------------
# Read tools and the auto-index gate. A tool never decides what may be indexed — it asks
# the manager and either queries or reports the manager's reason (INV-7: no silent empty
# page, no raised exception where a structured error is the contract).
# ---------------------------------------------------------------------------


class _AutoIndexingManager(FakeManager):
    """Stands in for a worktree the manager indexes inline on the first query."""

    def __init__(self) -> None:
        self.ensured: list[str] = []

    async def project_is_indexed(self, project: str) -> bool:
        # Deliberately still False: a tool that consulted this instead of the ensure
        # method would refuse a query the manager just made answerable.
        return False

    async def ensure_indexed_for_query(self, directory: str) -> tuple[str, str | None]:
        self.ensured.append(directory)
        return self.get_project(directory), None


class _RefusingManager(FakeManager):
    """Stands in for a directory that cannot be queried, for the manager's stated reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def project_is_indexed(self, project: str) -> bool:
        return False

    async def ensure_indexed_for_query(self, directory: str) -> tuple[str, str | None]:
        return self.get_project(directory), self.reason


@pytest.mark.asyncio
async def test_search_text_answers_after_the_manager_auto_indexes_the_directory() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.symbol_rows = [{**entity("pkg/a.py::a", "a", "function", "pkg/a.py", 1, 2, "Alpha."), "score": 5.0}]
    manager = _AutoIndexingManager()
    register_search_tools(mcp, manager, repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/wt", "query": "a"}))

    assert manager.ensured == ["/tmp/wt"]
    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/a.py::a"]
    assert "error" not in payload


@pytest.mark.asyncio
async def test_search_text_reports_a_failed_auto_index_as_not_indexed() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    manager = _RefusingManager("auto-index attempted and failed: OPENROUTER_API_KEY is not set")
    register_search_tools(mcp, manager, repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/wt", "query": "a"}))

    assert payload["error"]["code"] == "project_not_indexed"
    assert "auto-index attempted and failed: OPENROUTER_API_KEY is not set" in payload["error"]["message"]
    assert repo.symbol_calls == 0                  # refused before touching the graph


@pytest.mark.asyncio
async def test_search_text_still_refuses_a_never_indexed_directory() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    register_search_tools(mcp, _RefusingManager("index it with index_directory first"), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/nope", "query": "a"}))

    assert payload["error"]["code"] == "project_not_indexed"
    assert "index_directory" in payload["error"]["message"]
    assert repo.symbol_calls == 0


@pytest.mark.asyncio
async def test_search_semantic_and_hybrid_and_code_share_the_auto_index_gate(monkeypatch) -> None:
    mcp = FastMCP("test")
    manager = _RefusingManager("auto-index attempted and failed: boom")
    register_search_tools(mcp, manager, FakeSearchRepository(), Settings(_env_file=None))
    for name in ("search_code", "search_semantic", "search_hybrid"):
        tool = mcp._tool_manager.get_tool(name)
        monkeypatch.setattr(_closure_cell_with_attr(tool.fn, "model"), "embed", lambda texts: [[0.1, 0.2]])
        payload = _payload(await mcp.call_tool(name, {"directory": "/tmp/wt", "query": "q"}))
        assert payload["error"]["code"] == "project_not_indexed", name
        assert "auto-index attempted and failed: boom" in payload["error"]["message"], name


@pytest.mark.asyncio
async def test_cypher_query_answers_after_the_manager_auto_indexes_the_directory() -> None:
    mcp = FastMCP("test")
    conn = FakeConnection()
    conn.records = [{"count": 1}]
    manager = _AutoIndexingManager()
    register_cypher_tools(mcp, manager, conn)

    payload = _payload(await mcp.call_tool(
        "cypher_query",
        {"directory": "/tmp/wt", "query": "MATCH (n:Code {project: $project}) RETURN count(n) AS count"},
    ))

    assert manager.ensured == ["/tmp/wt"]
    assert payload["items"] == [{"count": 1}]


@pytest.mark.asyncio
async def test_cypher_query_reports_a_failed_auto_index_as_not_indexed() -> None:
    mcp = FastMCP("test")
    conn = FakeConnection()
    manager = _RefusingManager("auto-index attempted and failed: cost cap exceeded")
    register_cypher_tools(mcp, manager, conn)

    payload = _payload(await mcp.call_tool(
        "cypher_query",
        {"directory": "/tmp/wt", "query": "MATCH (n:Code {project: $project}) RETURN n.name"},
    ))

    assert payload["error"]["code"] == "project_not_indexed"
    assert "auto-index attempted and failed: cost cap exceeded" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_cypher_query_still_refuses_a_never_indexed_directory() -> None:
    mcp = FastMCP("test")
    conn = FakeConnection()
    register_cypher_tools(mcp, _RefusingManager("index it with index_directory first"), conn)

    payload = _payload(await mcp.call_tool(
        "cypher_query",
        {"directory": "/tmp/nope", "query": "MATCH (n:Code {project: $project}) RETURN n.name"},
    ))

    assert payload["error"]["code"] == "project_not_indexed"
    assert "index_directory" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# The navigation tools share the same two pre-DB guards as the search tools:
# the auto-index gate (INV-7: a typo'd directory or a fresh worktree must never
# read as "no matches") and node_type validation.
# ---------------------------------------------------------------------------


class _CountingGraphRepository(FakeGraphRepositoryV2):
    """Counts every read the tools issue, and records the node_types they pass down."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0
        self.seen_node_types: list[list[str] | None] = []

    async def list_entities(self, project, node_types, limit, offset):
        self.reads += 1
        self.seen_node_types.append(node_types)
        return await super().list_entities(project, node_types, limit, offset)

    async def get_structure_root(self, project, node_id=None, file_path=None):
        self.reads += 1
        return await super().get_structure_root(project, node_id=node_id, file_path=file_path)

    async def get_entity_structure(self, project, root_id, depth, node_types, limit, offset):
        self.reads += 1
        self.seen_node_types.append(node_types)
        return await super().get_entity_structure(project, root_id, depth, node_types, limit, offset)

    async def get_entity_context(self, project, node_id, relation_types, limit, offset):
        self.reads += 1
        return await super().get_entity_context(project, node_id, relation_types, limit, offset)

    async def get_related_entities(self, project, node_id, direction, relation_types, limit, offset):
        self.reads += 1
        return await super().get_related_entities(project, node_id, direction, relation_types, limit, offset)

    async def search_entities(self, project, query, node_types, limit, offset):
        self.reads += 1
        self.seen_node_types.append(node_types)
        return await super().search_entities(project, query, node_types, limit, offset)


_GRAPH_TOOL_ARGS = {
    "list_entities": {},
    "get_entity_structure": {"file_path": "pkg/a.py"},
    "get_entity_context": {"node_id": "pkg/a.py::Worker"},
    "get_related_entities": {"node_id": "pkg/a.py::Worker"},
    "search_entities": {"query": "Worker"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_GRAPH_TOOL_ARGS))
async def test_graph_tools_refuse_a_never_indexed_directory(tool_name: str) -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    register_graph_tools(mcp, _RefusingManager("index it with index_directory first"), repo)

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/nope", **_GRAPH_TOOL_ARGS[tool_name]}))

    assert payload["error"]["code"] == "project_not_indexed", payload
    assert "index_directory" in payload["error"]["message"]
    assert repo.reads == 0  # refused before touching the graph


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_GRAPH_TOOL_ARGS))
async def test_graph_tools_answer_after_the_manager_auto_indexes_the_directory(tool_name: str) -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    manager = _AutoIndexingManager()
    register_graph_tools(mcp, manager, repo)

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/wt", **_GRAPH_TOOL_ARGS[tool_name]}))

    # project_is_indexed still says False: only ensure_indexed_for_query makes this answerable.
    assert manager.ensured == ["/tmp/wt"]
    assert "error" not in payload, payload
    assert repo.reads > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_GRAPH_TOOL_ARGS))
async def test_graph_tools_report_a_failed_auto_index_as_not_indexed(tool_name: str) -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    manager = _RefusingManager("auto-index attempted and failed: boom")
    register_graph_tools(mcp, manager, repo)

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/wt", **_GRAPH_TOOL_ARGS[tool_name]}))

    assert payload["error"]["code"] == "project_not_indexed", payload
    assert "auto-index attempted and failed: boom" in payload["error"]["message"]


# --- node_types validation -------------------------------------------------


_NODE_TYPE_GRAPH_TOOLS = {
    "list_entities": {},
    "get_entity_structure": {"file_path": "pkg/a.py"},
    "search_entities": {"query": "Worker"},
}


def _assert_invalid_node_type(payload: dict, bad_value: str) -> None:
    assert payload["error"]["code"] == "invalid_node_type", payload
    message = payload["error"]["message"]
    assert bad_value in message
    for valid in NodeType:
        assert valid.value in message, message


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_NODE_TYPE_GRAPH_TOOLS))
async def test_graph_tools_reject_an_unknown_node_type_with_the_valid_list(tool_name: str) -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(tool_name, {"directory": "/tmp/project", "node_types": ["func"], **_NODE_TYPE_GRAPH_TOOLS[tool_name]})
    )

    _assert_invalid_node_type(payload, "func")
    assert repo.reads == 0  # rejected before the query


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(["search_code", "search_semantic", "search_hybrid"]))
async def test_vector_search_tools_reject_an_unknown_node_type_before_embedding(tool_name: str) -> None:
    mcp = FastMCP("test")
    repo = FakeSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "q", "node_types": ["Func"]}))

    _assert_invalid_node_type(payload, "Func")
    assert repo.code_searches == 0


@pytest.mark.asyncio
async def test_search_text_rejects_an_unknown_node_type_before_querying() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "q", "node_types": ["klass"]}))

    _assert_invalid_node_type(payload, "klass")
    assert repo.symbol_calls == 0


@pytest.mark.asyncio
async def test_list_entities_normalizes_node_type_case_before_querying() -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool("list_entities", {"directory": "/tmp/project", "node_types": ["FUNCTION", "Method"]})
    )

    assert repo.seen_node_types == [["function", "method"]]
    assert payload["meta"]["node_types"] == ["function", "method"]


@pytest.mark.asyncio
async def test_search_text_normalizes_node_type_case_before_filtering() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.symbol_rows = [{**entity("pkg/a.py::run", "run", "function", "pkg/a.py", 1, 2, "Runs."), "score": 5.0}]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(
        await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "run", "node_types": ["Function"]})
    )

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/a.py::run"]
    assert payload["meta"]["node_types"] == ["function"]


@pytest.mark.asyncio
async def test_list_entities_without_node_types_still_passes_none_to_the_repository() -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("list_entities", {"directory": "/tmp/project"}))

    assert repo.seen_node_types == [None]
    assert payload["meta"]["node_types"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_NODE_TYPE_GRAPH_TOOLS))
async def test_node_type_docs_stay_in_sync_with_the_enum(tool_name: str) -> None:
    """The docstrings spell the node types out; a new NodeType must not leave them stale."""
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), _CountingGraphRepository())

    description = mcp._tool_manager.get_tool(tool_name).description

    for node_type in NodeType:
        assert node_type.value in description, f"{tool_name} does not document {node_type.value}"


# ---------------------------------------------------------------------------
# Freshness. Every entity-returning tool re-hashes the files behind the page it
# is about to return, so an unflagged item means "checked, unchanged" — never
# "not checked". The meta block is therefore unconditional; the item flags are not.
# ---------------------------------------------------------------------------


def _write_file(root: Path, rel: str, text: str) -> str:
    """Write a file under *root* and return the MD5 an index would have stored for it."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 — matches the indexer


def test_freshness_checker_classifies_fresh_stale_missing_and_unknown(tmp_path: Path) -> None:
    stored = {
        "pkg/fresh.py": _write_file(tmp_path, "pkg/fresh.py", "unchanged\n"),
        "pkg/stale.py": _write_file(tmp_path, "pkg/stale.py", "before\n"),
        "pkg/gone.py": _write_file(tmp_path, "pkg/gone.py", "doomed\n"),
    }
    (tmp_path / "pkg/stale.py").write_text("after the index ran\n")
    (tmp_path / "pkg/gone.py").unlink()
    _write_file(tmp_path, "pkg/unstored.py", "no checksum was ever stored\n")
    checker = FreshnessChecker(tmp_path, stored)

    items: list[dict[str, object]] = [
        {"id": "1", "file_path": "pkg/fresh.py"},
        {"id": "2", "file_path": "pkg/stale.py"},
        {"id": "3", "file_path": "pkg/gone.py"},
        {"id": "4", "file_path": "pkg/unstored.py"},
        {"id": "5"},  # no file_path: nothing to check, nothing to count
    ]
    checker.annotate_items(items)

    assert items[0] == {"id": "1", "file_path": "pkg/fresh.py"}  # fresh = untouched
    assert items[1]["stale"] is True
    assert items[2]["file_missing"] is True
    assert "stale" not in items[3] and "file_missing" not in items[3]  # unknown: meta only
    assert items[4] == {"id": "5"}
    assert checker.meta() == {
        "checked": 4,
        "stale": ["pkg/stale.py"],
        "missing": ["pkg/gone.py"],
        "unknown": ["pkg/unstored.py"],
    }


def test_freshness_calls_an_unreadable_file_unknown_rather_than_missing(tmp_path: Path, monkeypatch) -> None:
    """A file that is there but cannot be read is not evidence of a ghost node."""
    stored = {"a.py": _write_file(tmp_path, "a.py", "x\n")}
    monkeypatch.setattr(freshness.FileChecksum, "try_compute", staticmethod(lambda path, root=None: None))

    checker = FreshnessChecker(tmp_path, stored)

    assert checker.status("a.py") == "unknown"
    assert checker.meta() == {"checked": 1, "stale": [], "missing": [], "unknown": ["a.py"]}


def test_freshness_hashes_each_file_once_per_page(tmp_path: Path, monkeypatch) -> None:
    digest = _write_file(tmp_path, "pkg/a.py", "def run(): ...\n")
    read_paths: list[Path] = []
    real_try_compute = freshness.FileChecksum.try_compute

    def counting_try_compute(path: Path, root: Path | None = None):
        read_paths.append(path)
        return real_try_compute(path, root)

    monkeypatch.setattr(freshness.FileChecksum, "try_compute", staticmethod(counting_try_compute))
    checker = FreshnessChecker(tmp_path, {"pkg/a.py": digest})

    checker.annotate_items([
        {"id": "pkg/a.py::run", "file_path": "pkg/a.py"},
        {"id": "pkg/a.py::Worker", "file_path": "pkg/a.py"},
        {"relation": "CALLS", "entity": {"id": "pkg/a.py::Worker.run", "file_path": "pkg/a.py"}},
    ])

    assert len(read_paths) == 1
    assert checker.meta()["checked"] == 1


@pytest.mark.asyncio
async def test_list_entities_reports_a_fresh_page_with_no_item_flags(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.checksums = {
        "pkg/a.py": _write_file(tmp_path, "pkg/a.py", "def run(): ...\n"),
        "tests/test_a.py": _write_file(tmp_path, "tests/test_a.py", "def test_run(): ...\n"),
    }
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(tmp_path), "limit": 10}))

    assert payload["meta"]["index_freshness"] == {
        "checked": 2, "stale": [], "missing": [], "unknown": [], "watcher": UNWATCHED,
    }
    assert payload["items"] == [
        entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs."),
        # Not a freshness flag: the path heuristic every listing applies, as search does.
        {**entity("tests/test_a.py::test_run", "test_run", "function", "tests/test_a.py", 1, 3, "Tests."), "is_test": True},
    ]
    assert repo.checksum_calls == 1  # one get_checksums per tool call, whole project


@pytest.mark.asyncio
async def test_list_entities_flags_only_the_entities_of_the_changed_file(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.checksums = {
        "pkg/a.py": _write_file(tmp_path, "pkg/a.py", "def run(): ...\n"),
        "tests/test_a.py": _write_file(tmp_path, "tests/test_a.py", "def test_run(): ...\n"),
    }
    (tmp_path / "pkg" / "a.py").write_text("def run():\n    return 'edited since the index ran'\n")
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(tmp_path), "limit": 10}))

    assert payload["meta"]["index_freshness"] == {
        "checked": 2,
        "stale": ["pkg/a.py"],
        "missing": [],
        "unknown": [],
        "watcher": UNWATCHED,
    }
    assert payload["items"][0]["id"] == "pkg/a.py::run"
    assert payload["items"][0]["stale"] is True
    assert "stale" not in payload["items"][1]  # the untouched file's entity stays unflagged


@pytest.mark.asyncio
async def test_search_text_reports_a_deleted_file_as_missing(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.summary_rows = [{**entity(), "score": 3.0}]  # pkg/module.py — deleted below
    repo.symbol_rows = [
        {**entity("pkg/other.py::other", "other", "function", "pkg/other.py", 1, 2, "Other."), "score": 1.0}
    ]
    repo.checksums = {
        "pkg/module.py": _write_file(tmp_path, "pkg/module.py", "class Worker: ...\n"),
        "pkg/other.py": _write_file(tmp_path, "pkg/other.py", "def other(): ...\n"),
    }
    (tmp_path / "pkg" / "module.py").unlink()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": str(tmp_path), "query": "worker"}))

    assert payload["meta"]["index_freshness"] == {
        "checked": 2,
        "stale": [],
        "missing": ["pkg/module.py"],
        "unknown": [],
        "watcher": UNWATCHED,
    }
    assert payload["items"][0]["entity"]["file_path"] == "pkg/module.py"
    assert payload["items"][0]["entity"]["file_missing"] is True
    assert "file_missing" not in payload["items"][1]["entity"]
    assert repo.checksum_calls == 1


@pytest.mark.asyncio
async def test_search_semantic_leaves_files_without_a_stored_checksum_unflagged(tmp_path: Path, monkeypatch) -> None:
    """No stored checksum (a node that is not a module/document) is 'unknown', not 'stale'."""
    mcp = FastMCP("test")
    repo = FakeSearchRepository()  # checksums stay empty
    _write_file(tmp_path, "pkg/module.py", "class Worker: ...\n")
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_semantic", {"directory": str(tmp_path), "query": "worker"}))

    assert payload["meta"]["index_freshness"] == {
        "checked": 3,
        "stale": [],
        "missing": [],
        "unknown": ["pkg/low.py", "pkg/module.py", "tests/test_module.py"],
        "watcher": UNWATCHED,
    }
    assert all("stale" not in item["entity"] for item in payload["items"])
    assert all("file_missing" not in item["entity"] for item in payload["items"])


@pytest.mark.asyncio
async def test_get_entity_context_annotates_entity_parent_and_sections(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeContextRepository()
    repo.checksums = {
        "pkg/a.py": _write_file(tmp_path, "pkg/a.py", "class Worker: ...\n"),
        "pkg/b.py": _write_file(tmp_path, "pkg/b.py", "def call(): ...\n"),
    }
    (tmp_path / "pkg" / "a.py").write_text("class Worker:\n    pass\n")
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_entity_context",
            {"directory": str(tmp_path), "node_id": "pkg/a.py::Worker", "limit": 5},
        )
    )

    assert payload["entity"]["stale"] is True
    assert payload["parent"]["stale"] is True
    assert payload["sections"]["children"]["items"][0]["stale"] is True
    assert "stale" not in payload["sections"]["incoming"]["items"][0]["entity"]  # pkg/b.py is fresh
    assert payload["meta"]["index_freshness"] == {
        "checked": 2,
        "stale": ["pkg/a.py"],
        "missing": [],
        "unknown": [],
        "watcher": UNWATCHED,
    }
    assert repo.checksum_calls == 1


@pytest.mark.asyncio
async def test_get_entity_structure_annotates_its_root_too(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.checksums = {"pkg/a.py": _write_file(tmp_path, "pkg/a.py", "class Worker: ...\n")}
    (tmp_path / "pkg" / "a.py").write_text("class Worker:\n    def run(self): ...\n")
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_entity_structure",
            {"directory": str(tmp_path), "file_path": "pkg/a.py", "limit": 10},
        )
    )

    assert payload["root"]["stale"] is True
    assert all(item["stale"] is True for item in payload["items"])
    assert payload["meta"]["index_freshness"]["stale"] == ["pkg/a.py"]
    assert payload["meta"]["index_freshness"]["checked"] == 1  # root and items share one file


@pytest.mark.asyncio
async def test_get_related_entities_flags_the_related_entity_itself(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.checksums = {"pkg/module.py": _write_file(tmp_path, "pkg/module.py", "class Worker: ...\n")}
    (tmp_path / "pkg" / "module.py").write_text("class Worker:\n    pass\n")
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(
        await mcp.call_tool(
            "get_related_entities",
            {"directory": str(tmp_path), "node_id": "pkg/a.py::Worker", "limit": 10},
        )
    )

    assert payload["items"][0]["entity"]["stale"] is True
    assert payload["items"][0]["relation"] == "CONTAINS"
    assert payload["meta"]["index_freshness"]["stale"] == ["pkg/module.py"]


class _FreshnessSearchRepository(StoredChecksums):
    """Answers every read any search tool issues with the same single entity."""

    _ROW = {**entity(), "score": 0.9}

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return [{**self._ROW, "chunk_id": "chunk:1", "chunk_kind": "function_body", "match_start_line": 1, "match_end_line": 2}]

    async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return [dict(self._ROW)]

    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return [dict(self._ROW)]

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["search_code", "search_hybrid", "search_semantic", "search_text"])
async def test_search_tools_always_report_index_freshness(tool_name: str, tmp_path: Path, monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = _FreshnessSearchRepository()
    repo.checksums = {"pkg/module.py": _write_file(tmp_path, "pkg/module.py", "class Worker: ...\n")}
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": str(tmp_path), "query": "worker"}))

    assert payload["meta"]["index_freshness"] == {
        "checked": 1, "stale": [], "missing": [], "unknown": [], "watcher": UNWATCHED,
    }
    assert repo.checksum_calls == 1, "one get_checksums per tool call"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(_GRAPH_TOOL_ARGS))
async def test_graph_tools_always_report_index_freshness(tool_name: str, tmp_path: Path) -> None:
    mcp = FastMCP("test")
    repo = _CountingGraphRepository()  # no stored checksums: every path lands in 'unknown'
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool(tool_name, {"directory": str(tmp_path), **_GRAPH_TOOL_ARGS[tool_name]}))

    block = payload["meta"]["index_freshness"]
    assert set(block) == {"checked", "stale", "missing", "unknown", "watcher"}
    assert block["checked"] >= 1 and block["stale"] == [] and block["missing"] == []
    assert repo.checksum_calls == 1


@pytest.mark.asyncio
async def test_search_text_reports_freshness_even_for_an_empty_query(tmp_path: Path) -> None:
    """The block is unconditional: a page with no entities still says it checked nothing."""
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": str(tmp_path), "query": "***"}))

    assert payload["items"] == []
    assert payload["meta"]["index_freshness"] == {
        "checked": 0, "stale": [], "missing": [], "unknown": [], "watcher": UNWATCHED,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["cypher_query", "graph_schema"])
async def test_raw_graph_tools_document_that_they_do_not_check_freshness(tool_name: str) -> None:
    mcp = FastMCP("test")
    register_cypher_tools(mcp, FakeManager(), FakeConnection())

    description = mcp._tool_manager.get_tool(tool_name).description

    assert "freshness is not checked here" in description


# ---------------------------------------------------------------------------
# Live worktrees: an index of the main checkout says nothing about the branches
# checked out in that checkout's worktrees, and a clean freshness block would
# otherwise read as "you are seeing everything".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_freshness_counts_only_worktrees_that_still_exist_on_disk(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    live = tmp_path / "wt-live"
    pruned = tmp_path / "wt-gone"
    git("-C", str(main), "worktree", "add", str(live))
    git("-C", str(main), "worktree", "add", str(pruned))
    shutil.rmtree(pruned)  # still registered, but no branch state anyone could query
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(main), "limit": 10}))

    assert payload["meta"]["index_freshness"]["live_worktrees"] == (
        "1 worktree is not in this index; pass a worktree directory to query its branch state"
    )


@pytest.mark.asyncio
async def test_freshness_omits_live_worktrees_when_the_query_is_the_worktree_itself(tmp_path: Path) -> None:
    """Inside a worktree the answer already *is* the branch state — no warning to give."""
    main = make_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    git("-C", str(main), "worktree", "add", str(wt))
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(wt), "limit": 10}))

    assert "live_worktrees" not in payload["meta"]["index_freshness"]


@pytest.mark.asyncio
async def test_freshness_omits_live_worktrees_for_a_checkout_without_any(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(main), "limit": 10}))

    assert "live_worktrees" not in payload["meta"]["index_freshness"]


@pytest.mark.asyncio
async def test_search_tools_report_live_worktrees_too(tmp_path: Path, monkeypatch) -> None:
    main = make_repo(tmp_path / "main")
    git("-C", str(main), "worktree", "add", str(tmp_path / "wt"))
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _FreshnessSearchRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_semantic", {"directory": str(main), "query": "worker"}))

    assert "1 worktree is not in this index" in payload["meta"]["index_freshness"]["live_worktrees"]


@pytest.mark.asyncio
async def test_search_text_reports_live_worktrees_on_its_empty_page(tmp_path: Path) -> None:
    """The degenerate page carries the warning too, or it reads as a clean bill of health."""
    main = make_repo(tmp_path / "main")
    git("-C", str(main), "worktree", "add", str(tmp_path / "wt"))
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeTextSearchRepository(), Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": str(main), "query": "***"}))

    assert payload["items"] == []
    assert "1 worktree is not in this index" in payload["meta"]["index_freshness"]["live_worktrees"]


@pytest.mark.asyncio
async def test_live_worktrees_value_explains_itself_and_pluralizes(tmp_path: Path) -> None:
    """A bare count read as "you are looking at the wrong tree"; the string says what it means."""
    main = make_repo(tmp_path / "main")
    git("-C", str(main), "worktree", "add", str(tmp_path / "wt1"))
    git("-C", str(main), "worktree", "add", str(tmp_path / "wt2"))
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(main), "limit": 10}))

    assert payload["meta"]["index_freshness"]["live_worktrees"] == (
        "2 worktrees are not in this index; pass a worktree directory to query its branch state"
    )


@pytest.mark.parametrize(
    "tool_name",
    ["search_code", "search_semantic", "search_hybrid", "search_text", *sorted(_GRAPH_TOOL_ARGS)],
)
def test_entity_tools_document_the_live_worktrees_warning(tool_name: str) -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())
    register_search_tools(mcp, FakeManager(), FakeSearchRepository(), Settings(_env_file=None))

    description = mcp._tool_manager.get_tool(tool_name).description

    assert "live_worktrees" in description


# ---------------------------------------------------------------------------
# Literal path patterns. "docs/" echoed back in meta while filtering nothing is
# exactly the silent no-op INV-7 forbids.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["docs/", "docs"])
def test_path_filter_literal_pattern_covers_the_path_and_its_subtree(pattern: str) -> None:
    config = PathFilterConfig.from_paths(include_paths=None, exclude_paths=[pattern])

    assert config.exclude == ["docs", "docs/**"]
    assert path_is_allowed("docs/readme.md", config) is False
    assert path_is_allowed("docs/a/b.md", config) is False
    assert path_is_allowed("docs", config) is False
    assert path_is_allowed("docsx/readme.md", config) is True  # prefix, not a path segment
    assert path_is_allowed("src/app.py", config) is True


def test_path_filter_literal_file_pattern_still_matches_that_one_file() -> None:
    config = PathFilterConfig.from_paths(include_paths=["src/main.py"], exclude_paths=None)

    assert config.include == ["src/main.py", "src/main.py/**"]
    assert path_is_allowed("src/main.py", config) is True
    assert path_is_allowed("src/other.py", config) is False


def test_path_filter_leaves_glob_patterns_untouched() -> None:
    config = PathFilterConfig.from_paths(include_paths=["**/*.py"], exclude_paths=["**/fixtures/**", "a?b", "x[0-9]"])

    assert config.include == ["**/*.py"]
    assert config.exclude == ["**/fixtures/**", "a?b", "x[0-9]"]


@pytest.mark.asyncio
async def test_list_entities_applies_and_reports_a_normalized_literal_exclude() -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool(
        "list_entities",
        {"directory": "/tmp/project", "exclude_paths": ["tests/"], "limit": 10},
    ))

    assert [item["id"] for item in payload["items"]] == ["pkg/a.py::run"]
    assert payload["meta"]["path_filters"]["exclude"] == ["tests", "tests/**"]


@pytest.mark.asyncio
async def test_search_code_applies_a_normalized_literal_exclude(monkeypatch) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeSearchRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "worker", "exclude_paths": ["tests"], "min_score": 0.2},
    ))

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/module.py::Worker.run"]


# ---------------------------------------------------------------------------
# Test-file hits: flagged by path heuristic, and excludable before pagination.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/helpers.py",
        "test/a.py",
        "src/__tests__/button.tsx",
        "pkg/testdata/fixture.json",
        "spec/models_spec.rb",
        "src/conftest.py",
        "src/test_app.py",
        "pkg/module_test.go",
        "src/app.test.ts",
        "src/app.spec.ts",
    ],
)
def test_is_test_path_flags_test_like_paths(path: str) -> None:
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["src/app.py", "pkg/latest/run.py", "src/protest.py", "docs/testing.md", "", "pkg/contest.py"],
)
def test_is_test_path_leaves_production_paths_alone(path: str) -> None:
    assert is_test_path(path) is False


class _MixedSearchRepository(StoredChecksums):
    """Every backend answers with one test-file hit (higher score) and one production hit."""

    _PROD = {**entity("pkg/inventory.py::loadInventory", "loadInventory", "function", "pkg/inventory.py", 10, 20, "Loads."), "score": 0.5}
    _TEST = {**entity("tests/helpers.py::makeInventory", "makeInventory", "function", "tests/helpers.py", 1, 5, "Builds fixtures."), "score": 0.9}

    def _rows(self) -> list[dict[str, object]]:
        return [dict(self._TEST), dict(self._PROD)]

    def _chunk_rows(self) -> list[dict[str, object]]:
        return [
            {**row, "chunk_id": f"chunk:{i}", "chunk_kind": "function_body", "match_start_line": 1, "match_end_line": 2}
            for i, row in enumerate(self._rows())
        ]

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._chunk_rows()

    async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._rows()

    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return self._rows()

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []


_SEARCH_TOOLS = ["search_code", "search_semantic", "search_hybrid", "search_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _SEARCH_TOOLS)
async def test_search_tools_flag_test_like_hits_and_nothing_else(tool_name: str, monkeypatch) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _MixedSearchRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "inventory"}))

    flags = {item["entity"]["id"]: item.get("is_test") for item in payload["items"]}
    assert flags == {"tests/helpers.py::makeInventory": True, "pkg/inventory.py::loadInventory": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _SEARCH_TOOLS)
async def test_search_tools_drop_test_hits_when_include_tests_is_false(tool_name: str, monkeypatch) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _MixedSearchRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(
        tool_name,
        {"directory": "/tmp/project", "query": "inventory", "include_tests": False},
    ))

    assert [item["entity"]["id"] for item in payload["items"]] == ["pkg/inventory.py::loadInventory"]
    assert "is_test" not in str(payload)


class _PagedMixedRepository(StoredChecksums):
    """Four code hits, one of them a test file — enough to page across the exclusion."""

    _ROWS = [
        ("pkg/a.py::a", "pkg/a.py", 0.9),
        ("tests/test_a.py::test_a", "tests/test_a.py", 0.8),
        ("pkg/b.py::b", "pkg/b.py", 0.7),
        ("pkg/c.py::c", "pkg/c.py", 0.6),
    ]

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return [
            {
                **entity(node_id, node_id.split("::")[1], "function", file_path, 1, 2, "S."),
                "score": score,
                "chunk_id": f"chunk:{node_id}",
                "chunk_kind": "function_body",
                "match_start_line": 1,
                "match_end_line": 2,
            }
            for node_id, file_path, score in self._ROWS
        ]


@pytest.mark.asyncio
async def test_include_tests_false_excludes_before_pagination(monkeypatch) -> None:
    """Exclusion happens on the candidate rows, so pages stay full and next_offset stays true."""
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _PagedMixedRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    first = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "q", "include_tests": False, "limit": 2, "offset": 0},
    ))
    second = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "q", "include_tests": False, "limit": 2, "offset": 2},
    ))

    assert [item["entity"]["id"] for item in first["items"]] == ["pkg/a.py::a", "pkg/b.py::b"]
    assert first["page"] == page_dict(2, 0, 2, True)
    assert [item["entity"]["id"] for item in second["items"]] == ["pkg/c.py::c"]
    assert second["page"] == page_dict(2, 2, None, False)


@pytest.mark.asyncio
async def test_include_tests_defaults_to_keeping_test_hits(monkeypatch) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _PagedMixedRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_code", {"directory": "/tmp/project", "query": "q"}))

    assert [item["entity"]["id"] for item in payload["items"]] == [
        "pkg/a.py::a",
        "tests/test_a.py::test_a",
        "pkg/b.py::b",
        "pkg/c.py::c",
    ]


@pytest.mark.asyncio
async def test_include_tests_false_can_starve_a_saturated_pool_and_says_so(monkeypatch) -> None:
    """Dropping tests is a mandatory filter, so a full candidate pool it empties is truncated."""
    mcp = FastMCP("test")

    class _AllTestsRepo(StoredChecksums):
        async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
            return [
                {
                    **entity(f"tests/test_{i}.py::t", "t", "function", f"tests/test_{i}.py", 1, 2, "S."),
                    "score": 0.5,
                    "chunk_id": f"c{i}",
                    "chunk_kind": "function_body",
                    "match_start_line": 1,
                    "match_end_line": 2,
                }
                for i in range(top_k)
            ]

    register_search_tools(mcp, FakeManager(), _AllTestsRepo(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(
        "search_code",
        {"directory": "/tmp/project", "query": "q", "include_tests": False},
    ))

    assert payload["items"] == []
    assert payload["page"]["truncated"] is True


@pytest.mark.parametrize("tool_name", _SEARCH_TOOLS)
def test_search_tools_document_the_test_flag_and_the_include_tests_switch(tool_name: str) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeSearchRepository(), Settings(_env_file=None))

    description = mcp._tool_manager.get_tool(tool_name).description

    assert "include_tests" in description
    assert "is_test" in description


# ---------------------------------------------------------------------------
# Match lists: a document hit with a dozen chunk matches is mostly tokens.
# ---------------------------------------------------------------------------


def _chunk_row(index: int, score: float) -> dict[str, object]:
    return {
        **entity("docs/guide.md", "guide.md", "document", "docs/guide.md", 1, 400, "Guide."),
        "score": score,
        "chunk_id": f"chunk:{index}",
        "chunk_kind": "document_section",
        "match_start_line": index,
        "match_end_line": index + 1,
    }


@pytest.mark.asyncio
async def test_search_text_caps_matches_at_three_and_reports_the_total() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.code_rows = [_chunk_row(i, float(i)) for i in range(12)]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "guide"}))

    item = payload["items"][0]
    assert [match["chunk_id"] for match in item["matches"]] == ["chunk:11", "chunk:10", "chunk:9"]
    assert item["matches_total"] == 12


@pytest.mark.asyncio
async def test_search_text_keeps_a_short_match_list_whole_and_unannotated() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.code_rows = [_chunk_row(0, 1.0), _chunk_row(1, 2.0)]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "guide"}))

    item = payload["items"][0]
    assert [match["chunk_id"] for match in item["matches"]] == ["chunk:0", "chunk:1"]
    assert "matches_total" not in item


@pytest.mark.asyncio
async def test_search_hybrid_caps_matches_at_three_and_reports_the_total(monkeypatch) -> None:
    mcp = FastMCP("test")

    class _ManyChunkRepo(StoredChecksums):
        async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
            return [_chunk_row(i, float(i) / 100) for i in range(12)]

        async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
            return []

    register_search_tools(mcp, FakeManager(), _ManyChunkRepo(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_hybrid", {"directory": "/tmp/project", "query": "guide"}))

    item = payload["items"][0]
    assert [match["chunk_id"] for match in item["matches"]] == ["chunk:11", "chunk:10", "chunk:9"]
    assert item["matches_total"] == 12


# ---------------------------------------------------------------------------
# Weak vector hits. A cosine search always returns its top-k, so a query about a
# feature that simply does not exist comes back with confident-looking 0.44s. The
# vectors cannot prove absence — but they must say plainly that nothing was close.
# ---------------------------------------------------------------------------

_VECTOR_TOOLS = ["search_code", "search_semantic", "search_hybrid"]


class _ScoredVectorRepository(StoredChecksums):
    """Serves the same caller-chosen (id, score) rows to both vector searches."""

    def __init__(self, scored: list[tuple[str, float]]) -> None:
        super().__init__()
        self.scored = scored

    def _rows(self) -> list[dict[str, object]]:
        return [
            {**entity(node_id, node_id.split("::")[1], "function", node_id.split("::")[0], 1, 2, "S."), "score": score}
            for node_id, score in self.scored
        ]

    async def search_code_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return [
            {
                **row,
                "chunk_id": f"chunk:{row['id']}",
                "chunk_kind": "function_body",
                "match_start_line": 1,
                "match_end_line": 2,
            }
            for row in self._rows()
        ]

    async def search_summary_vectors(self, project: str, embedding: list[float], top_k: int) -> list[dict[str, object]]:
        return self._rows()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _VECTOR_TOOLS)
async def test_vector_search_flags_only_the_hits_below_the_weak_threshold(tool_name: str, monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = _ScoredVectorRepository([("pkg/a.py::strong", 0.81), ("pkg/b.py::faint", 0.44)])
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "q"}))

    assert [(item["entity"]["id"], item.get("weak")) for item in payload["items"]] == [
        ("pkg/a.py::strong", None),
        ("pkg/b.py::faint", True),
    ]
    assert "weak_results" not in payload["meta"], "a mixed page still has a real answer on it"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _VECTOR_TOOLS)
async def test_vector_search_says_weak_results_when_the_whole_page_is_weak(tool_name: str, monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = _ScoredVectorRepository([("pkg/a.py::far", 0.46), ("pkg/b.py::farther", 0.43)])
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "q"}))

    assert all(item["weak"] is True for item in payload["items"])
    assert payload["meta"]["weak_results"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _VECTOR_TOOLS)
async def test_vector_search_leaves_a_strong_page_unflagged(tool_name: str, monkeypatch) -> None:
    mcp = FastMCP("test")
    repo = _ScoredVectorRepository([("pkg/a.py::near", 0.72), ("pkg/b.py::alsonear", 0.5)])
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "q"}))

    assert len(payload["items"]) == 2
    assert "weak" not in str(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", _VECTOR_TOOLS)
async def test_vector_search_takes_the_weak_threshold_from_settings(tool_name: str, monkeypatch) -> None:
    """The threshold is a setting because it is model-dependent, not a constant of nature."""
    mcp = FastMCP("test")
    repo = _ScoredVectorRepository([("pkg/a.py::high", 0.9)])
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None, semantic_weak_threshold=0.95))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool(tool_name, {"directory": "/tmp/project", "query": "q"}))

    assert payload["items"][0]["weak"] is True
    assert payload["meta"]["weak_results"] is True


@pytest.mark.asyncio
async def test_vector_search_reports_no_weak_results_on_an_empty_page(monkeypatch) -> None:
    """Zero hits is `min_score`'s or the index's answer — not evidence that hits were weak."""
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), _ScoredVectorRepository([]), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])

    payload = _payload(await mcp.call_tool("search_semantic", {"directory": "/tmp/project", "query": "q"}))

    assert payload["items"] == []
    assert "weak_results" not in payload["meta"]


@pytest.mark.asyncio
async def test_search_text_never_flags_weak_because_bm25_is_a_different_scale() -> None:
    mcp = FastMCP("test")
    repo = FakeTextSearchRepository()
    repo.summary_rows = [{**entity("pkg/a.py::a", "a", "function", "pkg/a.py", 1, 2, "A."), "score": 0.2}]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/project", "query": "a"}))

    assert len(payload["items"]) == 1
    assert "weak" not in str(payload)


@pytest.mark.parametrize("tool_name", _VECTOR_TOOLS)
def test_vector_search_tools_document_the_weak_flag(tool_name: str) -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeSearchRepository(), Settings(_env_file=None))

    description = mcp._tool_manager.get_tool(tool_name).description

    assert "weak" in description and "weak_results" in description
    assert "search_text" in description


def test_search_text_documents_multilingual_token_noise() -> None:
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeSearchRepository(), Settings(_env_file=None))

    description = mcp._tool_manager.get_tool("search_text").description

    assert "phrase" in description.lower()
    assert "include_paths" in description


# ---------------------------------------------------------------------------
# "Who calls X" in one call: get_related_entities resolves a name itself, and
# refuses — with candidates — rather than silently picking one.
# ---------------------------------------------------------------------------


class _NameResolvingRepository(FakeGraphRepositoryV2):
    """Records what `search_entities` was asked and what `get_related_entities` got."""

    def __init__(self, matches: list[dict[str, object]]) -> None:
        super().__init__()
        self.matches = matches
        self.name_queries: list[str] = []
        self.related_node_ids: list[str] = []

    async def search_entities(
        self, project: str, query: str, node_types: list[str] | None, limit: int, offset: int
    ) -> list[dict[str, object]]:
        self.name_queries.append(query)
        return self.matches[:limit]

    async def get_related_entities(
        self,
        project: str,
        node_id: str,
        direction: str,
        relation_types: list[str] | None,
        limit: int,
        offset: int,
    ) -> list[dict[str, object]]:
        self.related_node_ids.append(node_id)
        return [{"relation": "CALLS", "direction": "incoming", "entity": entity()}]


@pytest.mark.asyncio
async def test_get_related_entities_resolves_a_unique_name() -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([entity("pkg/a.py::run", "run", "function", "pkg/a.py", 2, 4, "Runs.")])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool(
        "get_related_entities",
        {"directory": "/tmp/project", "name": "run", "direction": "in", "relation_types": ["CALLS"]},
    ))

    assert repo.name_queries == ["run"]
    assert repo.related_node_ids == ["pkg/a.py::run"]
    assert payload["meta"]["resolved_node_id"] == "pkg/a.py::run"
    assert payload["items"][0]["relation"] == "CALLS"


@pytest.mark.asyncio
async def test_get_related_entities_refuses_an_ambiguous_name_with_candidates() -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([
        entity("pkg/a.py::run_all", "run_all", "function", "pkg/a.py", 2, 4, "Runs."),
        entity("pkg/b.py::Runner.run", "Runner.run", "method", "pkg/b.py", 9, 11, "Runs."),
    ])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", "name": "run"}))

    assert payload["error"]["code"] == "ambiguous_name"
    message = payload["error"]["message"]
    for fragment in ("pkg/a.py::run_all", "function", "pkg/b.py::Runner.run", "method", "pkg/b.py"):
        assert fragment in message
    assert repo.related_node_ids == [], "an ambiguous name must not pick one silently"


@pytest.mark.asyncio
async def test_an_exactly_spelled_name_wins_over_substring_matches() -> None:
    """Field report: name="TASKS" listed ten `readTasks`-like candidates and not the constant."""
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([
        entity("backend/jobs.go::readTasks", "readTasks", "function", "backend/jobs.go", 1, 9, "Reads."),
        entity("backend/jobs.go::TaskSource", "TaskSource", "interface", "backend/jobs.go", 11, 15, "Source."),
        entity("web/tasks/index.ts::TASKS", "TASKS", "variable", "web/tasks/index.ts", 3, 40, ""),
        entity("web/tasks/list.ts::tasks", "tasks", "variable", "web/tasks/list.ts", 1, 1, ""),
    ])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool(
        "get_related_entities", {"directory": "/tmp/project", "name": "TASKS", "direction": "incoming"}
    ))

    assert repo.related_node_ids == ["web/tasks/index.ts::TASKS"]
    assert payload["meta"]["resolved_node_id"] == "web/tasks/index.ts::TASKS"
    assert payload["meta"]["name_match"] == "exact"

    ignoring_case = _payload(await mcp.call_tool(
        "get_related_entities", {"directory": "/tmp/project", "name": "Tasks"}
    ))
    assert ignoring_case["error"]["code"] == "ambiguous_name"
    message = ignoring_case["error"]["message"]
    assert "is the name (ignoring case) of" in message
    assert "readTasks" not in message, "only the exact-ignoring-case names are candidates"


@pytest.mark.asyncio
async def test_path_filters_narrow_an_ambiguous_name_and_filter_the_related_rows() -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([
        entity("mocks/tasks.ts::TASKS", "TASKS", "variable", "mocks/tasks.ts", 1, 5, ""),
        entity("web/tasks/index.ts::TASKS", "TASKS", "variable", "web/tasks/index.ts", 3, 40, ""),
    ])

    async def related(project, node_id, direction, relation_types, limit, offset):
        repo.related_node_ids.append(node_id)
        return [
            {"relation": "IMPORTS", "direction": "incoming",
             "entity": entity("web/board.ts", "board", "module", "web/board.ts", 1, 90, "Board.")},
            {"relation": "IMPORTS", "direction": "incoming",
             "entity": entity("mocks/fake.ts", "fake", "module", "mocks/fake.ts", 1, 9, "Fake.")},
            {"relation": "IMPORTS", "direction": "incoming",
             "entity": entity("web/board.test.ts", "board.test", "module", "web/board.test.ts", 1, 9, "Test.")},
        ]

    repo.get_related_entities = related  # type: ignore[method-assign]
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {
        "directory": "/tmp/project", "name": "TASKS", "direction": "incoming",
        "exclude_paths": ["mocks/"], "include_tests": False,
    }))

    assert payload["meta"]["name_match"] == "exact_within_path_filters"
    assert repo.related_node_ids == ["web/tasks/index.ts::TASKS"]
    # The rows are relation rows — the path sits under `entity`, and the filters must see it.
    assert [item["entity"]["id"] for item in payload["items"]] == ["web/board.ts"]


@pytest.mark.asyncio
async def test_structure_of_a_directory_path_resolves_to_its_directory_node(tmp_path: Path) -> None:
    (tmp_path / "src" / "lib").mkdir(parents=True)
    seen: list[dict[str, object]] = []

    class DirectoryRepository(FakeGraphRepositoryV2):
        async def get_structure_root(self, project, node_id=None, file_path=None):
            seen.append({"node_id": node_id, "file_path": file_path})
            if node_id == "dir:src/lib":
                return entity("dir:src/lib", "lib", "directory", "src/lib", None, None, "Lib.")
            return None

    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), DirectoryRepository())

    payload = _payload(await mcp.call_tool(
        "get_entity_structure", {"directory": str(tmp_path), "file_path": "./src/lib/"}
    ))

    assert payload["root"]["id"] == "dir:src/lib"
    assert seen == [{"node_id": None, "file_path": "./src/lib/"}, {"node_id": "dir:src/lib", "file_path": None}]

    (tmp_path / "empty").mkdir()
    missing = _payload(await mcp.call_tool("get_entity_structure", {"directory": str(tmp_path), "file_path": "empty"}))
    assert missing["error"]["code"] == "not_found"
    assert "is a directory on disk" in missing["error"]["message"]


@pytest.mark.asyncio
async def test_listings_drop_tests_on_request_and_return_whole_summaries(tmp_path: Path) -> None:
    long_summary = "Capabilities: sheet construction, candidate extraction. " * 10

    class InventoryRepository(FakeGraphRepositoryV2):
        async def list_entities(self, project, node_types, limit, offset):
            return [
                entity("legacy/old.ts", "old", "module", "legacy/old.ts", 1, 9, long_summary),
                entity("src/rules.ts", "rules", "module", "src/rules.ts", 1, 90, long_summary),
                entity("src/rules.test.ts", "rules.test", "module", "src/rules.test.ts", 1, 9, "Tests."),
                entity("src/rules.ts::R23", "R23", "variable", "src/rules.ts", 4, 9, "", ),
            ]

    repo = InventoryRepository()
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("list_entities", {
        "directory": str(tmp_path), "include_tests": False, "full_summaries": True, "limit": 100,
    }))

    ids = [item["id"] for item in payload["items"]]
    assert ids == ["legacy/old.ts", "src/rules.ts", "src/rules.ts::R23"], "path order kept, tests dropped"
    assert all("archived" not in item for item in payload["items"])
    assert payload["items"][1]["summary"] == long_summary
    assert payload["meta"]["full_summaries"] is True and payload["meta"]["include_tests"] is False
    assert payload["page"]["limit"] == FULL_SUMMARY_MAX_LIMIT


def test_entity_refs_show_the_signature_only_where_there_is_no_summary() -> None:
    declaration = {**entity("a.ts::MAX", "MAX", "variable", "a.ts", 1, 1, ""), "signature": "export const MAX = 3"}
    function = {**entity("a.ts::run", "run", "function", "a.ts", 3, 9, "Runs."), "signature": "function run()"}

    assert entity_ref(declaration)["signature"] == "export const MAX = 3"
    assert "signature" not in entity_ref(function)


@pytest.mark.asyncio
async def test_get_related_entities_caps_the_candidate_list_and_counts_the_rest() -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([
        entity(f"pkg/f{i}.py::run", "run", "function", f"pkg/f{i}.py", 1, 2, "Runs.") for i in range(14)
    ])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", "name": "run"}))

    message = payload["error"]["message"]
    assert message.count("::run") == 10
    assert "and 4 more" in message


@pytest.mark.asyncio
async def test_get_related_entities_reports_an_unknown_name_as_not_found() -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", "name": "nope"}))

    assert payload["error"]["code"] == "not_found"
    assert "search_text" in payload["error"]["message"]
    assert repo.related_node_ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"node_id": "pkg/a.py::run", "name": "run"}, id="both"),
        pytest.param({}, id="neither"),
    ],
)
async def test_get_related_entities_requires_exactly_one_of_node_id_and_name(args: dict) -> None:
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([entity()])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", **args}))

    assert payload["error"]["code"] == "invalid_argument"
    assert repo.name_queries == [] and repo.related_node_ids == []


def test_get_related_entities_documents_the_name_shortcut() -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    description = mcp._tool_manager.get_tool("get_related_entities").description

    assert "name" in description
    assert "ambiguous_name" in description


@pytest.mark.asyncio
async def test_get_related_entities_marks_the_surplus_as_approximate_past_the_scan_cap() -> None:
    """Past the scan the exact total is unknown, so the count says "+" instead of inventing one."""
    mcp = FastMCP("test")
    repo = _NameResolvingRepository([
        entity(f"pkg/f{i}.py::run", "run", "function", f"pkg/f{i}.py", 1, 2, "Runs.") for i in range(200)
    ])
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_related_entities", {"directory": "/tmp/project", "name": "run"}))

    assert payload["error"]["code"] == "ambiguous_name"
    assert "and 41+ more" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# meta.answered_from: which tree, which commit. Two agents disagreed in the field
# about whether their branch work was in the graph, and no field of the response
# could settle it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answered_from_names_the_checkout_branch_and_commit(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    head = git("-C", str(main), "rev-parse", "HEAD").strip()
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(main), "limit": 10}))

    assert payload["meta"]["answered_from"] == f"{main.resolve()} (main @ {head[:7]})"


@pytest.mark.asyncio
async def test_answered_from_names_the_worktree_and_the_checkout_it_came_from(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    wt = tmp_path / "wt"
    git("-C", str(main), "worktree", "add", "-b", "feature/x", str(wt))
    head = git("-C", str(wt), "rev-parse", "HEAD").strip()
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(wt), "limit": 10}))

    assert payload["meta"]["answered_from"] == (
        f"{wt.resolve()} (feature/x @ {head[:7]}), a git worktree of {main.resolve()}"
    )


@pytest.mark.asyncio
async def test_answered_from_degrades_to_the_path_outside_git(tmp_path: Path) -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), FakeGraphRepositoryV2())

    payload = _payload(await mcp.call_tool("list_entities", {"directory": str(tmp_path), "limit": 10}))

    assert payload["meta"]["answered_from"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_search_text_reports_answered_from_on_its_empty_page(tmp_path: Path) -> None:
    main = make_repo(tmp_path / "main")
    mcp = FastMCP("test")
    register_search_tools(mcp, FakeManager(), FakeTextSearchRepository(), Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": str(main), "query": "***"}))

    assert payload["items"] == []
    assert payload["meta"]["answered_from"].startswith(str(main.resolve()))


@pytest.mark.parametrize(
    "tool_name",
    ["search_code", "search_semantic", "search_hybrid", "search_text", *sorted(_GRAPH_TOOL_ARGS)],
)
@pytest.mark.asyncio
async def test_entity_tools_always_report_answered_from(tool_name: str, tmp_path: Path, monkeypatch) -> None:
    mcp = FastMCP("test")
    register_graph_tools(mcp, FakeManager(), _CountingGraphRepository())
    register_search_tools(mcp, FakeManager(), _FreshnessSearchRepository(), Settings(_env_file=None))
    monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])
    args = {"directory": str(tmp_path), **_GRAPH_TOOL_ARGS.get(tool_name, {"query": "worker"})}

    payload = _payload(await mcp.call_tool(tool_name, args))

    assert payload["meta"]["answered_from"] == str(tmp_path.resolve())


def test_freshness_does_not_judge_directory_nodes(tmp_path: Path) -> None:
    """A directory has no content to hash; calling it "unknown" read as doubt about the index."""
    digest = _write_file(tmp_path, "pkg/a.py", "def run(): ...\n")
    checker = FreshnessChecker(tmp_path, {"pkg/a.py": digest})

    items: list[dict[str, object]] = [
        {"id": "dir:pkg", "node_type": "directory", "file_path": "pkg"},
        {"id": "pkg/a.py::run", "node_type": "function", "file_path": "pkg/a.py"},
    ]
    checker.annotate_items(items)

    assert items[0] == {"id": "dir:pkg", "node_type": "directory", "file_path": "pkg"}
    assert checker.meta() == {"checked": 1, "stale": [], "missing": [], "unknown": []}
