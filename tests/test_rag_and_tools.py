from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.code_indexer.models import CodeNode, NodeType
from inflorescence.config import Settings
from inflorescence.db import queries
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import setup_schema
from inflorescence.rag.chunker import Chunk
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.rag.indexer import RAGIndexer
from inflorescence.tools.cypher import register_cypher_tools
from inflorescence.tools.graph import register_graph_tools
from inflorescence.tools.indexing import register_indexing_tools
from inflorescence.tools.path_filters import PathFilterConfig, path_is_allowed
from inflorescence.tools.responses import (
    NAV_SUMMARY_MAX_CHARS,
    entity_ref,
    error_response,
    paginated_response,
    truncate_summary,
)
from inflorescence.tools.search import _code_hits, register_search_tools


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
    from inflorescence.db.schema import backfill_node_type

    conn = _BackfillConnection([{"id": "pkg/a.py::run"}])
    updated = await backfill_node_type(conn)

    assert updated == 9  # one SET per node-type label group
    assert any("MATCH (n:Function) WHERE n.node_type IS NULL SET n.node_type = 'function'" in q for q in conn.writes)
    assert any("MATCH (n:Module) WHERE n.node_type IS NULL SET n.node_type = 'module'" in q for q in conn.writes)
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
        assert "LIMIT $top_k" in query


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
    monkeypatch.setattr(tool.fn.__closure__[0].cell_contents, "embed", lambda texts: [None])

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
async def test_get_entity_structure_returns_structured_not_found_error() -> None:
    mcp = FastMCP("test")
    repo = FakeGraphRepositoryV2()
    repo.root = None
    register_graph_tools(mcp, FakeManager(), repo)

    payload = _payload(await mcp.call_tool("get_entity_structure", {"directory": "/tmp/project", "file_path": "missing.py"}))

    assert payload == {"error": {"code": "not_found", "message": "Structure root not found"}}


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
        {"id": "a.py::foo", "summary": "does foo", "summary_input_hash": "h1",
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
        "summary": "does foo", "summary_input_hash": "h1",
        "summary_embedding": [0.1, 0.2], "summary_embedding_hash": "h1",
    }
    assert stored["a.py"]["summary_embedding"] is None


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
    assert "n.node_type" not in conn.query
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
    assert "n.node_type" not in conn.query
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
    assert conn.params == {"query": "runs", "top_k": 25, "project": "project-id"}
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
    assert conn.params == {"query": "run", "top_k": 25, "project": "project-id"}
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
    assert conn.params == {"query": "runs", "top_k": 25, "project": "project-id"}
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


class FakeGraphRepositoryV2:
    def __init__(self) -> None:
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


class FakeContextRepository(FakeManager):
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


class FakeSearchRepository:
    def __init__(self) -> None:
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


class FakeTextSearchRepository:
    def __init__(self) -> None:
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


class FakeFailingTextRepository:
    async def search_summary_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        raise RuntimeError("text index summary_text does not exist")

    async def search_symbol_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []

    async def search_code_text(self, project: str, query: str, top_k: int) -> list[dict[str, object]]:
        return []


class FakeConnectionErrorTextRepository:
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

    assert repo.received_queries == ["ValueError invalid literal arr 0"] * 3


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
    assert payload["meta"] == {"project": "project-id", "node_id": "pkg/a.py::Worker", "relation_types": ["CALLS"]}


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


class _SaturatingSearchRepo:
    """A repo whose vector search fills the candidate pool with rows outside any filter."""

    def __init__(self, count: int) -> None:
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

    class _NeverMatchRepo(FakeManager):
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
