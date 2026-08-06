"""G3 · Self-healing repair sees staleness, and the DB tells the truth — oracles for INV-7/INV-8.

The contract: every incomplete OR stale state — a summary whose vector is from the old
text, a file holding chunks of old content, a docstring surrogate, a tree left edgeless by
a killed run — must be detectable by a query and healed by the next healthy run, without a
rebuild or manual surgery. And write counters must report the rows the DB actually wrote,
not the rows submitted.

These are **live** oracles against a real Memgraph: the desyncs live in the graph and a
mock can't reproduce detection over real property state. They are skipped unless
``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a **throwaway** instance, so CI and a
developer's real ``bolt://localhost:7687`` are never touched. External APIs (LLM,
embedder) are faked so the failure being reproduced — a dropped embedding, an LLM outage —
is deterministic; the graph state and every repair query/write are real.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, IndexerConfig, NodeType
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.registry import ParserRegistry
from inflorescence.config import Settings
from inflorescence.rag.indexer import RAGIndexer

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_NO_LLM = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)
_DIM = 1536  # must match the vector index dimension created by setup_schema
_VEC1 = [0.1] * _DIM
_VEC2 = [0.2] * _DIM


class _Embedder:
    """A deterministic embedder; ``fail=True`` drops every embedding (returns None)."""

    def __init__(self, vec: list[float] | None = None, fail: bool = False) -> None:
        self.model = "fake-embedder"
        self._vec = vec or _VEC1
        self._fail = fail

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [None if self._fail else list(self._vec) for _ in texts]

    def preflight(self) -> None:
        return None


class _LLM:
    """A fake summary LLM; ``fail=True`` raises like a transient outage."""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self.calls = 0

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        if self._fail:
            raise RuntimeError("LLM unavailable")
        return "regenerated summary"


async def _fresh_repo():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    return GraphRepository(conn), conn


def _registry(*parsers) -> ParserRegistry:
    reg = ParserRegistry()
    for p in parsers:
        reg.register(p)
    return reg


def _indexer(repo, *, embed_fail: bool = False, vec: list[float] | None = None) -> RAGIndexer:
    indexer = RAGIndexer(repo, Settings(_env_file=None, memgraph_url=LIVE_URL))
    emb = _Embedder(vec=vec, fail=embed_fail)
    indexer._code_embedder = emb  # type: ignore[assignment]
    indexer._summary_embedder = emb  # type: ignore[assignment]
    return indexer


@live
@pytest.mark.asyncio
async def test_stale_summary_embedding_is_detected_and_repaired() -> None:
    """Finding 8a: a summary regenerated with a dropped embedding keeps a vector from the
    OLD text — the node looks complete. The old repair (embedding IS NULL) can't see it;
    the freshness marker makes it detectable, and a healthy run re-embeds it.
    """
    repo, conn = await _fresh_repo()
    project = f"g3-stale-embed-{uuid.uuid4().hex[:8]}"
    try:
        node = CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION,
                        file_path="a.py", line_start=1, line_end=1, summary="the new summary text")
        node.summary_input_hash = "H_new"
        await repo.upsert_nodes(project, [node])
        # The vector left behind was computed for the OLD summary (its hash is the old input).
        await repo.store_summary_embeddings(
            project, [{"node_id": "a.py::foo", "embedding": _VEC1, "embedding_hash": "H_old"}]
        )

        # Detectable although summary AND embedding are both present.
        stale = await repo.get_stale_summary_embedding_nodes(project)
        assert [n.id for n in stale] == ["a.py::foo"], "stale summary vector was not detected"

        # A healthy run re-embeds from the current summary; the vector catches up.
        await _indexer(repo, vec=_VEC2).index_summaries(project, stale)

        assert await repo.get_stale_summary_embedding_nodes(project) == [], "repair did not converge"
        stored = await repo.get_stored_summaries(project)
        assert stored["a.py::foo"]["summary_embedding_hash"] == "H_new", "vector still tied to old text"
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_stale_chunks_are_detected_and_repaired(tmp_path: Path) -> None:
    """Finding 8b: a changed file whose new chunk-embedding was dropped keeps chunks of the
    OLD content (chunks > 0, so a "chunks == 0" probe misses it). The coverage marker makes
    it detectable, and a healthy re-chunk stores the new content and prunes the stale chunks.
    """
    repo, conn = await _fresh_repo()
    project = f"g3-stale-chunks-{uuid.uuid4().hex[:8]}"
    src = tmp_path / "a.py"
    try:
        builder = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)

        # v1: full, healthy index — chunks stored and the file marked covered for md5(v1).
        src.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
        r1 = await builder.build(project, tmp_path)
        await _indexer(repo).index_code(project, tmp_path, r1.nodes, prune_stale=True,
                                        reconcilable_files=r1.reconcilable_files)
        v1_chunks = set(await repo.get_chunk_index(project))
        assert v1_chunks, "v1 produced no chunks"
        assert await repo.get_files_with_stale_chunks(project) == [], "healthy index looked stale"

        # v2: file changes, but the chunk-embedding batch fails — new chunks not stored, the
        # old content lingers, and the coverage marker is NOT advanced.
        src.write_text("def foo():\n    return 111\n\ndef bar():\n    return 222\n", encoding="utf-8")
        r2 = await builder.build(project, tmp_path)
        await _indexer(repo, embed_fail=True).index_code(project, tmp_path, r2.nodes, prune_stale=True,
                                                         reconcilable_files=r2.reconcilable_files)

        # Detectable although the file still has chunks (of the old content).
        assert "a.py" in await repo.get_files_with_stale_chunks(project), "stale chunks not detected"

        # Healthy re-chunk heals: new chunks stored, stale ones pruned, marker advanced.
        nodes = await repo.get_nodes_for_files(project, ["a.py"])
        await _indexer(repo).index_code(project, tmp_path, nodes, prune_stale=True,
                                        reconcilable_files={"a.py"})

        assert await repo.get_files_with_stale_chunks(project) == [], "repair did not converge"
        now = set(await repo.get_chunk_index(project))
        assert now and now.isdisjoint(v1_chunks), "old-content chunks were not pruned"
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_repair_surrogate_is_reattempted_not_locked_in(tmp_path: Path) -> None:
    """Finding 8c: a repair that falls back to the docstring surrogate on an LLM outage
    stamps a retry marker (empty input hash), so it stays detectable and the next healthy
    run re-summarizes it — instead of a valid hash that would lock the surrogate in forever.
    """
    repo, conn = await _fresh_repo()
    project = f"g3-surrogate-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text('def foo():\n    "the docstring"\n    return 1\n', encoding="utf-8")
    try:
        # Seed a node with a docstring but no summary (as a killed run would leave it).
        node = CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION,
                        file_path="a.py", line_start=1, line_end=3, signature="def foo()",
                        docstring="the docstring", summary="")
        await repo.upsert_nodes(project, [node])
        assert [n.id for n, _ in await repo.get_nodes_missing_summaries(project)] == ["a.py::foo"]

        # Repair during an LLM outage falls back to the docstring surrogate.
        broken = GraphBuilder(repo=repo, llm=_LLM(fail=True), settings=Settings(_env_file=None),
                              config=IndexerConfig(use_llm_summaries=True))
        await broken.repair_missing_summaries(project, tmp_path)
        stored = await repo.get_stored_summaries(project)
        assert stored["a.py::foo"]["summary"] == "the docstring"      # surrogate written (stale beats empty)
        assert stored["a.py::foo"]["summary_input_hash"] == ""        # ...but marked for retry

        # Still detectable, so the next healthy run re-summarizes it.
        assert [n.id for n, _ in await repo.get_nodes_missing_summaries(project)] == ["a.py::foo"]
        healthy = GraphBuilder(repo=repo, llm=_LLM(fail=False), settings=Settings(_env_file=None),
                               config=IndexerConfig(use_llm_summaries=True))
        await healthy.repair_missing_summaries(project, tmp_path)

        stored = await repo.get_stored_summaries(project)
        assert stored["a.py::foo"]["summary"] == "regenerated summary"  # real summary now
        assert stored["a.py::foo"]["summary_input_hash"] != ""          # no longer retry-marked
        assert await repo.get_nodes_missing_summaries(project) == []    # converged
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_edgeless_graph_repairs_contains_backbone(tmp_path: Path) -> None:
    """Finding 9: a run killed after wiping edges but before re-MERGEing them can leave the
    tree edgeless — which the summary/chunk repairs don't touch. CONTAINS is reconstructable
    from parent_id, so repair rebuilds exactly the missing structural links.
    """
    repo, conn = await _fresh_repo()
    project = f"g3-edgeless-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text("class C:\n    def m(self):\n        return 1\n", encoding="utf-8")
    try:
        builder = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)
        await builder.build(project, tmp_path)
        assert len(await repo.get_code_edges(project)) > 0
        assert await repo.get_nodes_missing_contains_edges(project) == []

        # Reproduce the edgeless state the old wipe-then-crash left behind.
        await repo.delete_code_edges(project)
        missing = await repo.get_nodes_missing_contains_edges(project)
        assert missing, "edgeless graph was not detected as missing CONTAINS"
        assert ("dir:.", "CONTAINS", "a.py") not in await repo.get_code_edges(project)

        # Repair rebuilds the CONTAINS backbone from parent_id.
        restored = await repo.repair_contains_edges(project, missing)
        assert restored == len(missing)
        assert await repo.get_nodes_missing_contains_edges(project) == [], "repair did not converge"
        edges = await repo.get_code_edges(project)
        assert ("dir:.", "CONTAINS", "a.py") in edges
        assert ("a.py", "CONTAINS", "a.py::C") in edges
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_write_counters_report_actual_db_writes() -> None:
    """Finding 14 (INV-7): "Stored N" means N rows in the DB, not N submitted. An edge with an
    unresolved endpoint, and a store racing a delete, both report the shortfall instead of
    claiming success.
    """
    repo, conn = await _fresh_repo()
    project = f"g3-counters-{uuid.uuid4().hex[:8]}"
    try:
        a = CodeNode(id="a.py::a", name="a", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=1)
        b = CodeNode(id="a.py::b", name="b", node_type=NodeType.FUNCTION, file_path="a.py", line_start=2, line_end=2)
        await repo.upsert_nodes(project, [a, b])

        # One edge resolves; one points at a node that doesn't exist (unresolved CALLS target).
        edges = [
            Edge(source="a.py::a", target="a.py::b", edge_type=EdgeType.CALLS),
            Edge(source="a.py::a", target="a.py::ghost", edge_type=EdgeType.CALLS),
        ]
        written = await repo.upsert_edges(project, edges)
        assert written == 1, "edge counter claimed the unresolved edge was written"

        # A summary-embedding store for a node that isn't there writes nothing.
        wrote = await repo.store_summary_embeddings(
            project, [{"node_id": "a.py::ghost", "embedding": _VEC1, "embedding_hash": "h"}]
        )
        assert wrote == 0, "embedding counter claimed a write against a missing node"

        # A chunk whose owner node is missing is dropped, not counted.
        stored = await repo.store_code_chunks(project, [{
            "chunk_id": "chunk:x", "node_id": "a.py::ghost", "file_path": "a.py", "content": "x",
            "start_line": 1, "end_line": 1, "language": "python", "chunk_kind": "function_body",
            "embedding": _VEC1,
        }])
        assert stored == 0, "chunk counter claimed a write against a missing owner"
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_upsert_preserves_stored_summary_on_empty_incoming() -> None:
    """INV-1/INV-3: the skeleton upsert (summary='') must never wipe a paid summary,
    and a re-generated non-empty summary must still win."""
    repo, conn = await _fresh_repo()
    project = f"g3-preserve-summary-{uuid.uuid4().hex[:8]}"

    def _foo(**overrides) -> CodeNode:
        return CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION,
                        file_path="a.py", line_start=1, line_end=1, **overrides)

    try:
        await repo.upsert_nodes(project, [_foo()])
        await repo.store_summaries(
            project, [{"node_id": "a.py::foo", "summary": "paid summary", "summary_input_hash": "H1"}]
        )

        # Early skeleton upsert of the same node: summary='' must not clobber.
        await repo.upsert_nodes(project, [_foo()])
        stored = await repo.get_stored_summaries(project)
        assert stored["a.py::foo"]["summary"] == "paid summary", "skeleton upsert wiped a paid summary"
        assert stored["a.py::foo"]["summary_input_hash"] == "H1"

        # A re-generated (non-empty) summary still wins.
        regenerated = _foo(summary="new summary")
        regenerated.summary_input_hash = "H2"
        await repo.upsert_nodes(project, [regenerated])
        stored = await repo.get_stored_summaries(project)
        assert stored["a.py::foo"]["summary"] == "new summary", "regenerated summary did not land"
        assert stored["a.py::foo"]["summary_input_hash"] == "H2"
    finally:
        await repo.delete_project(project)
        await conn.close()
