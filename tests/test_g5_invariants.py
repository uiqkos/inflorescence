"""G5 · Search & query correctness — live oracles for INV-9/INV-7 (findings 10–15).

The contract:
  * Every mandatory filter (project scope, path, node_type, relation type) is applied so
    one project or filter-narrowed set cannot be evicted by a neighbour's limit (INV-9).
  * A tool parameter that is declared either changes the result or the tool says it does
    not support it — get_entity_context must honour offset (no infinite pagination) and
    relation_types (finding 10).
  * An empty result caused by an early-stopped scan or a failed backend is distinguishable
    in the response from a genuine "no matches" (INV-7 — findings 11, 13, 15).
  * The read-only Cypher tool cannot reach a global, cross-project index via CALL
    (finding 12 — a real leak in both cypher_query and the dashboard /query).

These are **live** oracles against a real Memgraph: cross-project leakage, vector-pool
saturation, and a missing-index failure only exist against real graph state and the real
MAGE search procedures. Skipped unless ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a
**throwaway** instance, so CI and a real ``bolt://localhost:7687`` are never touched. Only
the query embedder is faked (deterministic vector); every query, index, and CALL is real.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.config import Settings
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.tools.cypher import validate_readonly_query
from inflorescence.tools.graph import register_graph_tools
from inflorescence.tools.search import register_search_tools

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536
_VEC = [0.1] * _DIM


class _FixedManager:
    def __init__(self, project: str) -> None:
        self._project = project

    def get_project(self, directory: str) -> str:
        return self._project

    async def project_is_indexed(self, directory: str) -> bool:
        return True


async def _conn():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    return conn


def _payload(result):
    import json

    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


# ---------------------------------------------------------------------------
# Finding 10 — get_entity_context honours relation_types and offset
# ---------------------------------------------------------------------------

_SEED_CONTEXT = """
CREATE (m:Code:Module {project:'ctx', id:'mod', name:'mod', file_path:'a.py', line_start:1, line_end:99, summary:'m'})
CREATE (h:Code:Class {project:'ctx', id:'hub', name:'hub', parent_id:'mod', file_path:'a.py', line_start:1, line_end:9, summary:'h'})
CREATE (c1:Code:Method {project:'ctx', id:'c1', name:'c1', parent_id:'hub', file_path:'a.py', line_start:2, line_end:3, summary:'c1'})
CREATE (c2:Code:Method {project:'ctx', id:'c2', name:'c2', parent_id:'hub', file_path:'a.py', line_start:4, line_end:5, summary:'c2'})
CREATE (c3:Code:Method {project:'ctx', id:'c3', name:'c3', parent_id:'hub', file_path:'a.py', line_start:6, line_end:7, summary:'c3'})
CREATE (s1:Code:Function {project:'ctx', id:'s1', name:'s1', file_path:'b.py', line_start:1, line_end:2, summary:'s1'})
CREATE (s2:Code:Function {project:'ctx', id:'s2', name:'s2', file_path:'b.py', line_start:3, line_end:4, summary:'s2'})
CREATE (t1:Code:Function {project:'ctx', id:'t1', name:'t1', file_path:'c.py', line_start:1, line_end:2, summary:'t1'})
CREATE (t2:Code:Class {project:'ctx', id:'t2', name:'t2', file_path:'c.py', line_start:3, line_end:4, summary:'t2'})
CREATE (m)-[:CONTAINS]->(h)
CREATE (h)-[:CONTAINS]->(c1)
CREATE (h)-[:CONTAINS]->(c2)
CREATE (h)-[:CONTAINS]->(c3)
CREATE (s1)-[:CALLS]->(h)
CREATE (s2)-[:IMPORTS]->(h)
CREATE (h)-[:CALLS]->(t1)
CREATE (h)-[:INHERITS]->(t2)
"""


@live
async def test_get_entity_context_honours_relation_types_and_offset_live() -> None:
    from inflorescence.db.repository import GraphRepository

    conn = await _conn()
    try:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        await conn.execute_write(_SEED_CONTEXT)
        repo = GraphRepository(conn)

        # relation_types=["CALLS"] filters incoming/outgoing to CALLS only (finding 10a):
        scoped = await repo.get_entity_context("ctx", "hub", relation_types=["CALLS"], limit=26, offset=0)
        assert {r["relation"] for r in scoped["incoming"]} == {"CALLS"}  # not IMPORTS/CONTAINS
        assert {r["relation"] for r in scoped["outgoing"]} == {"CALLS"}  # not INHERITS/CONTAINS

        # No filter -> every relation type is present (baseline). incoming also carries the
        # parent module's CONTAINS edge (m)-[:CONTAINS]->(hub), which relation_types filters out.
        every = await repo.get_entity_context("ctx", "hub", relation_types=None, limit=26, offset=0)
        assert {r["relation"] for r in every["incoming"]} == {"CALLS", "IMPORTS", "CONTAINS"}
        assert {r["relation"] for r in every["outgoing"]} == {"CALLS", "INHERITS", "CONTAINS"}

        # offset actually paginates (finding 10b): a deep offset yields an empty slice,
        # so an agent walking next_offset terminates instead of looping the first page.
        page0 = await repo.get_entity_context("ctx", "hub", relation_types=None, limit=2, offset=0)
        page1 = await repo.get_entity_context("ctx", "hub", relation_types=None, limit=2, offset=2)
        page2 = await repo.get_entity_context("ctx", "hub", relation_types=None, limit=2, offset=4)
        seen = {c["id"] for c in page0["children"]} | {c["id"] for c in page1["children"]}
        assert seen == {"c1", "c2", "c3"}  # pages cover the set, not repeat page 0
        assert page2["children"] == []  # deep offset terminates the walk
    finally:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        await conn.close()


# ---------------------------------------------------------------------------
# Findings 11 & 12 — filtered vector search + global-CALL leak (one shared seed)
#
# Seeded once, no deletion of indexed nodes before searching: on this Memgraph a
# vector search that runs after its indexed nodes were deleted trips a "property from a
# deleted object" error, so the throwaway must be fresh (see the decisions log). Distinct
# embeddings are required — identical vectors build a degenerate index that returns a
# fraction of the rows — and >= 300 rows guarantee the candidate pool (>= 100) saturates.
# ---------------------------------------------------------------------------

_SEED_A = """
UNWIND range(0, $na - 1) AS i
CREATE (n:Code:Function {project:'A', id:'A/src/f'+toString(i)+'.py::fn', name:'fn', file_path:'src/f'+toString(i)+'.py', line_start:1, line_end:2, summary:'alpha', node_type:'function', checksum:'x'})
CREATE (c:CodeChunk {project:'A', id:'A/chunk'+toString(i), node_id:'A/src/f'+toString(i)+'.py::fn', file_path:'src/f'+toString(i)+'.py', content:'code', start_line:1, end_line:2, language:'python', chunk_kind:'function_body', embedding:[0.5 + i * 0.001] + $tail})
CREATE (n)-[:HAS_CHUNK]->(c)
"""
_SEED_B = """
UNWIND range(0, $nb - 1) AS i
CREATE (n:Code:Function {project:'B', id:'B/x'+toString(i)+'.py::fn', name:'fn', file_path:'x'+toString(i)+'.py', line_start:1, line_end:2, summary:'beta', node_type:'function', checksum:'x'})
CREATE (c:CodeChunk {project:'B', id:'B/chunk'+toString(i), node_id:'B/x'+toString(i)+'.py::fn', file_path:'x'+toString(i)+'.py', content:'code', start_line:1, end_line:2, language:'python', chunk_kind:'function_body', embedding:[0.6 + i * 0.001] + $tail})
CREATE (n)-[:HAS_CHUNK]->(c)
"""
_QVEC = [0.5] + [0.1] * (_DIM - 1)


@live
async def test_filtered_vector_search_truncation_and_global_call_leak_live(monkeypatch) -> None:
    from inflorescence.db.repository import GraphRepository

    conn = await _conn()
    try:
        # Drop the vector index, seed, THEN build the index over the fresh nodes. On this
        # Memgraph a vector index retains references to deleted nodes across DROP, so a
        # search after any prior test deleted indexed chunks trips "property from a deleted
        # object"; building the index *after* the current nodes exist gives a clean HNSW.
        # An explicit CREATE (not setup_schema, which skips an index it still detects)
        # guarantees the rebuild happens.
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        for idx in ("code_embeddings", "summary_embeddings"):
            with contextlib.suppress(Exception):  # absent index is fine
                await conn.execute_write(f"DROP VECTOR INDEX {idx}")
        tail = [0.1] * (_DIM - 1)
        await conn.execute_write(_SEED_A, {"na": 300, "tail": tail})
        await conn.execute_write(_SEED_B, {"nb": 5, "tail": tail})
        await conn.execute_write(
            f'CREATE VECTOR INDEX code_embeddings ON :CodeChunk(embedding) '
            f'WITH CONFIG {{"dimension": {_DIM}, "capacity": 10000, "metric": "cos"}};'
        )

        # --- Finding 12: a global CALL leaks across projects; the validator now blocks it. ---
        leaking_query = (
            "MATCH (n:Code {project: $project}) "
            "CALL vector_search.search('code_embeddings', 400, $emb) YIELD node "
            "RETURN DISTINCT node.project AS p"
        )
        # On a *shared* stand where an earlier test already deleted indexed chunks, this
        # Memgraph's HNSW can stay poisoned across DROP/CREATE and a deep vector_search
        # raises "deleted object". That's an external index quirk, not a G5 defect, so skip
        # and rely on the dedicated fresh-throwaway run (where the reset above is clean).
        try:
            rows = await conn.execute_query(leaking_query, {"project": "A", "emb": _QVEC})
        except Exception as exc:  # noqa: BLE001
            if "deleted object" in str(exc):
                pytest.skip("vector index poisoned by a prior test's deletions; run G5 on a fresh throwaway")
            raise
        assert {r["p"] for r in rows} == {"A", "B"}  # BEFORE: B leaked past the A scope
        error = validate_readonly_query(leaking_query)  # AFTER: refused before it runs
        assert error is not None and "CALL" in error
        assert validate_readonly_query("MATCH (n:Code {project: $project}) RETURN n.name") is None

        # --- Finding 11: a rare path filter over a saturated pool signals truncation. ---
        monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [list(_QVEC)])
        mcp = FastMCP("test")
        register_search_tools(mcp, _FixedManager("A"), GraphRepository(conn), Settings(_env_file=None))

        rare = _payload(await mcp.call_tool(
            "search_code",
            {"directory": "/x", "query": "alpha", "include_paths": ["rare/**"]},
        ))
        assert rare["items"] == []  # BEFORE: this empty read as "no matches"
        assert rare["page"]["truncated"] is True  # AFTER: flagged as an early-stopped scan

        # No filter: the similarity top-k is the answer (no truncation), and project scope
        # (INV-9) holds — not a single row leaks from project B.
        unfiltered = _payload(await mcp.call_tool("search_code", {"directory": "/x", "query": "alpha"}))
        assert unfiltered["items"], "expected project-A hits"
        assert all(h["entity"]["id"].startswith("A/") for h in unfiltered["items"])
        assert "truncated" not in unfiltered["page"]
    finally:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        await conn.close()


# ---------------------------------------------------------------------------
# Finding 13 — dashboard semantic search reports a failed backend
# ---------------------------------------------------------------------------


@live
async def test_dashboard_semantic_reports_failure_not_empty_live(monkeypatch) -> None:
    from inflorescence.dashboard.service import DashboardService

    conn = await _conn()
    try:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        # A real failure mode: no vector indexes -> vector_search.search raises for both
        # backends. Before the fix this was swallowed and returned as an empty result.
        await conn.execute_write("DROP VECTOR INDEX code_embeddings")
        await conn.execute_write("DROP VECTOR INDEX summary_embeddings")
        monkeypatch.setattr(EmbeddingClient, "embed", lambda self, texts: [list(_VEC)])

        settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
        settings.llm_api_key = "test-key"
        service = DashboardService(conn, settings)

        result = await service._semantic_search("A", "alpha")
        assert result["hits"] == []
        assert "note" in result and "unavailable" in result["note"].lower()
    finally:
        # Restore the search circuit for any later test on the shared throwaway DB.
        from inflorescence.db.schema import setup_schema

        await setup_schema(conn)
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        await conn.close()


# ---------------------------------------------------------------------------
# Finding 15 — scan-cap truncation is reported over a large project
# ---------------------------------------------------------------------------

_SEED_BIG = """
UNWIND range(0, $n - 1) AS i
CREATE (:Code:Function {project:'BIG', id:'BIG/n'+toString(i), name:'fn'+toString(i), file_path:'bulk/f'+toString(i)+'.py', line_start:1, line_end:2, summary:'s', node_type:'function', checksum:'x'})
"""


@live
async def test_list_entities_flags_scan_truncation_over_large_project_live() -> None:
    from inflorescence.db.repository import GraphRepository
    from inflorescence.tools.graph import MAX_FILTER_SCAN_ROWS

    conn = await _conn()
    try:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        # More entities than the scan cap, none under rare/ -> the sparse filter walks the
        # cap without a match. The empty page must be flagged truncated (finding 15).
        await conn.execute_write(_SEED_BIG, {"n": MAX_FILTER_SCAN_ROWS + 200})
        mcp = FastMCP("test")
        register_graph_tools(mcp, _FixedManager("BIG"), GraphRepository(conn))

        payload = _payload(await mcp.call_tool(
            "list_entities",
            {"directory": "/x", "include_paths": ["rare/**"], "limit": 10, "offset": 0},
        ))
        assert payload["items"] == []
        assert payload["page"]["truncated"] is True
    finally:
        await conn.execute_write("MATCH (n) DETACH DELETE n")
        await conn.close()
