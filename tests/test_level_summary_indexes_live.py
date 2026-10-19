"""Live oracle: directories and modules are ranked in summary indexes of their own.

Memgraph's ``vector_search.search`` returns the GLOBAL top-k of an index and only then lets
Cypher filter by project and type. In the shared ``summary_embeddings`` index — every entity
of every project — a project's few dozen directories practically never reach that pool, so
``search_semantic(node_types=["directory"])`` answered an empty page over a project with 39
of them (probed 2026-09-29 on a machine holding 61k summary vectors). The widening window
now reaches them eventually; the label-scoped indexes in ``schema.LEVEL_SUMMARY_INDEXES``
rank directories against directories, so they are the first window's answer.

Two facts about the engine this relies on can only be checked live, so both are asserted
here: that a second vector index on the same property under another label coexists with the
shared one, and that an index created over nodes that already carry the property includes
them (a database upgraded in place gets the index at its next ``setup_schema``, over data
indexed long before). Skipped unless ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a
*throwaway* Memgraph-MAGE instance.
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository
from tests.test_memgraph_quirks import LIVE_URL, _vec, _wait_for_a_quiet_vector_index, live

# More functions closer to the query than the directory than the shared search fetches.
_CROWD = 400


async def _connect():
    from inflorescence.db.connection import MemgraphConnection

    return MemgraphConnection(Settings(_env_file=None, memgraph_url=LIVE_URL))


async def _seed(conn, project: str) -> None:
    # Distinct vectors, all closer to the query than the directory: HNSW recall over hundreds
    # of *identical* vectors is lossy and nondeterministic, which would test the engine, not us.
    await conn.execute_write(
        "UNWIND range(1, size($near)) AS i "
        "CREATE (:Code:Function {id: $project + '::f' + toString(i), project: $project, "
        "name: 'f' + toString(i), node_type: 'function', file_path: 'src/pkg/m.py', "
        "line_start: i, line_end: i, summary: 'crowd', summary_embedding: $near[i - 1]})",
        {"project": project, "near": [_vec(i / 1000) for i in range(1, _CROWD + 1)]},
    )
    await conn.execute_write(
        "CREATE (:Code:Directory {id: 'dir:src/pkg', project: $project, name: 'pkg', "
        "node_type: 'directory', file_path: 'src/pkg', line_start: 0, line_end: 0, "
        "summary: 'The package.', summary_embedding: $far})",
        {"project": project, "far": _vec(3.0)},
    )


async def _drop(conn, project: str) -> None:
    await conn.execute_write("MATCH (n:Code {project: $project}) DETACH DELETE n", {"project": project})


@live
@pytest.mark.asyncio
async def test_a_directory_crowded_out_of_the_shared_index_is_found_in_its_own() -> None:
    from inflorescence.db.schema import setup_schema

    conn = await _connect()
    project = f"levels-{uuid.uuid4().hex[:8]}"
    # The in-place upgrade path: data first, then the schema pass that adds the index.
    with contextlib.suppress(Exception):  # absent on a fresh stand, which is fine
        await conn.execute_write("DROP VECTOR INDEX directory_summary_embeddings")
    try:
        await _seed(conn, project)
        missing = await setup_schema(conn)
        assert missing == []
        repo = GraphRepository(conn)
        # Another live test's deletion can leave every vector search failing for ~20 s.
        await _wait_for_a_quiet_vector_index(repo, project)

        # The crowd, reproduced: the shared index's top rows are all closer functions.
        shared = await repo.search_summary_vectors(project, _vec(0.0), top_k=10)
        assert "dir:src/pkg" not in [row["id"] for row in shared]

        def directories(rows):
            return [row for row in rows if row.get("node_type") == "directory"]

        levelled = await repo.vector_hits("summary", project, _vec(0.0), 10, directories, node_types=["directory"])
        assert [row["id"] for row in levelled.rows] == ["dir:src/pkg"]
        assert levelled.complete and levelled.fallback_from == ()
        # The shared index reaches the same answer by widening past the crowd — same set,
        # only more work, which is all the level index saves.
        widened = await repo.vector_hits("summary", project, _vec(0.0), 10, directories)
        assert [row["id"] for row in widened.rows] == ["dir:src/pkg"]

        # And the path a top-down walk takes: a directory addressed by its (normalized) path.
        root = await repo.get_structure_root(project, file_path="src/pkg")
        assert root is not None and root["id"] == "dir:src/pkg"
    finally:
        await _drop(conn, project)
        await conn.close()
