"""G1 · Concurrent substrate — invariant oracles for INV-4 (one mutation per project, *by construction*).

Two classes of oracle:

* **Unit** (no DB): finding 2 — a shared ``GraphBuilder`` must not let two concurrent runs
  overwrite each other's per-run state. Now that ``build()`` returns a :class:`BuildResult`
  instead of stashing state on ``self``, this holds by construction.
* **Live** (real Memgraph, gated on ``INFLORESCENCE_TEST_MEMGRAPH_URL``): finding 1 — the
  cross-process lease. Skipped unless that env var points at a **throwaway** instance, so CI
  and the developer's real ``bolt://localhost:7687`` are never touched. A race between two
  writers cannot be reproduced against a mock; it needs a real graph.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import CodeNode, IndexerConfig, NodeType
from inflorescence.config import Settings

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)


# ---------------------------------------------------------------------------
# Finding 2 — BuildResult isolation (unit, no DB)
# ---------------------------------------------------------------------------


class _SlowRepo:
    async def upsert_external_nodes(self, project: str, refs: list) -> int:
        return len(refs)

    async def delete_orphan_external_nodes(self, project: str) -> int:
        return 0

    """Minimal repo whose writes yield control, so two concurrent builds truly interleave."""

    async def get_node_ids(self, project: str) -> set[str]:
        await asyncio.sleep(0)
        return set()

    async def get_stored_node_index(self, project: str) -> list[tuple[str, str, str]]:
        await asyncio.sleep(0)
        return []

    async def get_code_edges(self, project: str) -> list[tuple[str, str, str]]:
        await asyncio.sleep(0)
        return []

    async def delete_edges_by_keys(self, project: str, edges: list) -> int:
        await asyncio.sleep(0)
        return len(edges)

    async def upsert_nodes(self, project: str, nodes: list) -> int:
        await asyncio.sleep(0.01)  # hold the event loop so the sibling build runs meanwhile
        return len(nodes)

    async def delete_nodes_by_ids(self, project: str, ids: list) -> int:
        await asyncio.sleep(0)
        return len(ids)

    async def delete_code_edges(self, project: str) -> None:
        await asyncio.sleep(0)

    async def upsert_edges(self, project: str, edges: list) -> int:
        await asyncio.sleep(0)
        return len(edges)

    async def update_checksums(self, project: str, items: list) -> int:
        await asyncio.sleep(0)
        return len(items)

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        return {}

    async def store_summaries(self, project: str, items: list) -> int:
        await asyncio.sleep(0)
        return len(items)

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_concurrent_builds_on_shared_builder_do_not_cross_contaminate(tmp_path):
    """INV-4/finding 2: one builder, two projects, concurrent — each run keeps its own state.

    Before the fix, ``build()`` wrote ``last_dirty_summary_ids`` onto the shared instance; two
    concurrent runs clobbered it between ``await`` points, so a run embedded the *other* run's
    dirty set (money-loss / durable wrong-results). Returning per-run state removes the shared
    mutable field entirely, so there is nothing left to clobber.
    """
    a = tmp_path / "a"
    a.mkdir()
    (a / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "b.py").write_text("def bar():\n    return 2\n\n\ndef baz():\n    return 3\n", encoding="utf-8")

    builder = GraphBuilder(
        repo=_SlowRepo(),
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
    )

    result_a, result_b = await asyncio.gather(builder.build("A", a), builder.build("B", b))

    ids_a = {n.id for n in result_a.nodes}
    ids_b = {n.id for n in result_b.nodes}
    assert ids_a and ids_b and ids_a != ids_b  # genuinely different node sets
    # Each result's dirty set is its own project's — not the last writer's.
    assert result_a.dirty_summary_ids == ids_a
    assert result_b.dirty_summary_ids == ids_b
    # The shared-state footgun is gone: no per-run field survives on the instance.
    assert not hasattr(builder, "last_dirty_summary_ids")
    assert not hasattr(builder, "last_stored_summaries")
    assert not hasattr(builder, "last_skipped_large_files")


# ---------------------------------------------------------------------------
# Finding 1 — cross-process lease (live, real Memgraph only)
# ---------------------------------------------------------------------------


async def _fresh_conn():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    return conn


async def _drop_lock(conn, project: str) -> None:
    await conn.execute_write("MATCH (l:IndexLock {project: $p}) DETACH DELETE l", {"p": project})


@live
@pytest.mark.asyncio
async def test_lease_is_mutually_exclusive_across_connections():
    """INV-4: two *separate connections* (= two processes) cannot both hold one project's lease."""
    from inflorescence.db.lease import ProjectLease, ProjectLockTimeoutError

    conn1 = await _fresh_conn()
    conn2 = await _fresh_conn()
    project = f"lease-mx-{uuid.uuid4().hex[:8]}"
    try:
        async with ProjectLease(conn1, project):
            # A second process must not be able to acquire while the first holds it.
            with pytest.raises(ProjectLockTimeoutError):
                async with ProjectLease(conn2, project, timeout_s=1.0, poll_s=0.1):
                    pass
        # Once released, the second process acquires cleanly.
        async with ProjectLease(conn2, project, timeout_s=3.0, poll_s=0.1):
            pass
    finally:
        await _drop_lock(conn1, project)
        await conn1.close()
        await conn2.close()


@live
@pytest.mark.asyncio
async def test_lease_is_taken_over_after_holder_crashes():
    """INV-8: a holder killed mid-run (heartbeat stops) must not wedge the project forever."""
    from inflorescence.db.lease import ProjectLease

    conn1 = await _fresh_conn()
    conn2 = await _fresh_conn()
    project = f"lease-crash-{uuid.uuid4().hex[:8]}"
    try:
        # Acquire with a short TTL and a heartbeat that never fires within the test, then
        # "crash": cancel the heartbeat task and never release.
        crashed = ProjectLease(conn1, project, ttl_s=1.0, heartbeat_s=1000.0)
        await crashed.__aenter__()
        assert crashed._heartbeat_task is not None
        crashed._heartbeat_task.cancel()  # simulate process death: no more renewals, no release

        # A fresh acquirer takes the lease over once the stale one expires.
        async with ProjectLease(conn2, project, ttl_s=5.0, timeout_s=5.0, poll_s=0.2):
            pass
    finally:
        await _drop_lock(conn1, project)
        await conn1.close()
        await conn2.close()


class _CountingBuilder:
    """Fake builder that records the peak number of concurrently-running mutations."""

    def __init__(self, state: dict) -> None:
        self._state = state  # shared across the two managers ("two processes")

    async def preflight(self) -> None:
        return None

    def estimate_paths(self, root_path, paths) -> dict:
        return {"estimated_tokens": 0, "estimated_nodes": 0}

    async def has_pending_changes(self, project, root_path, changed_files=None, **_kw) -> bool:
        return True  # always "work to do" so the update path (and its lease) is exercised

    async def update(self, project, root_path, changed_files=None, **_kw):
        from inflorescence.code_indexer.graph_builder import BuildResult

        self._state["in_flight"] += 1
        self._state["max"] = max(self._state["max"], self._state["in_flight"])
        try:
            await asyncio.sleep(0.15)  # long enough for a racing writer to overlap if unserialized
        finally:
            self._state["in_flight"] -= 1
        return BuildResult(nodes=[object()], edges=[])  # non-empty so the pipeline proceeds


class _NoopRepo:
    async def upsert_external_nodes(self, project: str, refs: list) -> int:
        return len(refs)

    async def delete_orphan_external_nodes(self, project: str) -> int:
        return 0

    async def get_project_stats(self, project):
        return {"nodes": 1, "edges": 0, "chunks": 0, "summaries": 0}

    async def get_stored_chunk_embeddings(self, project):
        return {}

    async def set_project_root_path(self, project, root_path):
        return None

    async def get_project_scope(self, project):
        return [], []


class _NoopRag:
    def preflight(self) -> None:
        return None

    async def index_code(self, *a, **k):
        return 0

    async def index_summaries(self, *a, **k):
        return 0


async def _run_two_writers(conn, project_dir: str, *, use_lease: bool) -> int:
    """Two ProjectManagers, each with its own connection state, racing one project.

    Separate managers ⇒ separate per-project ``asyncio.Lock`` dicts — exactly like two OS
    processes. Only the cross-process DB lease can keep them from overlapping. Returns the peak
    concurrency observed inside the mutation critical section.
    """
    from inflorescence.project_manager import ProjectManager

    state = {"in_flight": 0, "max": 0}
    lease_conn = conn if use_lease else None

    def make_manager():
        mgr = ProjectManager(
            repo=_NoopRepo(),
            graph_builder=_CountingBuilder(state),
            rag_indexer=_NoopRag(),
            settings=Settings(_env_file=None, memgraph_url=LIVE_URL),
            conn=lease_conn,
        )
        project = mgr.get_project(project_dir)  # populate the path→project map for update_project
        return mgr, project

    m1, project = make_manager()
    m2, _ = make_manager()
    await asyncio.gather(
        m1.update_project(project, ["x.py"]),
        m2.update_project(project, ["y.py"]),
    )
    return state["max"]


@live
@pytest.mark.asyncio
async def test_cross_process_writers_are_serialized_only_by_the_lease(tmp_path):
    """Finding 1: the in-process asyncio.Lock does NOT serialize two processes — the lease does."""
    conn = await _fresh_conn()
    project_dir = str(tmp_path)
    # Resolve the project id the manager will derive, so we can clean its lock node up.
    from inflorescence.project_manager import _project_name

    project = _project_name(tmp_path)
    try:
        # Without the DB lease, two separate managers (processes) overlap — the very race
        # that lets one reconcile delete the other's fresh writes.
        overlap_without = await _run_two_writers(conn, project_dir, use_lease=False)
        assert overlap_without == 2, "expected the two 'processes' to overlap without a shared lock"

        # With the lease, they run strictly one at a time.
        overlap_with = await _run_two_writers(conn, project_dir, use_lease=True)
        assert overlap_with == 1, "the cross-process lease must serialize the two writers"
    finally:
        await _drop_lock(conn, project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_store_racing_delete_loses_write_without_lease_but_not_with():
    """Finding 1 consequence — the "1660 re-embedded, stored 0" class, on a real graph.

    A store phase whose nodes are deleted by a concurrent reconcile writes **0** rows and the
    data is gone. Under the lease a store phase can never be interleaved by another writer's
    delete, so it always reflects the true count.
    """
    from inflorescence.db.lease import project_lease
    from inflorescence.db.repository import GraphRepository

    conn = await _fresh_conn()
    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    repo = GraphRepository(conn)
    dim = settings.embedding_dimension

    def emb(seed: int) -> list[float]:
        v = [0.0] * dim
        v[seed % dim] = 1.0  # unit vector — safe for the cosine index
        return v

    node = CodeNode(
        id="m.py", name="m", node_type=NodeType.MODULE, file_path="m.py",
        line_start=1, line_end=1, summary="a summary",
    )

    # ---- BEFORE (no lease): delete lands between upsert and store -> "stored 0", data lost.
    project_before = f"race-nolease-{uuid.uuid4().hex[:8]}"
    try:
        await repo.upsert_nodes(project_before, [node])          # writer B: fresh write
        await repo.delete_nodes_by_ids(project_before, [node.id])  # writer A's reconcile deletes it
        written = await repo.store_summary_embeddings(            # writer B stores into thin air
            project_before, [{"node_id": node.id, "embedding": emb(1)}]
        )
        assert written == 0, "reproduces the 'stored 0' data loss when a store races a delete"
        remaining = await repo.get_node_ids(project_before)
        assert node.id not in remaining  # the write is gone
    finally:
        await _drop_lock(conn, project_before)

    # ---- AFTER (lease): the store writer's upsert+store is atomic w.r.t. a concurrent delete.
    project_after = f"race-lease-{uuid.uuid4().hex[:8]}"
    store_written: dict[str, int] = {}
    b_holds = asyncio.Event()

    async def writer_store():
        async with project_lease(conn, project_after):
            await repo.upsert_nodes(project_after, [node])
            b_holds.set()
            store_written["n"] = await repo.store_summary_embeddings(
                project_after, [{"node_id": node.id, "embedding": emb(2)}]
            )

    async def writer_delete():
        await b_holds.wait()  # try hardest to interleave — but the lease blocks us until B is done
        async with project_lease(conn, project_after):
            await repo.delete_nodes_by_ids(project_after, [node.id])

    try:
        await asyncio.gather(writer_store(), writer_delete())
        # The store completed against live nodes — never the corrupt "stored 0".
        assert store_written["n"] == 1
    finally:
        await _drop_lock(conn, project_after)
        await conn.close()
