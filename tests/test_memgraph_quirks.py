"""Live oracles for two Memgraph 3.12 engine quirks the repository has to work around.

Neither defect is in our logic — both are in how this Memgraph executes perfectly legal
Cypher — and neither can be reproduced against a mock, so both oracles are **live**: they
are skipped unless ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a *throwaway* instance,
so CI and a developer's real ``bolt://localhost:7687`` are never touched. No LLM or
embedding key is needed: both quirks are structural.

**Quirk 1 — a batched write whose WHERE is a disjunction of pure label tests silently
writes only the first row.** With the project's schema in place (``Module.project`` /
``Document.project`` label-property indexes exist), Memgraph 3.12.0 plans
``UNWIND $rows … MATCH (m:Code {…}) WHERE m:Module OR m:Document SET …`` as an indexed
union scan and applies the SET to exactly ONE node, no matter how many rows matched —
and ``RETURN count(*)`` reports that same 1, so the caller cannot tell it was cheated.
The identical query with ``WHERE m.node_type IN ['module','document']`` writes all rows.
Only the *pure* label disjunction after an UNWIND is affected; the same disjunction in a
read, in a non-UNWIND write, or AND-ed with a non-label predicate executes correctly
(probed live — see the report). For ``MARK_CHUNKS_COVERED`` this was money: with the
marker stamped on one file per batch, ``GET_FILES_WITH_STALE_CHUNKS`` kept reporting the
rest as never-chunked, so every re-index re-chunked and re-embedded unchanged files
forever (INV-5).

**Quirk 2 — deleting any project blinds vector search for every project.** Memgraph's
vector indexes keep pointing at the nodes a DETACH DELETE removed until its garbage
collector catches up (~20 s). In that window *every* ``vector_search.search`` call —
including ones for projects that were never touched — fails outright with
``Memgraph.ClientError … Trying to get a property from a deleted object``. Since the
worktree janitor deletes projects on its own schedule, an agent's search could fail for
reasons nothing in its own request explains. The repository now retries that one error
for a bounded window and, if it never clears, replaces the bare engine message with one
that says what happened and when to retry.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from inflorescence.config import Settings
from inflorescence.db import repository as repository_module
from inflorescence.db.repository import GraphRepository

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536


async def _fresh_repo():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    # The label-property indexes are what make quirk 1 reproduce, so the oracle is only
    # meaningful against the real schema — never against a bare graph.
    await setup_schema(conn)
    return GraphRepository(conn), conn


def _vec(seed: float) -> list[float]:
    v = [0.0] * _DIM
    v[0] = 1.0
    v[1] = seed
    return v


async def _wait_for_a_quiet_vector_index(repo: GraphRepository, project: str) -> None:
    """Block until vector search works at all, so quirk 2's oracle starts from a clean slate.

    On a shared stand this file is not the first to delete a project: an earlier test's
    teardown can leave the vector index stale before this one even begins, and then the
    oracle would be measuring that test's deletion instead of its own. Waiting for a
    working search first makes the experiment start from a known-good baseline. If the
    stand never settles the oracle is skipped rather than reported as a defect — the
    dedicated fresh-throwaway run is where it must be conclusive.
    """
    deadline = asyncio.get_running_loop().time() + 90.0
    while True:
        try:
            if await repo.search_summary_vectors(project, _vec(0.0), top_k=10):
                return
        except repository_module.StaleVectorIndexError:
            pass
        if asyncio.get_running_loop().time() > deadline:
            pytest.skip("vector index never settled on this shared stand; run on a fresh throwaway")
        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Quirk 1 — batched coverage marker must reach every row of the batch
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_mark_chunks_covered_stamps_every_row_of_the_batch() -> None:
    """Every file in one ``mark_files_chunks_covered`` batch must end up stamped.

    Regression for the pure-label-disjunction quirk: with ``WHERE m:Module OR m:Document``
    exactly one of these nodes was stamped and ``written`` reported 1. The oracle asserts
    both the reported count and the state actually in the database (the reported count
    alone would have been a lie either way), and then asserts the consequence that costs
    money: ``get_files_with_stale_chunks`` must stop naming these files.
    """
    repo, conn = await _fresh_repo()
    project = f"quirk1-{uuid.uuid4().hex[:8]}"
    modules = [f"src/mod_{i}.py" for i in range(5)]
    documents = [f"docs/doc_{i}.md" for i in range(3)]
    try:
        for path in modules:
            await conn.execute_write(
                "CREATE (m:Code:Module {id: $id, project: $project, file_path: $path, "
                "node_type: 'module', checksum: $sum})",
                {"id": path, "project": project, "path": path, "sum": f"sum-{path}"},
            )
        for path in documents:
            await conn.execute_write(
                "CREATE (m:Code:Document {id: $id, project: $project, file_path: $path, "
                "node_type: 'document', checksum: $sum})",
                {"id": path, "project": project, "path": path, "sum": f"sum-{path}"},
            )
        every_file = modules + documents
        assert set(await repo.get_files_with_stale_chunks(project)) == set(every_file)

        written = await repo.mark_files_chunks_covered(
            project, [{"file_path": p, "checksum": f"sum-{p}"} for p in every_file]
        )
        assert written == len(every_file)  # BEFORE the fix: 1

        stamped = await conn.execute_query(
            "MATCH (m:Code {project: $project}) WHERE m.chunks_checksum IS NOT NULL "
            "RETURN m.file_path AS file_path, m.chunks_checksum AS marker",
            {"project": project},
        )
        assert {r["file_path"] for r in stamped} == set(every_file)
        assert all(r["marker"] == f"sum-{r['file_path']}" for r in stamped)

        # The money consequence: nothing is re-chunked and re-embedded next run.
        assert await repo.get_files_with_stale_chunks(project) == []
    finally:
        await conn.execute_write("MATCH (n {project: $project}) DETACH DELETE n", {"project": project})
        await conn.close()


# ---------------------------------------------------------------------------
# Quirk 2 — the retry helper, unit-tested against a fake connection
# ---------------------------------------------------------------------------


class _StaleVectorIndexError(Exception):
    """Stands in for the neo4j ClientError Memgraph raises while its GC is behind."""

    def __init__(self) -> None:
        super().__init__(
            "{code: Memgraph.ClientError.MemgraphError.MemgraphError} {message: "
            "Trying to get a property from a deleted object.}"
        )


class _FakeConn:
    """A connection whose ``execute_query`` fails ``failures`` times, then answers."""

    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self._remaining = failures
        self._error = error or _StaleVectorIndexError()
        self.calls = 0

    async def execute_query(self, query, params=None, max_records=None):  # noqa: ANN001, ANN202
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise self._error
        return [
            {
                "id": "a.py",
                "name": "a",
                "labels": ["Code", "Module"],
                "file_path": "a.py",
                "line_start": 1,
                "line_end": 2,
                "summary": "s",
                "score": 0.9,
                "chunk_id": "a.py#0",
                "chunk_kind": "code",
                "match_start_line": 1,
                "match_end_line": 2,
            }
        ]


@pytest.fixture
def _fast_retries(monkeypatch: pytest.MonkeyPatch):
    """Shrink the backoff so the unit tests exercise the policy, not the clock."""
    monkeypatch.setattr(repository_module, "_VECTOR_RETRY_BASE_DELAY_S", 0.001)
    monkeypatch.setattr(repository_module, "_VECTOR_RETRY_MAX_DELAY_S", 0.002)
    monkeypatch.setattr(repository_module, "_VECTOR_RETRY_BUDGET_S", 0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search_summary_vectors", "search_code_vectors"])
async def test_vector_search_recovers_when_the_gc_catches_up(method: str, _fast_retries) -> None:
    """N transient "deleted object" failures then success: the caller sees only the success."""
    conn = _FakeConn(failures=3)
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    rows = await getattr(repo, method)("proj", _vec(0.1), top_k=5)
    assert conn.calls == 4
    assert len(rows) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search_summary_vectors", "search_code_vectors"])
async def test_vector_search_does_not_retry_an_unrelated_error(method: str, _fast_retries) -> None:
    """Any other failure is a real failure: it propagates immediately, unaltered.

    A retry loop that swallowed every exception would turn a syntax error or a dead
    connection into an eight-second stall and then a misleading "recent deletion" story.
    """
    conn = _FakeConn(failures=99, error=RuntimeError("index 'summary_embeddings' does not exist"))
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="does not exist"):
        await getattr(repo, method)("proj", _vec(0.1), top_k=5)
    assert conn.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search_summary_vectors", "search_code_vectors"])
async def test_vector_search_exhaustion_explains_itself(method: str, _fast_retries) -> None:
    """When the window closes without recovery the caller gets an explanation, not the raw error.

    The message reaches a calling agent verbatim, so it has to carry the cause (a recent
    project deletion left the vector index stale) and the remedy (retry in ~20 s) — the
    bare engine text says neither. The original error is kept as ``__cause__``.
    """
    conn = _FakeConn(failures=99)
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    with pytest.raises(Exception) as excinfo:
        await getattr(repo, method)("proj", _vec(0.1), top_k=5)
    message = str(excinfo.value)
    assert "deleted" in message
    assert "20 second" in message
    assert "vector index" in message.lower()
    assert conn.calls > 1
    assert isinstance(excinfo.value.__cause__, _StaleVectorIndexError)


@pytest.mark.asyncio
async def test_vector_retry_budget_is_bounded(_fast_retries) -> None:
    """The loop gives up inside its budget instead of hanging the caller forever."""
    conn = _FakeConn(failures=99)
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    start = asyncio.get_running_loop().time()
    with pytest.raises(Exception):  # noqa: B017 — the type is asserted in the test above
        await repo.search_summary_vectors("proj", _vec(0.1), top_k=5)
    assert asyncio.get_running_loop().time() - start < 1.0


def test_vector_retry_budget_default_is_short_enough_for_a_tool_call() -> None:
    """The shipped budget must stay well inside an interactive call, not the full GC cycle."""
    assert 0 < repository_module._VECTOR_RETRY_BUDGET_S <= 8.0


# ---------------------------------------------------------------------------
# Quirk 2 — live oracle
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_vector_search_survives_another_projects_deletion() -> None:
    """Deleting project A must not leave project B's search failing with a bare engine error.

    Two outcomes are acceptable and both are the fix working: the retry outlives the GC lag
    and the search answers, or the window closes and the caller gets the enriched message.
    The third outcome — the raw ``Trying to get a property from a deleted object`` reaching
    the caller with no explanation — is the defect, and is what this asserts against.
    """
    repo, conn = await _fresh_repo()
    victim = f"quirk2-victim-{uuid.uuid4().hex[:8]}"
    survivor = f"quirk2-survivor-{uuid.uuid4().hex[:8]}"
    try:
        for project in (victim, survivor):
            for i in range(3):
                node_id = f"{project}/f{i}.py"
                await conn.execute_write(
                    "CREATE (n:Code:Module {id: $id, project: $project, file_path: $path, "
                    "node_type: 'module', name: $name, line_start: 1, line_end: 2, "
                    "summary: 'a summary', summary_embedding: $emb})",
                    {"id": node_id, "project": project, "path": f"f{i}.py",
                     "name": f"f{i}", "emb": _vec(0.01 * i)},
                )
                await conn.execute_write(
                    "MATCH (n:Code {id: $id, project: $project}) "
                    "CREATE (n)-[:HAS_CHUNK]->(c:CodeChunk {id: $cid, project: $project, "
                    "node_id: $id, file_path: $path, content: 'x', start_line: 1, end_line: 2, "
                    "chunk_kind: 'code', embedding: $emb})",
                    {"id": node_id, "cid": f"{node_id}#0", "project": project,
                     "path": f"f{i}.py", "emb": _vec(0.01 * i)},
                )

        await _wait_for_a_quiet_vector_index(repo, survivor)

        # The trigger: any project's deletion poisons the shared vector indexes.
        await repo.delete_project(victim)

        for search in (repo.search_summary_vectors, repo.search_code_vectors):
            try:
                await search(survivor, _vec(0.0), top_k=10)
            except Exception as exc:  # noqa: BLE001 — the message is what is under test
                message = str(exc)
                assert "20 second" in message, f"bare engine error reached the caller: {message}"
                assert "vector index" in message.lower(), message
    finally:
        for project in (victim, survivor):
            await conn.execute_write(
                "MATCH (n {project: $project}) DETACH DELETE n", {"project": project}
            )
        await conn.close()
