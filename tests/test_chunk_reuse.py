"""An incremental index must not re-write chunks that did not change.

A watcher flush re-chunks the *whole* project — it has to, because a one-line edit can move
what another file is recorded as calling — and then used to store every chunk it produced,
thousands of rows whose content, vector and every other property were byte-identical to what
was already there. Each of those writes costs a Bolt round trip and a vector-index update; on
a 3k-chunk project it was ~75% of the wall time of saving one file (INV-5).

The rows are now skipped, and skipping is only safe if the rest of the machinery keeps seeing
the same world. That is what this file pins down:

  * a chunk the database already holds exactly as this run would write it is neither embedded
    nor stored — but is still **kept** by pruning and still counts towards its file's chunk
    **coverage** marker, so no live chunk is deleted, no dead chunk survives, and the repair
    pass is not handed a file that looks permanently stale;
  * "already holds it" is decided precisely: the chunk id proves the content, node, file, line
    span and kind; the language and the HAS_CHUNK owner are read back and compared, and a
    stored chunk with no vector does not count. Anything that fails those tests is re-written
    — and its stored vector is recovered so the rewrite costs no embedding call;
  * an embedding failure still suppresses pruning and coverage, exactly as before.

The live half proves the Cypher behind all of it against a real Memgraph. It is skipped
unless ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a **throwaway** instance.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from inflorescence.code_indexer.models import CodeNode, NodeType
from inflorescence.config import Settings
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import setup_schema
from inflorescence.rag.chunker import Chunk
from inflorescence.rag.indexer import RAGIndexer, _chunk_id

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

MODULE_NODE = CodeNode(
    id="module.py::alpha",
    name="alpha",
    node_type=NodeType.FUNCTION,
    file_path="module.py",
    line_start=1,
    line_end=2,
)


def _chunk(content: str, start: int, language: str = "python", node_id: str = "module.py::alpha") -> Chunk:
    return Chunk(
        content=content,
        node_id=node_id,
        file_path="module.py",
        start_line=start,
        end_line=start,
        language=language,
        chunk_index=start - 1,
        chunk_kind="function_body",
    )


class FakeChunker:
    """Returns a fixed chunk list, so a test controls exactly what a run produces."""

    def __init__(self, chunks: Sequence[Chunk] | None = None) -> None:
        self.chunks = list(chunks) if chunks is not None else [
            _chunk("first chunk", 1), _chunk("second chunk", 2)
        ]

    def chunk_repository(
        self, root: Path, nodes: list[CodeNode], unreadable_out: set[str] | None = None
    ) -> list[Chunk]:
        return list(self.chunks)

    def chunk_file(self, file_path: Path, root: Path) -> list[Chunk]:
        return list(self.chunks)


class CountingEmbedder:
    """Hands out one vector per text and records every text it was asked to embed.

    Counting *texts* rather than calls is the point: the claim is that an unchanged chunk
    costs no embedding, and a batched embedder would hide that behind a single call.
    """

    def __init__(self, vector: list[float] | None = None, fail_texts: set[str] | None = None) -> None:
        self.model = "fake-model"
        self.vector = vector if vector is not None else [0.5, 0.5]
        self.fail_texts = fail_texts or set()
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        self.texts.extend(texts)
        return [None if t in self.fail_texts else list(self.vector) for t in texts]


class FakeChunkStore:
    """An in-memory stand-in for the chunk half of ``GraphRepository``.

    Models what ``STORE_CODE_CHUNKS_BATCH`` actually does — upsert the row *and* the
    HAS_CHUNK edge from its owning node — so the skip logic is tested against a store that
    can be inconsistent in the ways the real one can (a row with no vector, a row whose
    owner's edge is gone), rather than against a dictionary that never is.
    """

    def __init__(self, checksums: Mapping[str, str] | None = None) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.owners: dict[str, set[str]] = {}
        self.chunk_files: dict[str, str] = {}
        self.checksums = dict(checksums or {})
        self.chunks_covered: list[dict[str, object]] = []
        self.deleted_chunk_ids: list[str] = []
        self.store_calls = 0
        self.fact_reads: list[list[str]] = []
        self.vector_reads: list[list[str]] = []

    # -- the store ---------------------------------------------------------
    async def store_code_chunks(self, project: str, chunks: Sequence[Mapping[str, object]]) -> int:
        self.store_calls += 1
        for row in chunks:
            chunk_id = str(row["chunk_id"])
            self.rows[chunk_id] = dict(row)
            self.owners.setdefault(chunk_id, set()).add(str(row["node_id"]))
            self.chunk_files[chunk_id] = str(row["file_path"])
        return len(chunks)

    # -- the reads the skip decision uses ----------------------------------
    async def get_stored_chunk_facts(
        self, project: str, chunk_ids: Sequence[str]
    ) -> dict[str, dict[str, object]]:
        self.fact_reads.append(list(chunk_ids))
        facts: dict[str, dict[str, object]] = {}
        for chunk_id in chunk_ids:
            row = self.rows.get(chunk_id)
            if row is None or row.get("embedding") is None:
                continue
            facts[chunk_id] = {
                "language": str(row.get("language") or ""),
                "node_ids": set(self.owners.get(chunk_id, set())),
            }
        return facts

    async def get_stored_chunk_embeddings_for_ids(
        self, project: str, chunk_ids: Sequence[str]
    ) -> dict[str, list[float]]:
        self.vector_reads.append(list(chunk_ids))
        return {
            chunk_id: list(self.rows[chunk_id]["embedding"])  # type: ignore[arg-type]
            for chunk_id in chunk_ids
            if self.rows.get(chunk_id, {}).get("embedding") is not None
        }

    # -- pruning and coverage ---------------------------------------------
    async def get_chunk_ids(self, project: str) -> set[str]:
        return set(self.rows)

    async def get_chunk_index(self, project: str) -> dict[str, str]:
        return dict(self.chunk_files)

    async def delete_chunks_by_ids(self, project: str, ids: Sequence[str]) -> int:
        for chunk_id in ids:
            self.deleted_chunk_ids.append(chunk_id)
            self.rows.pop(chunk_id, None)
            self.owners.pop(chunk_id, None)
            self.chunk_files.pop(chunk_id, None)
        return len(ids)

    async def get_checksums(self, project: str) -> dict[str, str]:
        return dict(self.checksums)

    async def mark_files_chunks_covered(
        self, project: str, items: Sequence[Mapping[str, object]]
    ) -> int:
        self.chunks_covered.extend(dict(item) for item in items)
        return len(items)


def _indexer(repo: object, embedder: CountingEmbedder, chunker: FakeChunker) -> RAGIndexer:
    indexer = RAGIndexer(repo, Settings(_env_file=None), chunker=chunker)  # type: ignore[arg-type]
    indexer._code_embedder = embedder  # type: ignore[assignment]
    return indexer


# ---------------------------------------------------------------------------
# The skip itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_run_over_unchanged_files_writes_and_embeds_nothing(tmp_path: Path) -> None:
    """The headline claim: re-indexing unchanged content costs neither an embedding nor a write."""
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    chunker = FakeChunker()
    embedder = CountingEmbedder()

    first = await _indexer(repo, embedder, chunker).index_code("p", tmp_path, [MODULE_NODE])
    assert first == 2 and len(embedder.texts) == 2 and repo.store_calls == 1

    embedder = CountingEmbedder()
    second = await _indexer(repo, embedder, chunker).index_code("p", tmp_path, [MODULE_NODE])

    assert second == 0, "an unchanged chunk was written again"
    assert embedder.texts == [], "an unchanged chunk was embedded again"
    assert repo.store_calls == 1, "the store was called for a window with nothing to write"
    # And the data is still there — skipping is not deleting.
    assert len(repo.rows) == 2


@pytest.mark.asyncio
async def test_only_the_changed_chunk_is_rewritten(tmp_path: Path) -> None:
    """One edited chunk among many is the whole scenario this exists for."""
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    embedder = CountingEmbedder()
    await _indexer(repo, embedder, FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])

    edited = FakeChunker([_chunk("first chunk", 1), _chunk("second chunk EDITED", 2)])
    embedder = CountingEmbedder()
    written = await _indexer(repo, embedder, edited).index_code(
        "p", tmp_path, [MODULE_NODE], prune_stale=True, reconcilable_files={"module.py"}
    )

    assert written == 1
    assert embedder.texts == ["second chunk EDITED"]
    # The unchanged chunk survived pruning; only the superseded one was deleted.
    assert repo.deleted_chunk_ids == [_chunk_id(_chunk("second chunk", 2))]
    assert _chunk_id(_chunk("first chunk", 1)) in repo.rows
    assert _chunk_id(_chunk("second chunk EDITED", 2)) in repo.rows


@pytest.mark.asyncio
async def test_skipped_chunks_are_kept_by_pruning(tmp_path: Path) -> None:
    """The prune must see skipped chunks as *this run's own* chunks, not as absent ones.

    This is the failure mode that would make the optimisation catastrophic rather than merely
    wrong: chunks skipped because they were already correct, then deleted for not having been
    written by this run.
    """
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])
    stored_ids = set(repo.rows)

    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code(
        "p", tmp_path, [MODULE_NODE], prune_stale=True, reconcilable_files={"module.py"}
    )

    assert repo.deleted_chunk_ids == []
    assert set(repo.rows) == stored_ids


@pytest.mark.asyncio
async def test_a_file_whose_chunks_were_all_skipped_is_still_marked_covered(tmp_path: Path) -> None:
    """Coverage says "this file's chunks are fully stored for this checksum" — which is just
    as true when the run wrote none of them. Leaving it unstamped would make
    ``GET_FILES_WITH_STALE_CHUNKS`` report the file forever, and every re-index would
    re-chunk and re-embed it: the endless paid loop the marker exists to prevent.
    """
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])
    repo.chunks_covered.clear()

    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])

    assert repo.chunks_covered == [{"file_path": "module.py", "checksum": "c1"}]


@pytest.mark.asyncio
async def test_a_failed_embedding_still_suppresses_pruning_and_coverage(tmp_path: Path) -> None:
    """The "no embedding failed" gate is unchanged by skipping: a run that could not embed a
    new chunk must not prune, and must not claim any file is covered — even though most of
    its chunks were skipped and therefore trivially fine.
    """
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])
    repo.chunks_covered.clear()

    edited = FakeChunker([_chunk("first chunk", 1), _chunk("second chunk EDITED", 2)])
    embedder = CountingEmbedder(fail_texts={"second chunk EDITED"})
    await _indexer(repo, embedder, edited).index_code(
        "p", tmp_path, [MODULE_NODE], prune_stale=True, reconcilable_files={"module.py"}
    )

    assert repo.deleted_chunk_ids == [], "an API failure was allowed to shrink the chunk set"
    assert repo.chunks_covered == [], "a file with a failed chunk was marked fully covered"


# ---------------------------------------------------------------------------
# What counts as "already stored"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_chunk_stored_without_a_vector_is_rewritten(tmp_path: Path) -> None:
    """A row with no embedding cannot answer a search, so it is not stored in any sense that
    matters — it is re-embedded and written."""
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])
    victim = _chunk_id(_chunk("first chunk", 1))
    repo.rows[victim]["embedding"] = None

    embedder = CountingEmbedder()
    written = await _indexer(repo, embedder, FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])

    assert written == 1
    assert embedder.texts == ["first chunk"]
    assert repo.rows[victim]["embedding"] == [0.5, 0.5]


@pytest.mark.asyncio
async def test_a_chunk_whose_language_changed_is_rewritten_from_the_stored_vector(
    tmp_path: Path,
) -> None:
    """``language`` is the one stored field a matching chunk id does not prove equal, because
    it is not part of the hash. A chunker that re-maps a file's language must not leave the
    old value behind — and re-writing the row must not cost a second embedding of text whose
    vector the database already has.
    """
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(vector=[0.25, 0.75]), FakeChunker()).index_code(
        "p", tmp_path, [MODULE_NODE]
    )
    retyped = FakeChunker([
        _chunk("first chunk", 1, language="starlark"), _chunk("second chunk", 2)
    ])

    embedder = CountingEmbedder(vector=[9.0, 9.0])
    written = await _indexer(repo, embedder, retyped).index_code("p", tmp_path, [MODULE_NODE])

    assert written == 1
    assert embedder.texts == [], "the stored vector was re-embedded instead of reused"
    row = repo.rows[_chunk_id(_chunk("first chunk", 1))]
    assert row["language"] == "starlark"
    assert row["embedding"] == [0.25, 0.75]
    assert repo.vector_reads == [[_chunk_id(_chunk("first chunk", 1))]]


@pytest.mark.asyncio
async def test_a_chunk_whose_has_chunk_edge_is_missing_is_rewritten(tmp_path: Path) -> None:
    """The store creates the HAS_CHUNK edge along with the row, so skipping the store skips
    the edge. A chunk whose owning node was deleted and recreated has the row but not the
    edge; leaving it that way would strand the chunk outside its file's structure.
    """
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])
    orphan = _chunk_id(_chunk("first chunk", 1))
    repo.owners[orphan] = set()

    embedder = CountingEmbedder()
    written = await _indexer(repo, embedder, FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])

    assert written == 1
    assert embedder.texts == [], "re-attaching a chunk cost a paid embedding"
    assert repo.owners[orphan] == {"module.py::alpha"}


@pytest.mark.asyncio
async def test_the_freshness_read_is_scoped_to_this_run_s_chunk_ids(tmp_path: Path) -> None:
    """Scoped by id, not by project: an update must cost in proportion to what it re-chunked,
    which is exactly the property that made the whole-project vector dump it replaces
    unacceptable."""
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    await _indexer(repo, CountingEmbedder(), FakeChunker()).index_code("p", tmp_path, [MODULE_NODE])

    assert repo.fact_reads == [[
        _chunk_id(_chunk("first chunk", 1)), _chunk_id(_chunk("second chunk", 2))
    ]]
    assert repo.vector_reads == [], "vectors were read for chunks nothing was going to rewrite"


@pytest.mark.asyncio
async def test_a_donor_vector_still_seeds_a_project_that_stores_nothing_yet(tmp_path: Path) -> None:
    """Cross-project reuse is untouched: an empty project has no stored facts, so every chunk
    is written — with the donor's vector, at no embedding cost."""
    repo = FakeChunkStore(checksums={"module.py": "c1"})
    donor = {
        _chunk_id(_chunk("first chunk", 1)): [1.0, 2.0],
        _chunk_id(_chunk("second chunk", 2)): [3.0, 4.0],
    }
    embedder = CountingEmbedder()

    written = await _indexer(repo, embedder, FakeChunker()).index_code(
        "p", tmp_path, [MODULE_NODE], stored_chunk_embeddings=donor
    )

    assert written == 2
    assert embedder.texts == []
    assert repo.rows[_chunk_id(_chunk("first chunk", 1))]["embedding"] == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Live · the Cypher behind the decision
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_stored_chunk_facts_reads_language_owner_and_vector_presence_live() -> None:
    """``GET_STORED_CHUNK_FACTS`` decides whether a paid write is skipped, so its three
    conditions are checked against a real Memgraph rather than against a fake's dictionary:
    a chunk with no vector is absent, the language comes back as stored, and the owner id
    comes from the HAS_CHUNK edge (a chunk whose edge is gone reports no owner).
    """
    project = f"chunk-facts-{uuid.uuid4().hex[:8]}"
    conn = MemgraphConnection(Settings(_env_file=None, memgraph_url=LIVE_URL, llm_api_key="x"))
    try:
        await setup_schema(conn)
        repo = GraphRepository(conn)
        await repo.upsert_nodes(project, [MODULE_NODE])
        await repo.store_code_chunks(project, [
            {
                "chunk_id": "chunk:with-vector", "node_id": MODULE_NODE.id,
                "file_path": "module.py", "content": "a", "start_line": 1, "end_line": 1,
                "language": "python", "chunk_kind": "function_body", "embedding": [0.1] * 1536,
            },
            {
                "chunk_id": "chunk:no-vector", "node_id": MODULE_NODE.id,
                "file_path": "module.py", "content": "b", "start_line": 2, "end_line": 2,
                "language": "python", "chunk_kind": "function_body", "embedding": None,
            },
            {
                "chunk_id": "chunk:orphan", "node_id": MODULE_NODE.id,
                "file_path": "module.py", "content": "c", "start_line": 3, "end_line": 3,
                "language": "markdown", "chunk_kind": "prose", "embedding": [0.2] * 1536,
            },
        ])
        await conn.execute_write(
            "MATCH (:Code {project: $p})-[r:HAS_CHUNK]->(c:CodeChunk {id: 'chunk:orphan'}) DELETE r",
            {"p": project},
        )

        facts = await repo.get_stored_chunk_facts(
            project, ["chunk:with-vector", "chunk:no-vector", "chunk:orphan", "chunk:absent"]
        )

        assert set(facts) == {"chunk:with-vector", "chunk:orphan"}, (
            "a chunk with no vector, or none at all, must not read as already stored"
        )
        assert facts["chunk:with-vector"] == {
            "language": "python", "node_ids": {MODULE_NODE.id}
        }
        assert facts["chunk:orphan"]["language"] == "markdown"
        assert facts["chunk:orphan"]["node_ids"] == set(), (
            "a chunk whose HAS_CHUNK edge is gone reported an owner anyway"
        )

        vectors = await repo.get_stored_chunk_embeddings_for_ids(
            project, ["chunk:with-vector", "chunk:no-vector"]
        )
        assert list(vectors) == ["chunk:with-vector"]
        assert len(vectors["chunk:with-vector"]) == 1536
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()


@live
@pytest.mark.asyncio
async def test_reindexing_an_unchanged_project_writes_no_chunk_row_live(tmp_path: Path) -> None:
    """End to end against a real database: the second index of unchanged content stores
    nothing, embeds nothing, leaves every chunk in place, and still stamps coverage — so the
    repair pass does not see the file as stale on the next run.
    """
    project = f"chunk-reuse-{uuid.uuid4().hex[:8]}"
    conn = MemgraphConnection(Settings(_env_file=None, memgraph_url=LIVE_URL, llm_api_key="x"))
    try:
        await setup_schema(conn)
        repo = GraphRepository(conn)
        await repo.upsert_nodes(project, [MODULE_NODE, CodeNode(
            id="module.py", name="module", node_type=NodeType.MODULE,
            file_path="module.py", line_start=1, line_end=2,
        )])
        await repo.update_checksums(project, [{"node_id": "module.py", "checksum": "c1"}])

        chunker = FakeChunker([_chunk("first chunk", 1), _chunk("second chunk", 2)])
        first_embedder = CountingEmbedder(vector=[0.3] * 1536)
        written = await _indexer(repo, first_embedder, chunker).index_code(
            project, tmp_path, [MODULE_NODE], prune_stale=True, reconcilable_files={"module.py"}
        )
        assert written == 2 and len(first_embedder.texts) == 2
        stored_ids = await repo.get_chunk_ids(project)
        assert len(stored_ids) == 2
        assert await repo.get_files_with_stale_chunks(project) == []

        second_embedder = CountingEmbedder(vector=[0.9] * 1536)
        rewritten = await _indexer(repo, second_embedder, chunker).index_code(
            project, tmp_path, [MODULE_NODE], prune_stale=True, reconcilable_files={"module.py"}
        )

        assert rewritten == 0, "unchanged chunks were written again"
        assert second_embedder.texts == [], "unchanged chunks were embedded again"
        assert await repo.get_chunk_ids(project) == stored_ids, "a live chunk was pruned"
        assert await repo.get_files_with_stale_chunks(project) == [], (
            "skipping the write left the file looking permanently stale"
        )
        # The vectors are the *first* run's, untouched — the second run never wrote one.
        rows = await conn.execute_query(
            "MATCH (c:CodeChunk {project: $p}) RETURN c.embedding AS e", {"p": project}
        )
        assert all(abs(float(r["e"][0]) - 0.3) < 1e-6 for r in rows)
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()
