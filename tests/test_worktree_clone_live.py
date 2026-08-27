"""S10-S12 · The worktree's first index as a **server-side clone** — live oracles.

``tests/test_worktree_e2e.py`` proves what a worktree's first index *costs* (nothing) and
what it *contains* (everything). This file proves how it is now produced: the donor's graph
is copied inside Memgraph, so no summary and no 1536-float vector makes a round trip out of
the database and back through the client. The claims:

  * S10 the copy is exact and usable — every node, chunk, dependency and edge count matches
    the donor, every summary is the donor's text under this project's id, the copied vectors
    are searchable (a hollow copy would also be "complete" by count), and a branch that
    edited one file still pays for that file alone, with an immediately repeated index
    doing nothing at all.
  * S11 a clone interrupted mid-copy is **detectable and repairable**: while it is in flight
    the project reports itself as *not indexed* (so no read tool can serve a half-copied
    graph) and is invisible to ``list_projects`` (so the janitor and the dashboard never see
    a phantom project rooted at the donor's directory); the next index deletes it and clones
    again, leaving no duplicates behind.
  * S12 the verification is not decorative: a copy that silently loses rows is refused, and
    the index degrades to the original client-side seeding path — same (zero) LLM cost, just
    slower — rather than storing a graph that merely looks whole.

Live by necessity: every one of these is a claim about real graph state, real vector indexes
and a real ``git worktree``. They are skipped unless ``INFLORESCENCE_TEST_MEMGRAPH_URL``
points at a **throwaway** Memgraph-MAGE instance — a developer's ``bolt://localhost:7687``
holds indexed projects and is never touched.

The world (real git, a recording LLM that signs each summary with the run that paid for it,
recording embedders) is the one built in ``tests/test_worktree_e2e.py``; it is imported
rather than copied so both files keep testing against the same fakes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from inflorescence.db import queries
from inflorescence.db.repository import GraphRepository, ProjectCloneError
from tests.test_worktree_e2e import (
    BETA_PY,
    _vec,
    _vector_search,
    _world,
    commit_all,
    git,
    live,
    make_repo,
)


async def _code_vector_search(
    repo: GraphRepository, project: str, vector: list[float]
) -> list[dict]:
    """``search_code_vectors`` with the same bounded wait for Memgraph's garbage collector
    that ``_vector_search`` applies to summary vectors: for one GC cycle after *any* project
    is deleted — a neighbouring test's teardown will do — every vector search in the database
    fails. That is a known, worktree-independent defect (see ``tests/test_worktree_e2e.py``),
    and without this wait the oracle below would be green or red by coincidence."""
    deadline = time.monotonic() + 45.0
    while True:
        try:
            return await repo.search_code_vectors(project, vector, top_k=25)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless it is *that* error
            if "deleted object" not in str(exc) or time.monotonic() > deadline:
                raise
            await asyncio.sleep(2.0)


async def _counts(repo: GraphRepository, project: str) -> dict[str, object]:
    """Node/chunk/dependency/edge counts, read the same independent way the clone verifies."""
    conn = repo._conn  # noqa: SLF001 — a test may read the graph without a public wrapper
    async def one(query: str) -> int:
        rows = await conn.execute_query(query, {"project": project})
        return int(rows[0]["count"]) if rows else 0

    edges = await conn.execute_query(queries.COUNT_EDGES_BY_TYPE, {"project": project})
    return {
        "nodes": await one(queries.COUNT_CODE_NODES),
        "chunks": await one(queries.COUNT_CHUNKS),
        "externals": await one(queries.COUNT_EXTERNALS),
        "edges": {str(r["type"]): int(r["count"]) for r in edges},
    }


# ---------------------------------------------------------------------------
# S10 · the copy is exact, searchable, and still pays only for what differs
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s10_cloned_worktree_is_an_exact_searchable_copy_and_pays_only_for_diffs(
    tmp_path: Path,
) -> None:
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo

        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))
        donor_counts = await _counts(repo, main_project)
        donor_summaries = await repo.get_stored_summaries(main_project)
        donor_chunks = await repo.get_chunk_ids(main_project)
        assert donor_counts["nodes"] > 0 and donor_counts["chunks"] > 0
        assert donor_counts["edges"], "sanity: the donor has no edges to copy"

        # ---- an identical checkout: the clone is the whole index.
        same = tmp_path / "wt-same"
        git("-C", str(main), "worktree", "add", "--detach", str(same))
        same_project = manager.get_project(str(same))

        world.start_run("wt-same")
        started = time.monotonic()
        stats = await manager.index_directory(str(same))
        elapsed = time.monotonic() - started

        world.assert_free("S10 (clone of an identical worktree)")
        assert stats["worktree_of"] == main_project
        assert stats["changed_files"] == 0, "an identical checkout had files to re-index"
        cloned = stats["cloned"]
        assert isinstance(cloned, dict)

        # Exact, counted independently of what the clone said it wrote.
        clone_counts = await _counts(repo, same_project)
        assert clone_counts == donor_counts, (
            f"the copy differs from the donor: {clone_counts} vs {donor_counts}"
        )
        assert cloned["nodes"] == donor_counts["nodes"]
        assert cloned["chunks"] == donor_counts["chunks"]
        assert cloned["edges"] == sum(donor_counts["edges"].values())  # type: ignore[union-attr]

        # Identical id-by-id, including the summary text and its freshness marker — a copy
        # that renamed or re-generated anything would still pass the counts above.
        wt_summaries = await repo.get_stored_summaries(same_project)
        assert set(wt_summaries) == set(donor_summaries)
        for node_id, row in wt_summaries.items():
            donor_row = donor_summaries[node_id]
            assert row["summary"] == donor_row["summary"], f"{node_id}: summary text changed"
            assert row["summary_input_hash"] == donor_row["summary_input_hash"]
            assert row["summary_embedding"] is not None, f"{node_id}: vector lost in the copy"
            assert list(row["summary_embedding"]) == list(donor_row["summary_embedding"])  # type: ignore[arg-type]
        assert await repo.get_chunk_ids(same_project) == donor_chunks
        assert await repo.get_checksums(same_project) == await repo.get_checksums(main_project)

        # The copied vectors are in the vector indexes, not merely on the nodes: a clone
        # whose embeddings were invisible to search would be a silently useless index.
        wt_ids = await repo.get_node_ids(same_project)
        hits = await _vector_search(repo, same_project, _vec("anything"))
        assert hits, "semantic search over the cloned project returned nothing"
        assert {h["id"] for h in hits} <= wt_ids, "vector search leaked another project's ids"
        code_hits = await _code_vector_search(repo, same_project, _vec("anything"))
        assert code_hits, "the cloned chunk vectors are not searchable"

        # The identity is this project's own, not the donor's — the clone arrives carrying
        # the donor's root_path and scope, and both must have been re-stamped.
        assert await repo.get_project_worktree_meta(same_project) == {
            "worktree_of": main_project,
            "main_root_path": str(main.resolve()),
        }
        roots = {p["project"]: p["root_path"] for p in await repo.list_projects()}
        assert roots[same_project] == str(same.resolve())
        assert roots[main_project] == str(main.resolve())

        # ---- a branch that edited one file and deleted another. The deletion is the half
        # a copy can get quietly wrong: the clone brings the donor's gamma.py with it, and
        # only the reconcile that follows can take it back out again.
        feature = tmp_path / "wt-feature"
        git("-C", str(main), "worktree", "add", "-b", "feature", str(feature))
        (feature / "beta.py").write_text(
            BETA_PY + '\n\ndef beta_branch_only(x):\n    """Only this branch."""\n    return x - 1\n',
            encoding="utf-8",
        )
        (feature / "delta.py").write_text(
            '"""Entry point."""\n\nimport beta\n\n\ndef delta_main():\n'
            '    """Run the tiny pipeline."""\n    return beta.beta_compute(2)\n',
            encoding="utf-8",
        )
        (feature / "gamma.py").unlink()
        commit_all(feature, "branch edits beta, drops gamma")
        feature_project = manager.get_project(str(feature))

        world.start_run("wt-feature")
        feature_stats = await manager.index_directory(str(feature))
        assert feature_stats["changed_files"] == 3, (
            f"expected beta.py, delta.py and the deleted gamma.py, got "
            f"{feature_stats['changed_files']} file(s)"
        )

        # The deleted file left no trace in this project, and is untouched in the donor's.
        feature_ids = await repo.get_node_ids(feature_project)
        assert not any(nid == "gamma.py" or nid.startswith("gamma.py::") for nid in feature_ids), (
            "the clone kept the donor's copy of a file the branch deleted"
        )
        assert "gamma.py" in await repo.get_node_ids(main_project)
        assert "gamma.py" not in set((await repo.get_chunk_index(feature_project)).values()), (
            "chunks of a deleted file survived the clone's reconcile"
        )
        paid_ids = await world.summarized_ids(feature_project, "wt-feature")
        assert paid_ids, "the edited file was not summarized at all"
        stray = {
            nid for nid in paid_ids
            if not (
                nid.split("::")[0] in {"beta.py", "delta.py"} or nid.startswith("dir:")
            )
        }
        assert not stray, f"the clone path re-summarized unrelated nodes: {sorted(stray)}"

        # Untouched files keep the donor's text; the branch's own symbol is searchable here
        # and nowhere else (INV-9).
        feature_summaries = await repo.get_stored_summaries(feature_project)
        for node_id, row in feature_summaries.items():
            if node_id in paid_ids:
                continue
            assert row["summary"] == donor_summaries[node_id]["summary"], (
                f"{node_id} was re-summarized although the branch never touched it"
            )
        assert await repo.search_symbol_text(feature_project, "beta_branch_only", top_k=25)
        assert await repo.search_symbol_text(main_project, "beta_branch_only", top_k=25) == []

        # Only the edited file's chunks are new, and only they were embedded.
        feature_chunks = await repo.get_chunk_index(feature_project)
        new_chunks = {cid: fp for cid, fp in feature_chunks.items() if cid not in donor_chunks}
        assert new_chunks, "the edited file produced no new chunks"
        assert set(new_chunks.values()) == {"beta.py", "delta.py"}, (
            f"chunks were re-embedded for untouched files: {sorted(set(new_chunks.values()))}"
        )
        assert len(world.code_embedder.texts) == len(new_chunks)

        # ---- and an immediately repeated index does nothing.
        before = await _counts(repo, feature_project)
        world.start_run("wt-noop")
        repeat = await manager.index_directory(str(feature))
        assert repeat["reused"] is True
        assert world.llm.summary_calls == 0
        assert not world.summary_embedder.texts
        assert await _counts(repo, feature_project) == before

        print(
            f"\nS10: donor nodes={donor_counts['nodes']} chunks={donor_counts['chunks']} "
            f"edges={sum(donor_counts['edges'].values())}; identical worktree cloned in "  # type: ignore[union-attr]
            f"{elapsed:.2f}s with llm=0 embeds=0; one-file branch paid for "
            f"{len(paid_ids)} summar(ies) and {len(new_chunks)} chunk(s)"
        )
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S11 · an interrupted clone is detectable and repairable
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s11_an_interrupted_clone_is_invisible_and_the_next_index_repairs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process dies between the node copy and the edge copy.

    Simulated at the seam where a kill hurts most: nodes and chunks are already in the
    database under the new project id, the edges are not, and no identity has been stamped.
    Nothing cleans up — that is the point; a killed process runs no ``except`` block. What
    must hold is that the leftovers cannot be mistaken for an index (a read tool must not
    serve them, the janitor must not see a project rooted at the donor's directory) and that
    the next index converges to a correct graph with nothing duplicated.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))
        donor_counts = await _counts(repo, main_project)

        worktree = tmp_path / "wt-killed"
        git("-C", str(main), "worktree", "add", "--detach", str(worktree))
        wt_project = manager.get_project(str(worktree))
        world.extra_projects.append(wt_project)

        async def _die(self: GraphRepository, source: str, target: str) -> int:
            raise RuntimeError("process killed mid-clone")

        monkeypatch.setattr(GraphRepository, "_clone_edges", _die)
        # Called directly, not through the manager: the manager would clean up after itself,
        # and a killed process does not.
        with pytest.raises(ProjectCloneError):
            await repo.clone_project(main_project, wt_project)
        monkeypatch.undo()

        # The leftovers exist...
        partial = await _counts(repo, wt_project)
        assert partial["nodes"] == donor_counts["nodes"], "sanity: the node copy did land"
        assert not partial["edges"], "sanity: the edge copy must not have run"
        # ...and are, nonetheless, not an index:
        assert await repo.get_clone_marker(wt_project) == main_project
        assert not await repo.project_is_indexed(wt_project), (
            "a half-copied graph reports itself as indexed — a read tool would serve it"
        )
        listed = {p["project"] for p in await repo.list_projects()}
        assert wt_project not in listed, (
            "an unfinished clone appeared in the project list; the janitor and the dashboard "
            "would see a phantom project rooted at the donor's directory"
        )
        # The donor is untouched by any of this.
        assert await _counts(repo, main_project) == donor_counts

        # The next index repairs it: delete, clone again, and no duplicates.
        world.start_run("wt-repair")
        stats = await manager.index_directory(str(worktree))
        world.assert_free("S11 (re-clone after an interrupted one)")
        assert stats["worktree_of"] == main_project
        assert "cloned" in stats
        assert await repo.get_clone_marker(wt_project) is None
        assert await repo.project_is_indexed(wt_project)
        repaired = await _counts(repo, wt_project)
        assert repaired == donor_counts, (
            f"the repaired clone is not an exact copy: {repaired} vs {donor_counts}"
        )
        ids = await repo.get_node_ids(wt_project)
        assert len(ids) == repaired["nodes"], "the repair left duplicate node ids behind"
        assert (await repo.list_projects()) and wt_project in {
            p["project"] for p in await repo.list_projects()
        }
        print(f"\nS11: interrupted clone repaired; counts back to {repaired['nodes']} nodes")
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S12 · a copy that cannot be trusted is refused, and the slow path takes over
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s12_a_lossy_clone_is_refused_and_the_index_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memgraph 3.12 can write fewer rows than its own ``count(*)`` reports (see the note at
    ``queries.MARK_CHUNKS_COVERED``), so the clone is verified by an independent recount and
    not by its own numbers. Here the loss is induced — the edge copy writes nothing while
    claiming success — and the index must end up complete anyway, via the client-side path,
    with no trace of the refused copy.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))
        donor_counts = await _counts(repo, main_project)
        donor_summaries = await repo.get_stored_summaries(main_project)

        worktree = tmp_path / "wt-lossy"
        git("-C", str(main), "worktree", "add", "--detach", str(worktree))
        wt_project = manager.get_project(str(worktree))

        async def _write_nothing(self: GraphRepository, source: str, target: str) -> int:
            return 0  # "wrote everything", quietly writing nothing

        monkeypatch.setattr(GraphRepository, "_clone_edges", _write_nothing)
        world.start_run("wt-lossy")
        stats = await manager.index_directory(str(worktree))
        monkeypatch.undo()

        # The copy was refused and deleted; the proven path produced the index instead.
        assert "cloned" not in stats, "a clone that lost every edge was accepted"
        assert stats["worktree_of"] == main_project
        assert int(stats["seeded_summaries"]) > 0, "the fallback did not seed from the donor"
        world.assert_free("S12 (fallback after a refused clone)")
        assert await repo.get_clone_marker(wt_project) is None

        final = await _counts(repo, wt_project)
        assert final == donor_counts, f"the fallback index is incomplete: {final} vs {donor_counts}"
        wt_summaries = await repo.get_stored_summaries(wt_project)
        assert set(wt_summaries) == set(donor_summaries)
        for node_id, row in wt_summaries.items():
            assert row["summary"] == donor_summaries[node_id]["summary"]
            assert row["summary_embedding"] is not None
        print(f"\nS12: lossy clone refused; fallback produced {final['nodes']} nodes, llm=0")
    finally:
        await world.close()
