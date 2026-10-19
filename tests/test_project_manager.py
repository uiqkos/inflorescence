from __future__ import annotations

from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import BuildResult
from inflorescence.config import Settings
from inflorescence.project_manager import ProjectManager


@pytest.mark.asyncio
async def test_index_directory_checks_existing_project_and_reuses_when_unchanged(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert result == {
        "project": f"{tmp_path.name}_{repo.expected_hash}",
        "directory": str(tmp_path.resolve()),
        "nodes": 12,
        "edges": 34,
        "chunks": 5,
        "summaries": 6,
        "reused": True,
    }
    assert builder.builds == 0
    assert builder.updates == 1
    assert rag.code_indexes == 0
    assert rag.summary_indexes == 0


@pytest.mark.asyncio
async def test_unchanged_path_backfills_missing_summary_embeddings(tmp_path: Path) -> None:
    pending = [object(), object()]
    repo = FakeProjectRepository(
        {"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6},
        unembedded_summary_nodes=pending,
    )
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer(summary_count=2)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert result["reused"] is True
    assert result["summary_embeddings_repaired"] == 2
    assert rag.summary_indexes == 1
    assert rag.summary_nodes == [pending]        # exactly the incomplete nodes, nothing else
    assert rag.code_indexes == 0


@pytest.mark.asyncio
async def test_unchanged_path_rechunks_files_with_no_stored_chunks(tmp_path: Path) -> None:
    file_nodes = [object(), object(), object()]
    repo = FakeProjectRepository(
        {"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6},
        files_missing_chunks=["pkg/a.py"],
        nodes_for_files=file_nodes,
    )
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer(code_count=4)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert result["reused"] is True
    assert result["chunks_repaired"] == 4
    assert repo.files_requested == [["pkg/a.py"]]
    assert rag.code_indexes == 1
    assert rag.code_nodes == [file_nodes]
    assert rag.summary_indexes == 0
    # The repair rebuilds a named set of files, so it reads back the vectors of those files
    # and no others — the whole-project dump it used to be handed was paid for on every
    # unchanged re-index, whether or not anything needed repairing at all.
    assert repo.scoped_chunk_embedding_requests == [
        (manager.get_project(str(tmp_path)), ["pkg/a.py"])
    ]
    assert repo.chunk_embedding_requests == []


@pytest.mark.asyncio
async def test_unchanged_path_regenerates_missing_summaries(tmp_path: Path) -> None:
    repaired_nodes = [object(), object(), object()]
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(repair_result=repaired_nodes)
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert result["reused"] is True
    assert result["summaries_repaired"] == 3
    assert builder.repairs == 1


@pytest.mark.asyncio
async def test_unchanged_path_with_complete_state_makes_no_repair_calls(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert rag.code_indexes == 0 and rag.summary_indexes == 0
    assert "summaries_repaired" not in result
    assert "chunks_repaired" not in result
    assert "summary_embeddings_repaired" not in result


@pytest.mark.asyncio
async def test_index_directory_reindexes_existing_project_when_update_finds_changes(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(update_result=([object(), object()], [object()]))
    rag = FakeRAGIndexer(code_count=7, summary_count=8)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path))

    assert result == {
        "project": f"{tmp_path.name}_{repo.expected_hash}",
        "directory": str(tmp_path.resolve()),
        "nodes": 2,
        "edges": 1,
        "chunks": 7,
        "summaries": 8,
        "db": {"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6},
        "skipped_large_files": [],
        "reused": False,
    }
    assert builder.builds == 0
    assert builder.updates == 1
    assert rag.code_indexes == 1
    assert rag.summary_indexes == 1


@pytest.mark.asyncio
async def test_index_directory_force_rebuild_bypasses_checksum_shortcircuit(tmp_path: Path) -> None:
    """An indexer upgrade changes edge semantics without changing any file content; the
    incremental path would no-op on checksums. force_rebuild must take the full build
    path (re-parse + re-resolve + reconcile) while reusing stored summaries."""
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(build_result=([object(), object()], [object()]))
    rag = FakeRAGIndexer(code_count=7, summary_count=8)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path), force_rebuild=True)

    assert builder.builds == 1
    assert builder.updates == 0
    assert result["reused"] is False
    assert result["nodes"] == 2 and result["edges"] == 1
    # Reuse is by summary hash, and the vectors are already on this project's own nodes:
    # the rebuild must read the summaries WITHOUT dragging every vector over Bolt. Only
    # cross-project seeding has a use for the payload version.
    project = manager.get_project(str(tmp_path))
    assert repo.light_summary_requests == [project]
    assert repo.summary_requests == []


@pytest.mark.asyncio
async def test_index_directory_passes_include_and_exclude_filters_to_build(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=([object()], []))
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    await manager.index_directory(
        str(tmp_path),
        include_patterns=["backend/**"],
        exclude_patterns=[".worktrees"],
    )

    assert builder.build_filters == [(["backend/**"], [".worktrees"])]


@pytest.mark.asyncio
async def test_preview_directory_wraps_builder_preview_without_db_writes(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.preview_directory(str(tmp_path))

    assert result["preview"] is True
    assert result["files"] == 3
    assert result["estimated_tokens"] == 30
    assert "estimated_cost_usd" in result          # dry-run now surfaces the dollar estimate
    assert result["directory"] == str(tmp_path.resolve())
    assert builder.builds == 0 and builder.updates == 0   # no indexing
    assert rag.code_indexes == 0 and rag.summary_indexes == 0


def test_estimate_index_cost_scales_with_tokens_and_respects_use_llm_summaries() -> None:
    from inflorescence.cost import estimate_index_cost

    preview = {"estimated_tokens": 1_000_000, "estimated_nodes": 0}
    # 1M LLM-input tokens @ $0.25/M + 1M embedding tokens @ $0.02/M = $0.27
    assert estimate_index_cost(preview, Settings(_env_file=None))["usd"] == pytest.approx(0.27, abs=0.01)
    # No summaries -> no LLM spend, only chunk embeddings @ $0.02/M = $0.02
    off = estimate_index_cost(preview, Settings(_env_file=None, use_llm_summaries=False))
    assert off["usd"] == pytest.approx(0.02, abs=0.005)


@pytest.mark.asyncio
async def test_index_directory_aborts_first_index_over_cost_cap(tmp_path: Path) -> None:
    from inflorescence.cost import IndexCostExceededError

    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    with pytest.raises(IndexCostExceededError):
        await manager.index_directory(str(tmp_path), max_cost_usd=0.0001)

    assert builder.builds == 0            # aborted before any LLM/DB spend
    assert rag.summary_indexes == 0


@pytest.mark.asyncio
async def test_index_directory_proceeds_when_estimate_under_cost_cap(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    result = await manager.index_directory(str(tmp_path), max_cost_usd=10.0)

    assert result["reused"] is False
    assert builder.builds == 1            # estimate under cap -> proceeded


@pytest.mark.asyncio
async def test_index_directory_aborts_on_preflight_failure_before_any_mutation(tmp_path: Path) -> None:
    from inflorescence.project_manager import IndexPreflightError

    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    rag = FakeRAGIndexer()
    rag.preflight_error = RuntimeError("401 no key")
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    with pytest.raises(IndexPreflightError):
        await manager.index_directory(str(tmp_path))

    # Nothing was touched: no build/update, no RAG calls, no root-path write.
    assert builder.builds == 0 and builder.updates == 0
    assert rag.code_indexes == 0 and rag.summary_indexes == 0
    assert repo.root_paths == []


def test_default_settings_gate_first_index_by_default() -> None:
    # INV-10: the cost cap is ON by default. A None default (the bug) left the first index of any
    # size ungated — a monorepo could be indexed with no cost estimate at all.
    assert Settings(_env_file=None).max_index_cost_usd == 10.0


@pytest.mark.asyncio
async def test_first_index_is_gated_by_the_default_cap_without_an_explicit_limit(tmp_path: Path) -> None:
    from inflorescence.cost import IndexCostExceededError

    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    # A monorepo-scale estimate — well over the $10 default — with NO max_cost_usd argument.
    builder.preview_result = {"estimated_tokens": 5_000_000_000, "estimated_nodes": 5_000_000}
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    with pytest.raises(IndexCostExceededError):
        await manager.index_directory(str(tmp_path))   # no explicit cap -> default $10 gates it

    assert builder.builds == 0 and rag.summary_indexes == 0    # aborted before any spend


@pytest.mark.asyncio
async def test_index_directory_persists_scan_scope_for_the_watcher(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=([object()], []))
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))
    project = manager.get_project(str(tmp_path))

    await manager.index_directory(
        str(tmp_path), include_patterns=["backend/**"], exclude_patterns=["vendor"]
    )

    # The explicit index's scope is recorded so a later watcher update reproduces it (5c).
    assert repo.scopes == [(project, ["backend/**"], ["vendor"])]


@pytest.mark.asyncio
async def test_update_project_inherits_persisted_scope(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 1, "edges": 0, "chunks": 0, "summaries": 0})
    repo.stored_scope = (["backend/**"], ["vendor"])
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))
    project = manager.get_project(str(tmp_path))

    await manager.update_project(project, ["backend/app.py"])

    # A watcher update scans the SAME universe as the index that created the project, so it
    # cannot widen scope and silently pay for excluded paths (5c). Was: no scope passed at all.
    assert builder.update_filters == [(["backend/**"], ["vendor"])]


@pytest.mark.asyncio
async def test_update_project_reads_chunk_vectors_only_for_the_files_that_moved(
    tmp_path: Path,
) -> None:
    """A file save must not unload every vector in the project.

    The read used to happen *before* the update — the only place it could, when the caller
    had no idea which files had really changed — and it dumped the whole project: seconds of
    Bolt traffic per keystroke-triggered flush, to serve the handful of chunks the edit
    touched (INV-5). It now runs after the update, scoped to what the update found different.
    """
    repo = FakeProjectRepository({"nodes": 1, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    builder.update_changed_files = {"backend/app.py"}
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))
    project = manager.get_project(str(tmp_path))

    await manager.update_project(project, ["backend/app.py"])

    assert repo.chunk_embedding_requests == [], "the whole project's vectors were unloaded"
    assert repo.scoped_chunk_embedding_requests == [(project, ["backend/app.py"])]
    assert rag.code_stored_embeddings == [repo.stored_chunk_embeddings.get(project, {})]


@pytest.mark.asyncio
async def test_reindex_of_an_indexed_project_is_gated_by_the_cost_cap(tmp_path: Path) -> None:
    """INV-10 covers a re-index, not only a project's first index.

    The gate used to live on the first-time branch alone, so an already-indexed project could
    be re-indexed for any amount. That is not a theoretical hole: widening what the indexer
    admits (one new DOCUMENT_EXTENSIONS entry) makes thousands of files newly billable behind
    a call that reads like a cheap refresh — measured at ~$57 on a real repo, unasked.
    """
    from inflorescence.cost import IndexCostExceededError

    repo = FakeProjectRepository({"nodes": 500, "edges": 900, "chunks": 40, "summaries": 400})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    builder.pending_change_paths_result = [f"fixtures/case_{i}.json" for i in range(4000)]
    builder.estimate_result = {"estimated_tokens": 5_000_000_000, "estimated_nodes": 5_000_000}
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    with pytest.raises(IndexCostExceededError) as caught:
        await manager.index_directory(str(tmp_path))

    assert builder.updates == 0                      # gated before parse/summarize/embed
    assert rag.summary_indexes == 0
    assert caught.value.changed_files == 4000        # priced over the changed set, not the tree
    assert "4000 changed file(s)" in str(caught.value)
    # The full-tree preview is the first-time branch's question and must not be asked here:
    # pricing an incremental run over the whole repo would reject work it is about to reuse.
    assert builder.previews == 0


@pytest.mark.asyncio
async def test_reindex_with_a_small_changed_set_is_not_gated(tmp_path: Path) -> None:
    """The gate must not turn a routine refresh of a large project into a refusal."""
    repo = FakeProjectRepository({"nodes": 50_000, "edges": 90_000, "chunks": 4_000, "summaries": 40_000})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    builder.pending_change_paths_result = ["src/app.py"]
    builder.estimate_result = {"estimated_tokens": 900, "estimated_nodes": 30}
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    await manager.index_directory(str(tmp_path))

    assert builder.updates == 1
    assert builder.estimated_paths == [["src/app.py"]]


@pytest.mark.asyncio
async def test_reindex_passes_the_run_scope_to_the_cost_gate(tmp_path: Path) -> None:
    """Excluded paths are not billable, so the gate has to price the same scope the run uses."""
    repo = FakeProjectRepository({"nodes": 500, "edges": 900, "chunks": 40, "summaries": 400})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    await manager.index_directory(str(tmp_path), exclude_patterns=["tests/golden"])

    assert builder.pending_change_path_calls == [(None, ["tests/golden"])]


@pytest.mark.asyncio
async def test_update_project_is_gated_by_the_cost_cap(tmp_path: Path) -> None:
    from inflorescence.cost import IndexCostExceededError

    repo = FakeProjectRepository({"nodes": 1, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    builder.estimate_result = {"estimated_tokens": 5_000_000_000, "estimated_nodes": 5_000_000}
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))
    project = manager.get_project(str(tmp_path))

    # A `git checkout` touching thousands of files must not auto-spend beyond the cap (5b).
    with pytest.raises(IndexCostExceededError):
        await manager.update_project(project, ["a.py", "b.py"])

    assert builder.updates == 0            # gated before build/summarize/embed


@pytest.mark.asyncio
async def test_preflight_aborts_when_the_llm_chat_model_is_dead(tmp_path: Path) -> None:
    from inflorescence.project_manager import IndexPreflightError

    # INV-2/finding 6: the embedder can be fine while the LLM chat model is dead (bad LLM_MODEL).
    # The run must still abort before any mutation, not pay to build then write surrogates.
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    builder.preflight_error = RuntimeError("404 model not found")
    rag = FakeRAGIndexer()                 # embedder preflight passes
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))

    with pytest.raises(IndexPreflightError):
        await manager.index_directory(str(tmp_path))

    assert builder.builds == 0 and builder.updates == 0
    assert rag.code_indexes == 0 and rag.summary_indexes == 0
    assert repo.root_paths == []


@pytest.mark.asyncio
async def test_preflight_refuses_to_index_when_vector_search_indexes_are_missing(tmp_path: Path) -> None:
    from inflorescence.project_manager import SearchUnavailableError

    # DB-5/INV-2: a Memgraph without the vector indexes cannot search embeddings, so paying to
    # embed is waste — the server must refuse rather than pretend search works.
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    rag = FakeRAGIndexer()
    conn = FakeConnMissingSearchIndexes()
    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None), conn=conn
    )

    with pytest.raises(SearchUnavailableError):
        await manager.index_directory(str(tmp_path))

    assert builder.builds == 0 and builder.updates == 0
    assert repo.root_paths == []


@pytest.mark.asyncio
async def test_preflight_allows_indexing_without_search_when_opted_out(tmp_path: Path) -> None:
    # The gate is overridable for graph-only use on a search-less DB.
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    conn = FakeConnMissingSearchIndexes()
    settings = Settings(_env_file=None, require_search_indexes=False)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn)

    result = await manager.index_directory(str(tmp_path))
    assert result["reused"] is False and builder.builds == 1


@pytest.mark.asyncio
async def test_update_project_serializes_concurrent_updates(tmp_path: Path) -> None:
    import asyncio

    repo = FakeProjectRepository({"nodes": 1, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=([object()], [object()]))
    rag = FakeRAGIndexer()
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=Settings(_env_file=None))
    project = manager.get_project(str(tmp_path))

    in_flight = 0
    max_in_flight = 0
    original_update = builder.update

    async def observing_update(*args: object, **kwargs: object) -> tuple[list[object], list[object]]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.01)  # give a racing update the chance to overlap
        in_flight -= 1
        return await original_update(*args, **kwargs)

    builder.update = observing_update  # type: ignore[method-assign]

    await asyncio.gather(
        manager.update_project(project, ["a.py"]),
        manager.update_project(project, ["b.py"]),
    )

    # Two watcher flushes racing used to interleave delete/rebuild phases and destroy
    # each other's writes; the per-project lock forces them to run one at a time.
    assert max_in_flight == 1
    assert builder.updates == 2


# ---------------------------------------------------------------------------
# Worktree donor seeding: a linked worktree is its own project, but its content is the
# main checkout's, so its first index must reuse that project's summaries and vectors —
# zero LLM calls, zero embeddings — and must not be blocked by a cost cap for spend that
# will never happen. These tests pin the plumbing: which cache went where, what was
# written back, and what the gate priced.
#
# The reuse itself is normally done by copying the donor's graph inside Memgraph (see the
# clone tests further down). The tests here pin the **client-side** path it falls back to,
# so they switch the clone off explicitly — it is the path a clone failure lands on, and it
# has to keep working exactly as it did.
# ---------------------------------------------------------------------------


def _client_seeding() -> Settings:
    """Settings that force the pre-clone, client-side donor seeding path."""
    return Settings(_env_file=None, worktree_clone=False)


class _SummaryNode:
    """Minimal stand-in for a CodeNode as the seeded-summary write-back sees it."""

    def __init__(
        self,
        node_id: str,
        summary: str = "",
        summary_input_hash: str = "",
        file_path: str | None = None,
    ) -> None:
        self.id = node_id
        self.summary = summary
        self.summary_input_hash = summary_input_hash
        # The clone path scopes its paid phases by file, so these stand-ins carry one.
        # Defaults to the id, which is what a module node's id already is.
        self.file_path = node_id if file_path is None else file_path


def _worktree_dirs(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    main.mkdir()
    worktree = tmp_path / "wt-feature"
    worktree.mkdir()
    return main, worktree


def _worktree_manager(
    repo: FakeProjectRepository,
    builder: FakeGraphBuilder,
    rag: FakeRAGIndexer,
    main_root: Path | None,
    settings: Settings | None = None,
) -> ProjectManager:
    """A manager whose worktree detection is stubbed — no real git layout needed here."""
    return ProjectManager(
        repo=repo,
        graph_builder=builder,
        rag_indexer=rag,
        settings=settings or Settings(_env_file=None),
        worktree_resolver=lambda path: main_root,
    )


@pytest.mark.asyncio
async def test_first_index_of_a_worktree_seeds_caches_from_the_indexed_main_checkout(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(build_result=([_SummaryNode("a.py", "s", "h1")], []))
    rag = FakeRAGIndexer(code_count=1, summary_count=1)
    manager = _worktree_manager(repo, builder, rag, main, settings=_client_seeding())

    donor = manager.get_project(str(main))
    donor_summaries = {
        "a.py": {
            "summary": "s", "summary_input_hash": "h1",
            "summary_embedding": [0.1, 0.2], "summary_embedding_hash": "h1",
        }
    }
    donor_chunks = {"a.py#0": [0.3, 0.4]}
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = donor_summaries
    repo.stored_chunk_embeddings[donor] = donor_chunks

    result = await manager.index_directory(str(worktree))

    project = manager.get_project(str(worktree))
    # The donor's caches — not the worktree's own (empty) ones — feed every paid phase.
    assert repo.summary_requests == [donor]
    assert repo.chunk_embedding_requests == [donor]
    assert builder.build_stored_summaries == [donor_summaries]
    assert rag.code_stored_embeddings == [donor_chunks]
    assert rag.summary_stored_summaries == [donor_summaries]
    assert rag.summary_dirty_ids == [set()]          # nothing regenerated -> nothing re-embedded
    assert result["worktree_of"] == donor
    assert repo.worktree_meta == [(project, donor, str(main.resolve()))]


@pytest.mark.asyncio
async def test_worktree_seeding_writes_reused_summaries_into_the_new_project(tmp_path: Path) -> None:
    """The build callback only persists *generated* batches; summaries reused from the donor
    live on in-memory nodes only. Without an explicit write-back the new project keeps the
    skeleton's empty summaries and the repair pass regenerates them for real money."""
    main, worktree = _worktree_dirs(tmp_path)
    nodes = [
        _SummaryNode("a.py", "reused from donor", "h1"),   # reused -> must be written back
        _SummaryNode("b.py", "freshly generated", "h2"),   # dirty -> already persisted by build
        _SummaryNode("c.py", "", ""),                      # failed with nothing to keep
    ]
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(build_result=(nodes, []))
    builder.build_dirty_ids = {"b.py"}
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main, settings=_client_seeding())

    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = {"a.py": {"summary": "reused from donor", "summary_input_hash": "h1"}}

    await manager.index_directory(str(worktree))

    project = manager.get_project(str(worktree))
    assert repo.summaries_written == [
        (project, [{"node_id": "a.py", "summary": "reused from donor", "summary_input_hash": "h1"}])
    ]
    # The invariant the repair pass depends on: every node that ends the run with a summary
    # is either dirty (persisted batch-by-batch by the build) or explicitly written back —
    # so get_nodes_missing_summaries has nothing left to regenerate.
    written_ids = {row["node_id"] for _, rows in repo.summaries_written for row in rows}
    assert all(n.id in written_ids or n.id in builder.build_dirty_ids for n in nodes if n.summary)
    assert "c.py" not in written_ids       # empty summaries are never written (INV-1)


@pytest.mark.asyncio
async def test_worktree_cost_gate_prices_nothing_when_the_donor_covers_every_file(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    builder.scan_checksums_result = {"a.py": "m1", "pkg/b.py": "m2"}
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main, settings=_client_seeding())

    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.checksums[donor] = {"a.py": "m1", "pkg/b.py": "m2", "gone.py": "m3"}

    # A cap this tight rejects even the tiny default preview — but a fully donor-covered
    # worktree costs nothing, so it must index anyway (INV-10 gates *spend*, not work).
    result = await manager.index_directory(str(worktree), max_cost_usd=0.0001)

    assert builder.builds == 1
    assert builder.previews == 0                # the full-scan preview is not what gates this
    assert builder.estimated_paths == [[]]      # no file is paid for
    assert repo.checksum_requests == [donor]
    assert result["reused"] is False


@pytest.mark.asyncio
async def test_worktree_cost_gate_prices_exactly_the_files_the_donor_lacks(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    builder.scan_checksums_result = {"same.py": "m1", "edited.py": "NEW", "added.py": "m9"}
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main, settings=_client_seeding())

    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.checksums[donor] = {"same.py": "m1", "edited.py": "OLD"}

    await manager.index_directory(str(worktree))

    assert builder.estimated_paths == [["added.py", "edited.py"]]
    assert builder.scan_checksum_calls == [(None, None)]


@pytest.mark.asyncio
async def test_first_index_without_a_donor_is_gated_by_the_full_preview(tmp_path: Path) -> None:
    """Control for the two gate tests above: with no donor, the tight cap still bites."""
    from inflorescence.cost import IndexCostExceededError

    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, None)

    with pytest.raises(IndexCostExceededError):
        await manager.index_directory(str(tmp_path), max_cost_usd=0.0001)


@pytest.mark.asyncio
async def test_worktree_with_an_unindexed_main_checkout_indexes_normally_but_keeps_the_marker(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(build_result=([_SummaryNode("a.py", "s", "h1")], []))
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))

    result = await manager.index_directory(str(worktree))

    assert repo.project_is_indexed_calls == [donor]   # asked, and the answer was "no"
    assert repo.summary_requests == []                # nothing to seed from
    assert repo.summaries_written == []               # so nothing to write back either
    assert builder.build_stored_summaries == [None]
    assert builder.previews == 1                      # normal first-index cost gate
    assert "worktree_of" not in result
    # The marker is a fact about the directory, independent of whether a donor existed:
    # a future janitor must be able to tell this index apart from a standalone project.
    assert repo.worktree_meta == [(project, donor, str(main.resolve()))]


@pytest.mark.asyncio
async def test_plain_directory_never_reaches_the_donor_path(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, None)

    await manager.index_directory(str(tmp_path))

    assert repo.project_is_indexed_calls == []
    assert repo.checksum_requests == []
    assert repo.summary_requests == []
    assert repo.worktree_meta == []
    assert repo.summaries_written == []
    assert builder.scan_checksum_calls == []
    assert builder.previews == 1
    assert rag.code_stored_embeddings == [{}]


@pytest.mark.asyncio
async def test_reindexing_an_already_indexed_worktree_uses_its_own_caches(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder(update_result=([_SummaryNode("a.py", "s", "h1")], [object()]))
    builder.update_changed_files = {"a.py"}
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = {"a.py": {"summary": "donor", "summary_input_hash": "h1"}}

    await manager.index_directory(str(worktree))

    # Seeding is a first-index-only affair: the worktree now has its own stored state, and
    # re-seeding from the donor would overwrite it with the other checkout's content.
    assert builder.updates == 1 and builder.builds == 0
    assert repo.project_is_indexed_calls == []
    assert repo.summary_requests == []
    # Its own caches, and only for the file that moved: an update rewrites almost no chunk
    # rows, so unloading every vector in the project to feed one file was pure cost.
    assert repo.scoped_chunk_embedding_requests == [(project, ["a.py"])]
    assert repo.chunk_embedding_requests == []
    assert repo.summaries_written == []
    assert repo.worktree_meta == [(project, donor, str(main.resolve()))]


@pytest.mark.asyncio
async def test_worktree_auto_index_disabled_skips_seeding_but_still_marks_the_project(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    settings = Settings(_env_file=None, worktree_auto_index=False)
    manager = _worktree_manager(repo, builder, rag, main, settings=settings)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = {"a.py": {"summary": "donor", "summary_input_hash": "h1"}}

    await manager.index_directory(str(worktree))

    assert repo.project_is_indexed_calls == []       # the flag switches the branch off entirely
    assert builder.build_stored_summaries == [None]
    assert builder.previews == 1
    # Detection is cheap and deletion safety outranks the flag, so the marker is still set.
    assert repo.worktree_meta == [(project, donor, str(main.resolve()))]


@pytest.mark.asyncio
async def test_donor_summary_with_a_failed_embedding_is_passed_through_untouched(tmp_path: Path) -> None:
    """A donor node whose embedding never landed (a failed embed run) reaches the reuse
    logic as ``summary_embedding=None``; the RAG layer then simply re-embeds that one node.
    The manager must not filter, patch, or choke on such an entry."""
    main, worktree = _worktree_dirs(tmp_path)
    donor_summaries = {
        "ok.py": {"summary": "s", "summary_input_hash": "h1",
                  "summary_embedding": [0.1], "summary_embedding_hash": "h1"},
        "broken.py": {"summary": "s2", "summary_input_hash": "h2",
                      "summary_embedding": None, "summary_embedding_hash": None},
    }
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(
        build_result=([_SummaryNode("ok.py", "s", "h1"), _SummaryNode("broken.py", "s2", "h2")], [])
    )
    rag = FakeRAGIndexer(summary_count=2)
    manager = _worktree_manager(repo, builder, rag, main, settings=_client_seeding())
    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = donor_summaries

    result = await manager.index_directory(str(worktree))

    assert rag.summary_stored_summaries == [donor_summaries]
    assert result["summaries"] == 2
    project = manager.get_project(str(worktree))
    # Both are reused text, so both are written back — the missing vector is an embedding
    # concern, not a summary one.
    assert {row["node_id"] for _, rows in repo.summaries_written for row in rows} == {"ok.py", "broken.py"}
    assert repo.summaries_written[0][0] == project


# ---------------------------------------------------------------------------
# Server-side clone: the same free first index, done inside the database. These pin the
# decision (when it runs at all), the order (copy -> identity -> incremental update), the
# scoping of the paid phases, and the two ways it must degrade — a clone that fails, and a
# clone that never finished because the process died mid-copy.
# ---------------------------------------------------------------------------


def _cloning_world(
    tmp_path: Path,
    *,
    nodes: list[_SummaryNode] | None = None,
    worktree_files: dict[str, str] | None = None,
    donor_files: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> tuple[ProjectManager, FakeProjectRepository, FakeGraphBuilder, FakeRAGIndexer, Path, Path]:
    """A first index of a worktree whose donor is indexed, with the clone path available."""
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder(update_result=(list(nodes or []), []))
    builder.call_log = repo.call_log
    builder.scan_checksums_result = dict(worktree_files or {"same.py": "m1"})
    rag = FakeRAGIndexer(code_count=2, summary_count=3)
    manager = _worktree_manager(repo, builder, rag, main, settings=settings)

    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    # The donor's stored checksums are the whole diff mechanism: whatever this tree hashes
    # to that the donor does not already hold is what the branch changed.
    repo.checksums[donor] = dict(donor_files or {"same.py": "m1"})
    return manager, repo, builder, rag, main, worktree


@pytest.mark.asyncio
async def test_first_index_of_a_worktree_clones_the_donor_graph_inside_the_database(
    tmp_path: Path,
) -> None:
    """The default first index of a worktree is a server-side copy, not a client-side rebuild."""
    manager, repo, builder, rag, main, worktree = _cloning_world(tmp_path)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))

    result = await manager.index_directory(str(worktree))

    assert repo.clone_calls == [(donor, project)]
    assert builder.builds == 0, "the tree was re-parsed from scratch despite the clone"
    # Nothing was pulled through the client: no whole-project summary dump, no whole-project
    # vector dump, and no summary write-back — the values never left the database.
    assert repo.summary_requests == []
    assert repo.chunk_embedding_requests == []
    assert repo.summaries_written == []
    assert result["worktree_of"] == donor
    assert result["cloned"] == repo.clone_counts
    assert "seeded_summaries" not in result, (
        "the clone path must not claim summaries were seeded through the client"
    )
    assert repo.worktree_meta == [(project, donor, str(main.resolve()))]
    assert repo.root_paths == [(project, str(worktree.resolve()))]


@pytest.mark.asyncio
async def test_cloned_worktree_copies_first_then_stamps_identity_then_updates(
    tmp_path: Path,
) -> None:
    """Order is the atomicity contract: the copy carries the donor's identity, so this
    project's own root path and worktree marker may only be stamped once the copy is
    verified — and the incremental update must run against a project that already has them."""
    manager, repo, builder, _rag, _main, worktree = _cloning_world(
        tmp_path,
        nodes=[_SummaryNode("edited.py", "s", "h")],
        worktree_files={"same.py": "m1", "edited.py": "NEW"},
        donor_files={"same.py": "m1", "edited.py": "OLD"},
    )

    await manager.index_directory(str(worktree))

    assert repo.call_log == [
        "clone_project",
        "set_project_root_path",
        "set_project_worktree_meta",
        "set_project_scope",
        "update",
    ]
    assert builder.updates == 1


@pytest.mark.asyncio
async def test_cloned_worktree_identical_to_the_donor_does_not_even_update(tmp_path: Path) -> None:
    """A worktree whose files all match the donor's checksums has nothing left to do: the
    copy *is* the index. Running the incremental path anyway would re-scan the tree and
    re-dump the project's summaries for a guaranteed no-op."""
    manager, repo, builder, rag, main, worktree = _cloning_world(
        tmp_path,
        worktree_files={"same.py": "m1", "also.py": "m2"},
        donor_files={"same.py": "m1", "also.py": "m2"},
    )

    result = await manager.index_directory(str(worktree))

    assert builder.updates == 0 and builder.builds == 0
    assert rag.code_indexes == 0 and rag.summary_indexes == 0
    assert result["changed_files"] == 0
    assert result["worktree_of"] == manager.get_project(str(main))
    assert result["reused"] is False, "the index was created by this run, not reused"


@pytest.mark.asyncio
async def test_cloned_worktree_pays_only_for_the_files_that_differ(tmp_path: Path) -> None:
    """The clone left every other file's chunks and summary vectors already correct in the
    database. Re-writing them would mean a vector-index update per chunk — the exact cost
    the clone exists to avoid — so both paid phases are scoped to the differing files."""
    nodes = [
        _SummaryNode("edited.py", "new summary", "h-new"),
        _SummaryNode("same.py", "old summary", "h-old"),
        _SummaryNode("dir:.", "container", "h-dir", file_path=""),
    ]
    manager, repo, builder, rag, _main, worktree = _cloning_world(
        tmp_path,
        nodes=nodes,
        worktree_files={"same.py": "m1", "edited.py": "NEW", "added.py": "m9"},
        donor_files={"same.py": "m1", "edited.py": "OLD", "gone.py": "m3"},
    )
    builder.update_reconcilable = {"same.py", "edited.py", "added.py", "gone.py"}
    builder.update_dirty_ids = {"edited.py", "dir:."}
    project = manager.get_project(str(worktree))

    result = await manager.index_directory(str(worktree))

    # Changed content, a file the branch added, and a file it deleted — nothing else.
    assert result["changed_files"] == 3
    assert repo.scoped_chunk_embedding_requests == [(project, ["added.py", "edited.py", "gone.py"])]
    assert repo.chunk_embedding_requests == [], "the whole project's vectors were unloaded"
    # Chunks: only nodes of the differing files. Pruning is scoped the same way, and only
    # to files the build actually reconciled.
    assert [n.id for n in rag.code_nodes[0]] == ["edited.py"]
    assert rag.code_reconcilable == [{"edited.py", "added.py", "gone.py"}]
    # Summaries: the differing files plus whatever this run regenerated (a container's
    # summary moves when a child's does, so its vector has to follow).
    assert {n.id for n in rag.summary_nodes[0]} == {"edited.py", "dir:."}
    assert rag.summary_dirty_ids == [{"edited.py", "dir:."}]


@pytest.mark.asyncio
async def test_clone_failure_deletes_the_partial_copy_and_falls_back_to_client_seeding(
    tmp_path: Path,
) -> None:
    """A clone that fails must cost the worktree its speed, not its index: the partial copy
    is deleted and the original, proven client-side seeding runs — still at zero LLM cost."""
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    repo.clone_error = RuntimeError("clone verification failed: chunks source=8100 target=8099")
    builder = FakeGraphBuilder(build_result=([_SummaryNode("a.py", "s", "h1")], []))
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, main)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))
    donor_summaries = {"a.py": {"summary": "s", "summary_input_hash": "h1"}}
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = donor_summaries
    repo.stored_chunk_embeddings[donor] = {"a.py#0": [0.1]}

    result = await manager.index_directory(str(worktree))

    assert repo.deleted_projects == [project], "the partial copy was left behind"
    # ...and the old path really ran: the donor's caches fed the build and the reused
    # summaries were written back into the new project.
    assert builder.builds == 1
    assert builder.build_stored_summaries == [donor_summaries]
    assert repo.chunk_embedding_requests == [donor]
    assert repo.summaries_written == [
        (project, [{"node_id": "a.py", "summary": "s", "summary_input_hash": "h1"}])
    ]
    assert result["worktree_of"] == donor
    assert result["seeded_summaries"] == 1
    assert "cloned" not in result


@pytest.mark.asyncio
async def test_worktree_clone_switched_off_uses_the_client_side_path(tmp_path: Path) -> None:
    manager, repo, builder, _rag, main, worktree = _cloning_world(
        tmp_path, settings=Settings(_env_file=None, worktree_clone=False)
    )
    donor = manager.get_project(str(main))
    repo.stored_summaries[donor] = {"a.py": {"summary": "s", "summary_input_hash": "h1"}}

    result = await manager.index_directory(str(worktree))

    assert repo.clone_calls == []
    assert repo.deleted_projects == []
    assert builder.builds == 1                       # the client-side seeding path
    assert repo.summary_requests == [donor]
    assert "cloned" not in result


@pytest.mark.asyncio
async def test_plain_directory_is_never_cloned(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()
    manager = _worktree_manager(repo, builder, rag, None)

    await manager.index_directory(str(tmp_path))

    assert repo.clone_calls == []
    assert repo.deleted_projects == []
    assert builder.builds == 1


@pytest.mark.asyncio
async def test_the_cost_cap_is_checked_before_anything_is_cloned(tmp_path: Path) -> None:
    """INV-10 outranks the fast path: a worktree whose *uncovered* files would cost more than
    the cap must be refused before a single node is copied, not after."""
    from inflorescence.cost import IndexCostExceededError

    manager, repo, builder, _rag, _main, worktree = _cloning_world(
        tmp_path,
        worktree_files={"same.py": "m1", "huge.py": "NEW"},
        donor_files={"same.py": "m1"},
    )
    builder.estimate_result = {"estimated_tokens": 10_000_000, "estimated_nodes": 5_000}

    with pytest.raises(IndexCostExceededError):
        await manager.index_directory(str(worktree), max_cost_usd=0.0001)

    assert builder.estimated_paths == [["huge.py"]]
    assert repo.clone_calls == []


@pytest.mark.asyncio
async def test_first_query_against_a_worktree_clones_it_within_the_donor_scope(
    tmp_path: Path,
) -> None:
    """The read tools' entry point reaches the clone path too, and must not widen the scan:
    a tree the donor excluded is not to be parsed — let alone paid for — by a mere query."""
    manager, repo, builder, _rag, main, worktree = _cloning_world(
        tmp_path,
        nodes=[_SummaryNode("edited.py", "s", "h")],
        worktree_files={"same.py": "m1", "edited.py": "NEW"},
        donor_files={"same.py": "m1", "edited.py": "OLD"},
    )
    donor = manager.get_project(str(main))
    repo.stored_scope = (["backend/**"], ["vendor/**"])

    project, reason = await manager.ensure_indexed_for_query(str(worktree))

    assert reason is None
    assert repo.clone_calls == [(donor, project)]
    assert repo.scope_requests == [donor]          # inherited, not invented
    assert builder.scan_checksum_calls == [(["backend/**"], ["vendor/**"])]
    assert builder.update_filters == [(["backend/**"], ["vendor/**"])]
    assert repo.scopes == [(project, ["backend/**"], ["vendor/**"])]


@pytest.mark.asyncio
async def test_an_unfinished_clone_is_deleted_and_the_directory_indexed_from_scratch(
    tmp_path: Path,
) -> None:
    """A process killed mid-copy leaves nodes that belong to *another* project. Updating
    them in place would reconcile a fresh parse against a graph that was never whole, so the
    next index throws them away and starts over — and nothing paid-for is lost, because the
    donor still holds every summary and vector."""
    manager, repo, builder, _rag, main, worktree = _cloning_world(tmp_path)
    donor = manager.get_project(str(main))
    project = manager.get_project(str(worktree))
    repo.clone_markers[project] = donor
    # A half-copy really does have nodes: the stats probe must not be what decides.
    repo.stats = {"nodes": 812, "edges": 400, "chunks": 0, "summaries": 0}

    await manager.index_directory(str(worktree))

    assert repo.deleted_projects == [project]
    assert repo.call_log[:2] == ["delete_project", "clone_project"]
    assert builder.updates == 0, "a half-copied graph was reconciled instead of discarded"


@pytest.mark.asyncio
async def test_worktree_detection_failure_degrades_to_a_standalone_index(tmp_path: Path) -> None:
    """Detection is fail-closed by contract; a resolver that raises anyway must not be able
    to abort an index — the worst acceptable outcome is "indexed as an ordinary project"."""
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    rag = FakeRAGIndexer()

    def _boom(path: Path) -> Path | None:
        raise OSError("git metadata unreadable")

    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag,
        settings=Settings(_env_file=None), worktree_resolver=_boom,
    )

    result = await manager.index_directory(str(tmp_path))

    assert result["reused"] is False and builder.builds == 1
    assert repo.worktree_meta == []


# ---------------------------------------------------------------------------
# Auto-index on the first query from a worktree. A read tool that lands on an unindexed
# worktree of an *indexed* checkout indexes it inline — donor-seeded, so parsing and writes
# but no LLM spend — instead of refusing. Every other unindexed directory still refuses,
# because indexing it would cost real money (INV-10), and a *failed* attempt refuses with
# the cause attached rather than raising or returning an empty page (INV-7).
# ---------------------------------------------------------------------------


def _seeded_worktree(
    tmp_path: Path,
    settings: Settings | None = None,
) -> tuple[ProjectManager, _SeedingRepository, FakeGraphBuilder, Path, str]:
    """A worktree whose main checkout is indexed: the one case that may auto-index.

    Pinned to the client-side seeding path unless a test says otherwise, so these assertions
    stay about *what a query is allowed to index* rather than about which mechanism copies
    the donor's cache; the clone path's own auto-index case is covered separately below.
    """
    main, worktree = _worktree_dirs(tmp_path)
    builder = FakeGraphBuilder(build_result=([_SummaryNode("a.py", "s", "h1")], []))
    repo = _SeedingRepository(builder)
    manager = _worktree_manager(
        repo, builder, FakeRAGIndexer(), main, settings=settings or _client_seeding()
    )
    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.stored_summaries[donor] = {"a.py": {"summary": "s", "summary_input_hash": "h1"}}
    return manager, repo, builder, worktree, donor


@pytest.mark.asyncio
async def test_ensure_indexed_for_query_is_a_no_op_for_an_already_indexed_project(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 12, "edges": 34, "chunks": 5, "summaries": 6})
    builder = FakeGraphBuilder()
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), None)
    project = manager.get_project(str(tmp_path))
    repo.indexed_projects.add(project)

    assert await manager.ensure_indexed_for_query(str(tmp_path)) == (project, None)
    # The common case costs one cheap probe: no detection, no scan, no build.
    assert builder.builds == 0 and builder.updates == 0
    assert builder.scan_checksum_calls == []


@pytest.mark.asyncio
async def test_ensure_indexed_for_query_refuses_a_plain_unindexed_directory(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), None)

    project, reason = await manager.ensure_indexed_for_query(str(tmp_path))

    assert project == manager.get_project(str(tmp_path))
    # Indexing this would cost real money, so it stays an explicit, preview-approved act.
    assert reason is not None and "index_directory" in reason
    assert builder.builds == 0


@pytest.mark.asyncio
async def test_first_query_from_a_worktree_auto_indexes_it_seeded_from_the_donor(tmp_path: Path) -> None:
    manager, repo, builder, worktree, donor = _seeded_worktree(tmp_path)

    project, reason = await manager.ensure_indexed_for_query(str(worktree))

    assert reason is None
    assert project == manager.get_project(str(worktree))
    assert builder.builds == 1
    # Seeded, not paid for: the donor's summaries are what the build was handed.
    assert builder.build_stored_summaries == [{"a.py": {"summary": "s", "summary_input_hash": "h1"}}]
    assert repo.worktree_meta == [(project, donor, str((tmp_path / "main").resolve()))]


@pytest.mark.asyncio
async def test_auto_index_inherits_the_donor_scan_scope(tmp_path: Path) -> None:
    """The worktree must scan the universe its donor scanned: a path excluded there —
    vendor/**, generated code — must not be indexed here, for money, by a mere query."""
    manager, repo, builder, worktree, donor = _seeded_worktree(tmp_path)
    repo.stored_scope = (["backend/**"], ["vendor/**"])

    await manager.ensure_indexed_for_query(str(worktree))

    assert repo.scope_requests == [donor]        # inherited from the donor, not from nowhere
    assert builder.build_filters == [(["backend/**"], ["vendor/**"])]
    assert builder.scan_checksum_calls == [(["backend/**"], ["vendor/**"])]


@pytest.mark.asyncio
async def test_auto_index_of_an_unscoped_donor_passes_no_filters(tmp_path: Path) -> None:
    """An empty stored scope means "no filters", not "include nothing"."""
    manager, repo, builder, worktree, _ = _seeded_worktree(tmp_path)
    repo.stored_scope = ([], [])

    await manager.ensure_indexed_for_query(str(worktree))

    assert builder.build_filters == [(None, None)]


@pytest.mark.asyncio
async def test_failed_auto_index_reports_the_cause_instead_of_raising(tmp_path: Path) -> None:
    manager, _repo, builder, worktree, _donor = _seeded_worktree(tmp_path)
    builder.preflight_error = RuntimeError("OPENROUTER_API_KEY is not set")

    project, reason = await manager.ensure_indexed_for_query(str(worktree))

    assert project == manager.get_project(str(worktree))
    assert reason is not None and reason.startswith("auto-index attempted and failed:")
    assert "OPENROUTER_API_KEY is not set" in reason
    assert builder.builds == 0                    # aborted before the first write (INV-2)


@pytest.mark.asyncio
async def test_auto_index_blocked_by_the_cost_gate_reports_the_estimate(tmp_path: Path) -> None:
    """The gate is the other realistic failure: a branch that rewrote half the tree is not
    free, and the cap must still bite — reported, not raised."""
    settings = Settings(_env_file=None, max_index_cost_usd=0.0)
    manager, _repo, builder, worktree, _donor = _seeded_worktree(tmp_path, settings=settings)
    builder.scan_checksums_result = {"rewritten.py": "new"}
    builder.estimate_result = {"estimated_tokens": 50_000_000, "estimated_nodes": 100_000}

    _project, reason = await manager.ensure_indexed_for_query(str(worktree))

    assert reason is not None and reason.startswith("auto-index attempted and failed:")
    assert "exceeds the limit" in reason
    assert builder.builds == 0


@pytest.mark.asyncio
async def test_auto_index_is_refused_when_the_worktree_flag_is_off(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, worktree_auto_index=False)
    manager, _repo, builder, worktree, _donor = _seeded_worktree(tmp_path, settings=settings)

    _project, reason = await manager.ensure_indexed_for_query(str(worktree))

    assert reason is not None and "index_directory" in reason
    assert builder.builds == 0


@pytest.mark.asyncio
async def test_auto_index_is_refused_for_a_worktree_whose_main_checkout_is_unindexed(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    builder = FakeGraphBuilder()
    repo = _SeedingRepository(builder)
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), main)

    _project, reason = await manager.ensure_indexed_for_query(str(worktree))

    # Nothing to seed from, so this index would be paid for in full — refuse it.
    assert reason is not None and "index_directory" in reason
    assert builder.builds == 0


@pytest.mark.asyncio
async def test_concurrent_first_queries_from_one_worktree_build_once(tmp_path: Path) -> None:
    """N parallel first queries must not start N indexes: the per-project lock serializes
    them and the loser degrades to a checksum no-op."""
    import asyncio

    manager, _repo, builder, worktree, _donor = _seeded_worktree(tmp_path)
    original_build = builder.build

    async def slow_build(*args: object, **kwargs: object) -> BuildResult:
        await asyncio.sleep(0.02)      # hold the lock long enough for the racer to pile up
        return await original_build(*args, **kwargs)

    builder.build = slow_build  # type: ignore[method-assign]

    results = await asyncio.gather(
        manager.ensure_indexed_for_query(str(worktree)),
        manager.ensure_indexed_for_query(str(worktree)),
    )

    assert [reason for _project, reason in results] == [None, None]
    assert builder.builds == 1
    assert builder.updates == 1          # the second call found the index and re-checked it


# ---------------------------------------------------------------------------
# Honest preview for a worktree: quoting the whole tree advertised ~$50 for an index that
# donor seeding makes ~$0. The estimate must price the uncovered files only, and say why.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_of_a_worktree_prices_only_what_the_donor_does_not_cover(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    builder.scan_checksums_result = {"same.py": "m1", "edited.py": "NEW", "added.py": "m9"}
    builder.estimate_result = {"estimated_tokens": 40, "estimated_nodes": 4}
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), main)
    donor = manager.get_project(str(main))
    repo.indexed_projects.add(donor)
    repo.checksums[donor] = {"same.py": "m1", "edited.py": "OLD", "gone.py": "m3"}

    result = await manager.preview_directory(str(worktree))

    assert builder.previews == 0                          # the full tree is not what is priced
    assert builder.estimated_paths == [["added.py", "edited.py"]]
    assert builder.scan_checksum_calls == [(None, None)]
    assert result["preview"] is True
    assert result["worktree_of"] == donor
    assert result["uncovered_files"] == 2
    assert result["donor_covered_files"] == 1             # same.py; the donor's gone.py is gone
    assert result["files"] == 2                           # "files" = what this index would pay for
    assert result["estimated_tokens"] == 40
    assert "estimated_cost_usd" in result


@pytest.mark.asyncio
async def test_a_worktree_preview_names_its_biggest_files_and_what_each_costs(tmp_path: Path) -> None:
    """"$0.62 for 64 files" with no breakdown was abandoned; the biggest files are what to ignore."""
    from inflorescence.code_indexer.graph_builder import GraphBuilder
    from inflorescence.code_indexer.models import IndexerConfig

    main, worktree = _worktree_dirs(tmp_path)
    (worktree / "table.json").write_text("[" + ",\n".join(["1"] * 400) + "]\n")
    (worktree / "added.py").write_text("x = 1\n")
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    builder.scan_checksums_result = {"added.py": "a", "table.json": "t"}
    real = GraphBuilder(repo=None, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    builder.estimate_paths = real.estimate_paths  # type: ignore[method-assign]
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), main)
    repo.indexed_projects.add(manager.get_project(str(main)))
    repo.checksums[manager.get_project(str(main))] = {}

    result = await manager.preview_directory(str(worktree))

    largest = result["largest_files"]
    assert [f["path"] for f in largest] == ["table.json", "added.py"]
    assert set(largest[0]) == {"path", "bytes", "estimated_cost_usd"}
    assert largest[0]["estimated_cost_usd"] > largest[1]["estimated_cost_usd"]
    assert sum(f["estimated_cost_usd"] for f in largest) == pytest.approx(result["estimated_cost_usd"], abs=1e-3)


@pytest.mark.asyncio
async def test_preview_of_a_plain_directory_still_prices_the_whole_tree(tmp_path: Path) -> None:
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), None)

    result = await manager.preview_directory(str(tmp_path))

    assert builder.previews == 1
    assert builder.scan_checksum_calls == []
    assert result["files"] == 3 and "worktree_of" not in result


@pytest.mark.asyncio
async def test_preview_of_a_worktree_without_an_indexed_donor_prices_the_whole_tree(tmp_path: Path) -> None:
    main, worktree = _worktree_dirs(tmp_path)
    repo = FakeProjectRepository({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
    builder = FakeGraphBuilder()
    manager = _worktree_manager(repo, builder, FakeRAGIndexer(), main)

    result = await manager.preview_directory(str(worktree))

    # No donor means no seeding, so the whole tree really will be paid for.
    assert builder.previews == 1
    assert result["files"] == 3 and "worktree_of" not in result


def test_worktree_settings_default_to_on() -> None:
    settings = Settings(_env_file=None)
    assert settings.worktree_auto_index is True
    assert settings.worktree_janitor_enabled is True


@pytest.mark.asyncio
async def test_repository_worktree_meta_round_trip() -> None:
    from inflorescence.db.repository import GraphRepository

    conn = _RecordingConn([{"worktree_of": "main_abc12345", "main_root_path": "/repo/main"}])
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    await repo.set_project_worktree_meta("wt_deadbeef", "main_abc12345", "/repo/main")
    assert conn.writes == [{
        "project": "wt_deadbeef",
        "worktree_of": "main_abc12345",
        "main_root_path": "/repo/main",
    }]
    assert await repo.get_project_worktree_meta("wt_deadbeef") == {
        "worktree_of": "main_abc12345",
        "main_root_path": "/repo/main",
    }

    # An ordinary project's root node carries no marker: absent row, or a null property.
    assert await GraphRepository(_RecordingConn([])).get_project_worktree_meta("p") is None  # type: ignore[arg-type]
    blank = _RecordingConn([{"worktree_of": None, "main_root_path": None}])
    assert await GraphRepository(blank).get_project_worktree_meta("p") is None  # type: ignore[arg-type]


class _RecordingConn:
    """Minimal MemgraphConnection stand-in: records write params, replays canned read rows."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.writes: list[dict[str, object]] = []

    async def execute_write(self, query: str, params: dict[str, object] | None = None) -> None:
        self.writes.append(dict(params or {}))

    async def execute_query(
        self, query: str, params: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        return self.rows


class FakeProjectRepository:
    def __init__(
        self,
        stats: dict[str, int],
        unembedded_summary_nodes: list[object] | None = None,
        files_missing_chunks: list[str] | None = None,
        nodes_for_files: list[object] | None = None,
        missing_contains_edges: list[tuple[str, str]] | None = None,
    ) -> None:
        self.stats = stats
        self.expected_hash = ""
        self.root_paths: list[tuple[str, str]] = []
        self.unembedded_summary_nodes = unembedded_summary_nodes or []
        self.files_missing_chunks = files_missing_chunks or []
        self.nodes_for_files = nodes_for_files or []
        self.missing_contains_edges = missing_contains_edges or []
        self.repaired_contains_edges: list[tuple[str, str]] = []
        self.files_requested: list[list[str]] = []
        self.scopes: list[tuple[str, list[str], list[str]]] = []
        self.stored_scope: tuple[list[str], list[str]] = ([], [])
        # Which project's scope was read back — an auto-index must inherit the *donor's*.
        self.scope_requests: list[str] = []
        # Donor (worktree seeding) state, keyed by project id, plus a call log per read so a
        # test can assert that an ordinary directory triggers no donor lookups at all.
        self.indexed_projects: set[str] = set()
        self.checksums: dict[str, dict[str, str]] = {}
        self.stored_summaries: dict[str, dict[str, dict[str, object]]] = {}
        self.stored_chunk_embeddings: dict[str, dict[str, list[float]]] = {}
        self.project_is_indexed_calls: list[str] = []
        self.checksum_requests: list[str] = []
        self.summary_requests: list[str] = []
        self.light_summary_requests: list[str] = []
        self.chunk_embedding_requests: list[str] = []
        self.summaries_written: list[tuple[str, list[dict[str, object]]]] = []
        self.worktree_meta: list[tuple[str, str, str]] = []
        # Server-side clone state. `call_log` records the ordered sequence of calls that the
        # clone path's contract is about (copy, then identity, then the incremental update) —
        # the builder appends to the same list when a test hands it over.
        self.call_log: list[str] = []
        self.clone_calls: list[tuple[str, str]] = []
        self.clone_error: BaseException | None = None
        self.clone_counts = {"nodes": 5, "chunks": 3, "externals": 1, "edges": 7}
        self.clone_markers: dict[str, str] = {}
        self.deleted_projects: list[str] = []
        self.scoped_chunk_embedding_requests: list[tuple[str, list[str]]] = []

    async def get_project_stats(self, project: str) -> dict[str, int]:
        self.expected_hash = project.rsplit("_", 1)[-1]
        return self.stats

    async def project_is_indexed(self, project: str) -> bool:
        self.project_is_indexed_calls.append(project)
        return project in self.indexed_projects

    async def get_checksums(self, project: str) -> dict[str, str]:
        self.checksum_requests.append(project)
        return dict(self.checksums.get(project, {"module.py": "abc123"}))

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        self.summary_requests.append(project)
        return self.stored_summaries.get(project, {})

    async def get_stored_summaries_light(self, project: str) -> dict[str, dict[str, object]]:
        """The vector-free read. Recorded apart from :meth:`get_stored_summaries` so a test
        can pin *which* of the two a path uses — the difference is megabytes of Bolt traffic
        on a real project, and only cross-project seeding has any use for the vectors."""
        self.light_summary_requests.append(project)
        return {
            nid: {
                "summary": entry.get("summary"),
                "summary_input_hash": entry.get("summary_input_hash"),
                "summary_embedding_hash": entry.get("summary_embedding_hash"),
                "has_embedding": entry.get("summary_embedding") is not None,
            }
            for nid, entry in self.stored_summaries.get(project, {}).items()
        }

    async def get_stored_chunk_embeddings(self, project: str) -> dict[str, list[float]]:
        self.chunk_embedding_requests.append(project)
        return self.stored_chunk_embeddings.get(project, {})

    async def store_summaries(self, project: str, items: list[dict[str, object]]) -> int:
        rows = [dict(item) for item in items]
        self.summaries_written.append((project, rows))
        return len(rows)

    async def set_project_worktree_meta(
        self, project: str, worktree_of: str, main_root_path: str
    ) -> None:
        self.call_log.append("set_project_worktree_meta")
        self.worktree_meta.append((project, worktree_of, main_root_path))

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        self.call_log.append("set_project_root_path")
        self.root_paths.append((project, root_path))

    async def set_project_scope(self, project: str, include: list[str], exclude: list[str]) -> None:
        self.call_log.append("set_project_scope")
        self.scopes.append((project, list(include), list(exclude)))

    async def get_clone_marker(self, project: str) -> str | None:
        return self.clone_markers.get(project)

    async def clone_project(self, source: str, target: str) -> dict[str, int]:
        self.call_log.append("clone_project")
        self.clone_calls.append((source, target))
        if self.clone_error is not None:
            raise self.clone_error
        self.indexed_projects.add(target)
        return dict(self.clone_counts)

    async def delete_project(self, project: str) -> None:
        self.call_log.append("delete_project")
        self.deleted_projects.append(project)
        self.clone_markers.pop(project, None)
        # A deleted project really is empty afterwards: the caller decides "first index or
        # incremental" from these counts, and a fake that kept reporting the old ones would
        # hide exactly the bug this models (a half-copy reconciled instead of rebuilt).
        self.stats = {"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0}

    async def get_stored_chunk_embeddings_for_files(
        self, project: str, file_paths: list[str]
    ) -> dict[str, list[float]]:
        self.scoped_chunk_embedding_requests.append((project, list(file_paths)))
        return dict(self.stored_chunk_embeddings.get(project, {}))

    async def get_project_scope(self, project: str) -> tuple[list[str], list[str]]:
        self.scope_requests.append(project)
        return self.stored_scope

    async def get_stale_summary_embedding_nodes(self, project: str) -> list[object]:
        return self.unembedded_summary_nodes

    async def get_files_with_stale_chunks(self, project: str) -> list[str]:
        return self.files_missing_chunks

    async def get_nodes_for_files(self, project: str, file_paths: list[str]) -> list[object]:
        self.files_requested.append(list(file_paths))
        return self.nodes_for_files

    async def get_nodes_missing_contains_edges(self, project: str) -> list[tuple[str, str]]:
        return self.missing_contains_edges

    async def repair_contains_edges(self, project: str, pairs: list[tuple[str, str]]) -> int:
        self.repaired_contains_edges = list(pairs)
        return len(pairs)


class FakeGraphBuilder:
    def __init__(
        self,
        update_result: tuple[list[object], list[object]] | None = None,
        repair_result: list[object] | None = None,
        build_result: tuple[list[object], list[object]] | None = None,
    ) -> None:
        self.builds = 0
        self.updates = 0
        self.repairs = 0
        self.preflights = 0
        self.update_result = update_result or ([], [])
        self.build_result = build_result or ([], [])
        self.repair_result = repair_result or []
        self.build_filters: list[tuple[list[str] | None, list[str] | None]] = []
        self.update_filters: list[tuple[list[str] | None, list[str] | None]] = []
        self.preflight_error: BaseException | None = None
        # {estimated_tokens, estimated_nodes} the update-cost gate reads; zero by default so
        # a tiny changed set never trips the cap in tests that don't exercise it.
        self.estimate_result: dict[str, object] = {"estimated_tokens": 0, "estimated_nodes": 0}
        # The no-op gate consults this cheaply before any paid work; default True so the
        # normal "there is work to do" path runs. A no-op test flips it to False and asserts
        # neither preflight nor the vector dump happened.
        self.pending_changes = True
        # The files a re-index would pay for, as the cost gate on the incremental path reads
        # them. Empty by default so that gate short-circuits and tests about other things are
        # unaffected; a gate test fills it and sets `estimate_result` to something expensive.
        self.pending_change_paths_result: list[str] = []
        self.pending_change_path_calls: list[tuple[list[str] | None, list[str] | None]] = []
        # Worktree seeding observation: which stored_summaries the build was handed, which
        # paths the cost gate priced, and how often the full-scan preview was used at all.
        self.build_stored_summaries: list[dict[str, dict[str, object]] | None] = []
        self.build_dirty_ids: set[str] = set()
        self.previews = 0
        self.estimated_paths: list[list[str]] = []
        self.scan_checksums_result: dict[str, str] = {}
        self.scan_checksum_calls: list[tuple[list[str] | None, list[str] | None]] = []
        # Set to a FakeProjectRepository's list so a test can assert the *order* of the
        # clone path's steps across both fakes (copy -> identity -> update).
        self.call_log: list[str] | None = None
        # What an update reports back about itself; the clone path scopes the paid RAG
        # phases with these, so a test can prove only the changed files were re-processed.
        self.update_reconcilable: set[str] = set()
        self.update_dirty_ids: set[str] = set()
        self.update_stored_summaries: dict[str, dict[str, object]] = {}
        # The files an update found different. `None` (a full build reports nothing) means
        # "not narrowed"; the manager scopes its chunk-vector read to this set.
        self.update_changed_files: set[str] | None = None

    def scan_checksums(
        self,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, str]:
        self.scan_checksum_calls.append((include_patterns, exclude_patterns))
        return dict(self.scan_checksums_result)

    async def has_pending_changes(
        self,
        project: str,
        root_path: Path,
        changed_files: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        return self.pending_changes

    async def pending_change_paths(
        self,
        project: str,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[str]:
        self.pending_change_path_calls.append((include_patterns, exclude_patterns))
        return list(self.pending_change_paths_result)

    async def preflight(self) -> None:
        self.preflights += 1
        if self.preflight_error is not None:
            raise self.preflight_error

    def estimate_paths(self, root_path: Path, paths: list[str]) -> dict[str, object]:
        self.estimated_paths.append(list(paths))
        return self.estimate_result

    async def build(
        self,
        project: str,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        stored_summaries: dict[str, dict[str, object]] | None = None,
        progress: object | None = None,
    ) -> BuildResult:
        self.builds += 1
        self.build_filters.append((include_patterns, exclude_patterns))
        self.build_stored_summaries.append(stored_summaries)
        # Mirrors the real builder: whatever stored summaries came in are handed back on the
        # result, which is what lets index_summaries reuse the donor's vectors.
        return BuildResult(
            nodes=list(self.build_result[0]),
            edges=list(self.build_result[1]),
            stored_summaries=stored_summaries or {},
            dirty_summary_ids=set(self.build_dirty_ids),
        )

    async def update(
        self,
        project: str,
        root_path: Path,
        changed_files: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        progress: object | None = None,
    ) -> BuildResult:
        self.updates += 1
        if self.call_log is not None:
            self.call_log.append("update")
        self.update_filters.append((include_patterns, exclude_patterns))
        nodes, edges = self.update_result
        return BuildResult(
            nodes=list(nodes),
            edges=list(edges),
            stored_summaries=dict(self.update_stored_summaries),
            dirty_summary_ids=set(self.update_dirty_ids),
            reconcilable_files=set(self.update_reconcilable),
            changed_files=(
                set(self.update_changed_files) if self.update_changed_files is not None else None
            ),
        )

    async def repair_missing_summaries(self, project: str, root_path: Path) -> list[object]:
        self.repairs += 1
        return self.repair_result

    preview_result: dict[str, object] | None = None

    def preview(
        self,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, object]:
        self.previews += 1
        return self.preview_result or {
            "files": 3,
            "total_bytes": 120,
            "languages": {"python": {"files": 3, "bytes": 120}},
            "largest_files": [],
            "estimated_nodes": 5,
            "estimated_tokens": 30,
        }


class FakeRAGIndexer:
    def __init__(self, code_count: int = 0, summary_count: int = 0) -> None:
        self.code_indexes = 0
        self.summary_indexes = 0
        self.code_count = code_count
        self.summary_count = summary_count
        self.code_nodes: list[list[object]] = []
        self.summary_nodes: list[list[object]] = []
        # Reuse plumbing: the caches each phase was handed, so a test can prove the donor's
        # vectors (not a fresh embed) are what the run will store.
        self.code_stored_embeddings: list[dict[str, list[float]] | None] = []
        self.code_reconcilable: list[set[str] | None] = []
        self.summary_stored_summaries: list[dict[str, dict[str, object]] | None] = []
        self.summary_dirty_ids: list[set[str] | None] = []

    preflight_error: BaseException | None = None

    def preflight(self) -> None:
        if self.preflight_error is not None:
            raise self.preflight_error

    async def index_code(
        self,
        project: str,
        root_path: Path,
        nodes: list[object],
        stored_chunk_embeddings: dict[str, list[float]] | None = None,
        prune_stale: bool = False,
        reconcilable_files: set[str] | None = None,
        progress: object | None = None,
    ) -> int:
        self.code_indexes += 1
        self.code_nodes.append(list(nodes))
        self.code_stored_embeddings.append(stored_chunk_embeddings)
        self.code_reconcilable.append(reconcilable_files)
        return self.code_count

    async def index_summaries(
        self,
        project: str,
        nodes: list[object],
        stored_summaries: dict[str, dict[str, object]] | None = None,
        dirty_ids: set[str] | None = None,
        progress: object | None = None,
    ) -> int:
        self.summary_indexes += 1
        self.summary_nodes.append(list(nodes))
        self.summary_stored_summaries.append(stored_summaries)
        self.summary_dirty_ids.append(dirty_ids)
        return self.summary_count


class _SeedingRepository(FakeProjectRepository):
    """A repository whose answers reflect the index a build has actually written.

    Two callers need that: ``ensure_indexed_for_query`` re-probes after auto-indexing (a run
    that stored nothing must not be reported as queryable), and a second, racing ensure must
    see the first one's nodes so it degrades to a no-op instead of building again.
    """

    def __init__(self, builder: FakeGraphBuilder) -> None:
        super().__init__({"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0})
        self._builder = builder

    async def get_project_stats(self, project: str) -> dict[str, int]:
        self.expected_hash = project.rsplit("_", 1)[-1]
        if self._builder.builds:
            return {"nodes": 7, "edges": 2, "chunks": 1, "summaries": 1}
        return self.stats

    async def project_is_indexed(self, project: str) -> bool:
        self.project_is_indexed_calls.append(project)
        return project in self.indexed_projects or self._builder.builds > 0


class _FakeRecord:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def data(self) -> dict[str, object]:
        return self._data


class FakeConnMissingSearchIndexes:
    """A connection to a plain Memgraph: property indexes exist, vector/text ones don't.

    Grants the project lease immediately (so the mutation path is reached) and reports only
    property indexes from SHOW INDEX INFO, so ``missing_search_indexes`` finds both vector
    indexes absent — the DB-5 search-less server the preflight must refuse.
    """

    async def execute_write_query(
        self, query: str, params: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        token = (params or {}).get("token")
        return [{"acquired": True, "holder": token}]

    async def execute_query(
        self, query: str, params: dict[str, object] | None = None
    ) -> list[_FakeRecord]:
        return [
            _FakeRecord({"index type": "label+property", "label": "Code", "property": ["id"], "name": ""}),
            _FakeRecord({"index type": "label+property", "label": "Code", "property": ["project"], "name": ""}),
        ]
