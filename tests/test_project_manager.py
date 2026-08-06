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

    async def get_project_stats(self, project: str) -> dict[str, int]:
        self.expected_hash = project.rsplit("_", 1)[-1]
        return self.stats

    async def get_checksums(self, project: str) -> dict[str, str]:
        return {"module.py": "abc123"}

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        return {}

    async def get_stored_chunk_embeddings(self, project: str) -> dict[str, list[float]]:
        return {}

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        self.root_paths.append((project, root_path))

    async def set_project_scope(self, project: str, include: list[str], exclude: list[str]) -> None:
        self.scopes.append((project, list(include), list(exclude)))

    async def get_project_scope(self, project: str) -> tuple[list[str], list[str]]:
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

    async def has_pending_changes(
        self,
        project: str,
        root_path: Path,
        changed_files: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        return self.pending_changes

    async def preflight(self) -> None:
        self.preflights += 1
        if self.preflight_error is not None:
            raise self.preflight_error

    def estimate_paths(self, root_path: Path, paths: list[str]) -> dict[str, object]:
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
        return BuildResult(nodes=list(self.build_result[0]), edges=list(self.build_result[1]))

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
        self.update_filters.append((include_patterns, exclude_patterns))
        nodes, edges = self.update_result
        return BuildResult(nodes=list(nodes), edges=list(edges))

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
        return self.summary_count


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
