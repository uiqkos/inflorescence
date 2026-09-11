"""G2 · Update must not destroy good data — invariant oracles for INV-1/INV-3.

The contract: neither a failure (read/parse/network) nor a degraded environment (a
missing language parser) may delete previously-indexed data. "Absent from this scan
because something broke" is NOT "deleted". Deletion is reserved for a file confirmed
gone from disk — or for the one explicit, opt-in operation that exists for narrowing a
project's scope (``inflorescence prune`` / ``ProjectManager.prune_out_of_scope``), which
an ordinary index never performs on its own. Edge updates never pass through an edgeless
state.

These are **live** oracles against a real Memgraph — a wipe/reconcile race and a
crash-in-the-window cannot be reproduced against a mock. They are skipped unless
``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a **throwaway** instance, so CI and a
developer's real ``bolt://localhost:7687`` are never touched. Findings 3, 4 and the
destructive half of 9 are structural, so no LLM/embedding key is needed.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import CodeNode, Edge, IndexerConfig, NodeType
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.base_parser import BaseParser
from inflorescence.code_indexer.parser.registry import ParserRegistry
from inflorescence.config import Settings

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_NO_LLM = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)


class _XyzParser(BaseParser):
    """A trivial parser for a made-up ``.xyz`` language — one module node per file.

    Lets a test index a second "language" and then rebuild with a registry that no
    longer has it, reproducing a parser going missing (registry degradation) without
    depending on which tree-sitter extras happen to be installed.
    """

    def get_supported_extensions(self) -> list[str]:
        return [".xyz"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        rel = str(file_path.relative_to(root_path))
        return [CodeNode(id=rel, name=file_path.stem, node_type=NodeType.MODULE,
                         file_path=rel, line_start=1, line_end=1)], []


async def _fresh_repo():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    return GraphRepository(conn), conn


def _registry(*parsers: BaseParser) -> ParserRegistry:
    reg = ParserRegistry()
    for p in parsers:
        reg.register(p)
    return reg


@live
@pytest.mark.asyncio
async def test_missing_language_parser_keeps_that_language_stale_not_deleted(tmp_path: Path) -> None:
    """Finding 3: a language whose parser vanished must survive a rebuild as stale data.

    Before the fix, ``.xyz`` fell out of ``extensions`` → out of the scan → its nodes
    were computed as ``stored − scanned`` and deleted wholesale. Now "not scanned
    because unsupported" is not "deleted": the nodes are kept.
    """
    repo, conn = await _fresh_repo()
    project = f"g2-registry-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "note.xyz").write_text("hello\n", encoding="utf-8")
    try:
        full = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser(), _XyzParser()), config=_NO_LLM)
        await full.build(project, tmp_path)
        assert "note.xyz" in await repo.get_node_ids(project)  # both languages indexed

        # The .xyz parser is gone. Touch a.py and reconcile with the degraded registry.
        (tmp_path / "a.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
        degraded = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)
        await degraded.update(project, tmp_path, changed_files=[str((tmp_path / "a.py").resolve())])

        stored = await repo.get_node_ids(project)
        assert "note.xyz" in stored, "orphaned-language node was destroyed instead of kept stale"
        assert "a.py" in stored
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_a_deleted_name_routed_document_is_swept_not_kept_stale(tmp_path: Path) -> None:
    """A Makefile deleted from disk must be *deleted*, like any other indexed file.

    The delete/keep decision used to ask ``Path(fp).suffix in extensions``. A Makefile has no
    suffix, so admitting it by name without changing that question would have filed every
    deleted build file under "its parser vanished, keep it stale" — a ghost node that no
    reconcile ever sweeps, pointing at a file that is gone. Same predicate for both routes
    (``GraphBuilder.indexes``) is what keeps INV-1 and this from pulling in opposite directions.
    """
    repo, conn = await _fresh_repo()
    project = f"g2-named-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("run:\n\tpython -m app\n", encoding="utf-8")
    try:
        builder = GraphBuilder(repo=repo, llm=None, config=_NO_LLM)
        await builder.build(project, tmp_path)
        assert "Makefile" in await repo.get_node_ids(project)

        (tmp_path / "Makefile").unlink()
        await builder.update(project, tmp_path, changed_files=[str((tmp_path / "Makefile").resolve())])

        stored = await repo.get_node_ids(project)
        assert "Makefile" not in stored, "a deleted document was kept as a ghost node"
        assert "a.py" in stored
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_parse_failure_during_rebuild_keeps_the_file_stale(tmp_path: Path) -> None:
    """Finding 4: a file that fails to parse mid-rebuild keeps its stored nodes.

    A parser returns ``[], []`` both for "empty file" and for a read/parse failure.
    Every parser emits at least a module node for a file it truly read, so an empty
    result means failure — and a failure must never be read as "delete this file".
    Here b.py is broken while a.py's edit drives the rebuild.
    """
    repo, conn = await _fresh_repo()
    project = f"g2-parsefail-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    try:
        builder = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)
        await builder.build(project, tmp_path)
        assert {"b.py", "b.py::bar"} <= await repo.get_node_ids(project)

        # b.py becomes unparseable (stands in for a transient read/parse failure);
        # a.py's edit triggers a full reconcile.
        (tmp_path / "b.py").write_text("def bar(:\n    syntax error\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("def keep():\n    return 99\n", encoding="utf-8")
        await builder.update(project, tmp_path, changed_files=[str((tmp_path / "a.py").resolve())])

        stored = await repo.get_node_ids(project)
        assert {"b.py", "b.py::bar"} <= stored, "a parse failure deleted the file's good nodes"
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_edge_update_never_passes_through_edgeless_state(tmp_path: Path) -> None:
    """Finding 9: edges are MERGEd before any deletion, so a crash never empties them.

    Reproduces the deletion-only-trigger crash: delete a file, then kill the run at the
    final stale-edge prune. The old wipe-then-recreate left the graph with *no* edges;
    the reconcile leaves the surviving file's edges intact.
    """
    repo, conn = await _fresh_repo()
    project = f"g2-edges-{uuid.uuid4().hex[:8]}"
    (tmp_path / "a.py").write_text("import b\n\ndef call():\n    return b.target()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    try:
        builder = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)
        await builder.build(project, tmp_path)
        assert len(await repo.get_code_edges(project)) > 0

        # Kill the run at node deletion — the step that in the OLD code came right after
        # the edge wipe (delete_code_edges) and before edges were recreated, i.e. exactly
        # the window that left the graph edgeless. The reconcile MERGEs edges before any
        # deletion, so at this point edges are already in place.
        async def _boom(*_a, **_k):
            raise RuntimeError("killed mid-reconcile")

        repo.delete_nodes_by_ids = _boom  # type: ignore[method-assign]
        (tmp_path / "b.py").unlink()  # deletion-only trigger

        with pytest.raises(RuntimeError):
            await builder.update(project, tmp_path, changed_files=[str((tmp_path / "b.py").resolve())])

        # The crash happened *after* fresh edges were MERGEd in — the graph is not edgeless.
        edges = await repo.get_code_edges(project)
        assert len(edges) > 0, "reconcile left the graph with no edges after a crash"
        assert ("dir:.", "CONTAINS", "a.py") in edges, "surviving file's structural edges were lost"
    finally:
        await repo.delete_project(project)
        await conn.close()


# ---------------------------------------------------------------------------
# Narrowing scope · a re-index never deletes; `prune` does, and only when asked
# ---------------------------------------------------------------------------


def _manager_for(repo, conn, builder):
    """A ProjectManager wired for structural work only — prune touches no LLM and no RAG."""
    from inflorescence.project_manager import ProjectManager
    from inflorescence.rag.indexer import RAGIndexer

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    return ProjectManager(
        repo=repo,
        graph_builder=builder,
        rag_indexer=RAGIndexer(repo=repo, settings=settings),
        settings=settings,
        conn=conn,
    )


@live
@pytest.mark.asyncio
async def test_reindex_with_exclude_keeps_nodes_and_prune_removes_them(tmp_path: Path) -> None:
    """Excluding a path must not delete on re-index; the explicit prune must.

    The two halves are one invariant seen from both sides. A narrowed ``exclude_patterns``
    looks exactly like a scan that silently lost those files, so the reconcile must keep
    them (INV-1/INV-3) — which is also why the fixtures that motivated this survived every
    re-index. Removing them is therefore its own command, and it must take the file's nodes,
    its chunks and the directory nodes left holding nothing, while everything in scope and
    the project's paid summaries stay put.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    fixtures = tmp_path / "tests" / "golden" / "site.example"
    fixtures.mkdir(parents=True)
    (fixtures / "case.py").write_text("def fixture_case():\n    return 2\n", encoding="utf-8")
    (tmp_path / "tests" / "golden" / "other.py").write_text("def other():\n    return 3\n", encoding="utf-8")

    repo, conn = await _fresh_repo()
    builder = GraphBuilder(repo=repo, llm=None, registry=_registry(PythonAstParser()), config=_NO_LLM)
    manager = _manager_for(repo, conn, builder)
    # `prune` addresses a project by directory, exactly as the CLI does, so the fixture has
    # to be indexed under that same derived id rather than a hand-picked one.
    project = manager.get_project(str(tmp_path))
    try:
        await builder.build(project, tmp_path)
        stored = await repo.get_node_ids(project)
        assert "src/app.py" in stored
        assert "tests/golden/site.example/case.py" in stored

        # 1. A re-index that excludes the fixtures leaves them alone: they are still on disk,
        #    so nothing distinguishes "excluded" from "the scan lost them".
        (tmp_path / "src" / "app.py").write_text("def app():\n    return 11\n", encoding="utf-8")
        await builder.update(project, tmp_path, exclude_patterns=["tests/golden"])
        stored = await repo.get_node_ids(project)
        assert "tests/golden/site.example/case.py" in stored, "a re-index deleted an excluded file"

        # 2. The dry run reports them and writes nothing.
        plan = await manager.prune_out_of_scope(
            str(tmp_path), exclude_patterns=["tests/golden"], dry_run=True
        )
        assert plan["files"] == 2, plan
        assert plan["dry_run"] is True
        assert "tests/golden/site.example/case.py" in await repo.get_node_ids(project)

        # 3. The real run removes exactly the out-of-scope subtree.
        applied = await manager.prune_out_of_scope(str(tmp_path), exclude_patterns=["tests/golden"])
        assert applied["files"] == 2, applied
        stored = await repo.get_node_ids(project)
        assert "src/app.py" in stored, "prune removed a file that is still in scope"
        assert not any(nid.startswith("tests/golden") for nid in stored), stored
        assert not any(nid == "dir:tests/golden" or nid.startswith("dir:tests/golden/") for nid in stored), (
            "directory nodes left behind with nothing under them"
        )

        # 4. The narrowing is persisted, so a watcher update reproduces it.
        _include, exclude = await repo.get_project_scope(project)
        assert "tests/golden" in exclude, exclude

        # 5. Running it again is a no-op rather than an error.
        again = await manager.prune_out_of_scope(str(tmp_path), exclude_patterns=["tests/golden"])
        assert again["files"] == 0, again
    finally:
        await repo.delete_project(project)
        await conn.close()
