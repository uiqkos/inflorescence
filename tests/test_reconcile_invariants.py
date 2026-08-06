"""G2 · Update must not destroy good data — invariant oracles for INV-1/INV-3.

The contract: neither a failure (read/parse/network) nor a degraded environment (a
missing language parser) may delete previously-indexed data. "Absent from this scan
because something broke" is NOT "deleted". Deletion is reserved for a file confirmed
gone from disk. Edge updates never pass through an edgeless state.

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
