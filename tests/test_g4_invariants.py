"""G4 · Fail-closed dependencies and spend discipline — live oracles for INV-2/INV-10/INV-5.

The contract:
  * No mutation starts when ANY dependency it needs is unavailable — both embedding
    endpoints AND the LLM chat model are proven before the first write (finding 6 / INV-2).
  * A background watcher update obeys the same cost cap (finding 5b) and the same scan
    scope (finding 5c) as the explicit index that created the project (INV-10 / INV-6).
  * Repeat work on unchanged content costs 0 API calls — an unchanged broken .py is not
    re-paid to the LLM fallback on every flush (finding 7 / INV-5).
  * A server whose vector search indexes are missing refuses to index rather than pay for
    unsearchable embeddings and pretend search works (finding DB-5 / INV-2).

These are **live** oracles against a real Memgraph: the effects (aborting before a write,
scope-scoped reconcile, index presence) only exist against real graph state. Skipped unless
``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a **throwaway** instance, so CI and a real
``bolt://localhost:7687`` are never touched. External APIs (LLM, embedder) are faked so the
failure being reproduced — a dead model, a dropped batch — is deterministic; the graph
state, every reconcile/preflight query, and the index DDL are real.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import IndexerConfig
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.registry import ParserRegistry
from inflorescence.config import Settings
from inflorescence.cost import IndexCostExceededError
from inflorescence.project_manager import (
    IndexPreflightError,
    ProjectManager,
    SearchUnavailableError,
)
from inflorescence.rag.indexer import RAGIndexer

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536
_VEC = [0.1] * _DIM


class _Embedder:
    """A deterministic embedder. ``fail_preflight`` makes the pre-flight probe raise."""

    def __init__(self, *, fail_preflight: bool = False) -> None:
        self.model = "fake-embedder"
        self._fail_preflight = fail_preflight
        self.embed_calls = 0

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        self.embed_calls += 1
        return [list(_VEC) for _ in texts]

    def preflight(self) -> None:
        if self._fail_preflight:
            raise RuntimeError("401 embedding endpoint rejected the key")


class _LLM:
    """A fake chat LLM. ``fail`` raises like a dead model; otherwise returns fallback JSON.

    ``fallback_calls`` counts only LLM *fallback-parse* invocations (whose prompt starts with
    the fallback parser's "Analyze this Python file"), separate from the tiny preflight ping,
    so a test can assert an unchanged broken file is not re-parsed.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls = 0
        self.fallback_calls = 0

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        if prompt.startswith("Analyze this Python file"):
            self.fallback_calls += 1
        if self._fail:
            raise RuntimeError("404 model not found")
        return (
            '{"nodes":[{"name":"broken","type":"module","line_start":1,"line_end":1,'
            '"signature":"","docstring":"","parent":null}],"edges":[]}'
        )


async def _conn():
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.schema import setup_schema

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    return conn


def _manager(
    conn,
    *,
    llm: _LLM | None = None,
    summary_embedder: _Embedder | None = None,
    code_embedder: _Embedder | None = None,
    use_llm_summaries: bool = False,
    use_llm_fallback_parser: bool = False,
    max_index_cost_usd: float | None = 10.0,
    require_search_indexes: bool = True,
):
    from inflorescence.db.repository import GraphRepository

    settings = Settings(
        _env_file=None,
        memgraph_url=LIVE_URL,
        max_index_cost_usd=max_index_cost_usd,
        require_search_indexes=require_search_indexes,
    )
    repo = GraphRepository(conn)
    registry = ParserRegistry()
    registry.register(PythonAstParser())
    builder = GraphBuilder(
        repo=repo,
        llm=llm,
        settings=settings,
        registry=registry,
        config=IndexerConfig(
            use_llm_summaries=use_llm_summaries,
            use_llm_fallback_parser=use_llm_fallback_parser,
        ),
    )
    rag = RAGIndexer(repo=repo, settings=settings)
    rag._summary_embedder = summary_embedder or _Embedder()  # type: ignore[assignment]
    rag._code_embedder = code_embedder or _Embedder()  # type: ignore[assignment]
    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn
    )
    return manager, repo


# ---------------------------------------------------------------------------
# Finding 6 · fail closed on ANY dead dependency, before the first write (INV-2)
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_dead_llm_model_aborts_before_any_write(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    conn = await _conn()
    # Embedders are fine; the LLM chat model is dead. Before the fix preflight only probed the
    # summary embedder, so this run would build the graph and only then fail on every summary.
    manager, repo = _manager(conn, llm=_LLM(fail=True), use_llm_summaries=True)
    project = manager.get_project(str(tmp_path))
    try:
        with pytest.raises(IndexPreflightError):
            await manager.index_directory(str(tmp_path))
        stats = await repo.get_project_stats(project)
        assert stats["nodes"] == 0, "a dead LLM model must abort BEFORE the first write"
    finally:
        await repo.delete_project(project)
        await conn.close()


@live
@pytest.mark.asyncio
async def test_broken_code_embedder_aborts_before_any_write(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    conn = await _conn()
    # The summary embedder is healthy; the *code* embedder (separate endpoint/key) is broken.
    # Before the fix only the summary side was probed, so the graph was built and summaries
    # paid for, then every chunk batch failed — a first index with no chunks at all.
    settings_with_code_endpoint = dict(
        code_embedding_model="codestral-embed", code_embedding_api_key="broken-key"
    )
    manager, repo = _manager(
        conn,
        summary_embedder=_Embedder(fail_preflight=False),
        code_embedder=_Embedder(fail_preflight=True),
    )
    # The manager's RAGIndexer must actually treat the code endpoint as distinct for the probe
    # to run; rebuild its settings to reflect a separate code endpoint.
    manager._rag._settings = Settings(_env_file=None, memgraph_url=LIVE_URL, **settings_with_code_endpoint)
    project = manager.get_project(str(tmp_path))
    try:
        with pytest.raises(IndexPreflightError):
            await manager.index_directory(str(tmp_path))
        stats = await repo.get_project_stats(project)
        assert stats["nodes"] == 0, "a broken code embedder must abort BEFORE the first write"
    finally:
        await repo.delete_project(project)
        await conn.close()


# ---------------------------------------------------------------------------
# Finding 5c · a watcher update inherits the index's scan scope (INV-6/INV-10)
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_watcher_update_inherits_exclude_scope(tmp_path: Path) -> None:
    # "thirdparty" is deliberately NOT in DEFAULT_EXCLUDE_PATTERNS, so only the persisted
    # per-index scope — not the global config — can keep the watcher from scanning it.
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    third = tmp_path / "thirdparty"
    third.mkdir()
    (third / "junk.py").write_text("def junk():\n    return 1\n", encoding="utf-8")

    conn = await _conn()
    manager, repo = _manager(conn)
    project = manager.get_project(str(tmp_path))
    try:
        await manager.index_directory(str(tmp_path), exclude_patterns=["thirdparty"])
        files_after_index = {n["file_path"] for n in await repo.list_entities(project, limit=200)}
        assert "thirdparty/junk.py" not in files_after_index, "exclude was not honored at index time"

        # A watcher event touches the excluded file. Before the fix the update rescanned with
        # only the global config excludes, so thirdparty/junk.py would be parsed, summarized and
        # embedded — silently widening scope and spending money.
        (third / "junk.py").write_text("def junk():\n    return 2  # edited\n", encoding="utf-8")
        await manager.update_project(project, [str(third / "junk.py")])

        files_after_update = {n["file_path"] for n in await repo.list_entities(project, limit=200)}
        assert "thirdparty/junk.py" not in files_after_update, "watcher widened scope past the index's exclude"
        assert "keep.py" in files_after_update
    finally:
        await repo.delete_project(project)
        await conn.close()


# ---------------------------------------------------------------------------
# Finding 5b · a watcher update obeys the cost cap (INV-10)
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_watcher_update_is_cost_gated(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    conn = await _conn()
    manager, repo = _manager(conn)
    project = manager.get_project(str(tmp_path))
    try:
        await manager.index_directory(str(tmp_path))

        # A `git checkout` lands a large changed set (many sizeable files). Before the fix a
        # watcher flush re-summarized/re-embedded them with NO cap at all. Now a low cap must
        # gate it. Each file stays under the per-file size limit, so this is a realistic flush.
        big = "# " + ("x " * 20000) + "\ndef g():\n    return 1\n"  # ~40 KB
        changed = []
        for i in range(30):
            p = tmp_path / f"big_{i:02d}.py"
            p.write_text(big, encoding="utf-8")
            changed.append(str(p))
        manager._settings.max_index_cost_usd = 0.001  # ~$0.006 estimated for the batch

        with pytest.raises(IndexCostExceededError):
            await manager.update_project(project, changed)
    finally:
        await repo.delete_project(project)
        await conn.close()


# ---------------------------------------------------------------------------
# Finding 7 · an unchanged broken .py is not re-paid on every flush (INV-5)
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_unchanged_broken_file_is_not_repaid_on_the_next_flush(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")  # AST fails -> LLM fallback
    conn = await _conn()
    llm = _LLM()
    manager, repo = _manager(conn, llm=llm, use_llm_fallback_parser=True)
    project = manager.get_project(str(tmp_path))
    try:
        await manager.index_directory(str(tmp_path))
        assert llm.fallback_calls == 1, "the broken file should take the LLM fallback exactly once"

        # An unrelated file changes; the update re-parses ALL files (including broken.py). Before
        # the fix the unchanged broken.py hit the paid LLM fallback again on this flush.
        (tmp_path / "ok.py").write_text("def ok():\n    return 2  # edited\n", encoding="utf-8")
        await manager.update_project(project, [str(tmp_path / "ok.py")])
        assert llm.fallback_calls == 1, "unchanged broken file was re-paid to the LLM on the next flush"
    finally:
        await repo.delete_project(project)
        await conn.close()


# ---------------------------------------------------------------------------
# Finding DB-5 · refuse to index when the vector search circuit is missing (INV-2)
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_indexing_refuses_when_vector_search_indexes_are_missing(tmp_path: Path) -> None:
    from inflorescence.db.schema import missing_search_indexes

    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    conn = await _conn()
    manager, repo = _manager(conn)
    project = manager.get_project(str(tmp_path))
    try:
        # Simulate a plain Memgraph (no MAGE): the vector indexes don't exist.
        await conn.execute_write("DROP VECTOR INDEX code_embeddings;")
        await conn.execute_write("DROP VECTOR INDEX summary_embeddings;")
        assert set(await missing_search_indexes(conn)) == {"code_embeddings", "summary_embeddings"}

        with pytest.raises(SearchUnavailableError):
            await manager.index_directory(str(tmp_path))
        assert (await repo.get_project_stats(project))["nodes"] == 0, "indexed on a search-less DB"

        # Recreate the circuit; indexing proceeds again (search now real).
        from inflorescence.db.schema import setup_schema

        assert await setup_schema(conn) == []
        result = await manager.index_directory(str(tmp_path))
        assert result["reused"] is False and result["nodes"] > 0
    finally:
        await repo.delete_project(project)
        await conn.close()
