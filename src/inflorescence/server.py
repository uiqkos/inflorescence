"""FastMCP server: wires up all components and registers MCP tools."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from mcp.server.fastmcp import FastMCP

from inflorescence import janitor
from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import IndexerConfig
from inflorescence.config import Settings
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import setup_schema
from inflorescence.llm_client import LLMClient
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.indexer import RAGIndexer
from inflorescence.tools.cypher import register_cypher_tools
from inflorescence.tools.graph import register_graph_tools
from inflorescence.tools.indexing import register_indexing_tools
from inflorescence.tools.search import register_search_tools
from inflorescence.watcher import FileWatcher

logger = logging.getLogger(__name__)

# How often the long-running server re-checks for worktree graphs to reap. A worktree is
# deleted by a human, so nothing is gained by sweeping more often than a coffee break; the
# sweep itself is a handful of reads plus a few `stat` calls.
WORKTREE_JANITOR_INTERVAL_S = 30 * 60.0

# The periodic sweep tasks this process started, cancelled by :func:`shutdown`. A module-level
# set rather than a return value because ``startup``/``shutdown`` are called by an entry point
# whose signature is fixed; in practice a process runs one server, so this holds 0 or 1 task.
_janitor_tasks: set[asyncio.Task[None]] = set()


def create_server(*, enable_watcher: bool = True) -> tuple[FastMCP, MemgraphConnection, LLMClient, FileWatcher]:
    """Create and configure the MCP server with all components.

    ``enable_watcher=False`` builds the same tools but never hands ``index_directory``
    a watcher, so a completed index leaves no background re-indexer behind. The
    dashboard's tools console uses this: it is a viewer, and a watcher started there
    would keep spending on summaries for every file the user saves, from a process
    nobody asked to watch anything.
    """
    settings = Settings()

    # Server-level instructions are injected into consuming agents' context — this is
    # the only place an agent learns what the server is FOR before picking a tool.
    mcp = FastMCP(
        "inflorescence",
        instructions=(
            "Code intelligence over indexed repositories: a code graph in Memgraph "
            "(functions/classes/modules/directories with CALLS / IMPORTS / INHERITS / "
            "IMPLEMENTS / CONTAINS edges), LLM-written summaries for every entity, and "
            "embeddings for search.\n\n"
            "Use this server instead of grepping the filesystem when the question is about "
            "code structure, relationships, or purpose: find code by meaning "
            "(search_semantic), by exact identifier / string literal / error text "
            "(search_text), by name (search_entities), or by code-vector similarity "
            "(search_code, search_hybrid). Deep-dive a found entity with "
            "get_entity_context / get_entity_structure, trace callers/callees and imports "
            "with get_related_entities, and answer arbitrary structural questions with a "
            "read-only cypher_query (call graph_schema first).\n\n"
            "An ordinary project must be indexed before anything else works: call "
            "index_directory(path, preview=true) to see the estimated dollar cost, then "
            "preview=false to build. A git worktree of an already-indexed checkout is the "
            "exception — the first read query indexes it inline, reusing the main "
            "checkout's summaries and embeddings by content hash (the graph is copied "
            "inside the database, so this is quick as well as nearly free), which means it "
            "needs no explicit index_directory call. Indexing persists "
            "incrementally — entities are name-searchable right after parsing, summaries "
            "stream in batch by batch, and an interrupted run resumes with no repeated LLM "
            "spend. Costs real money (LLM summarization): never index a new project "
            "without a preview-approved estimate.\n\n"
            "Summaries and embeddings are English-only, so phrase semantic queries "
            "(search_semantic, search_code, search_hybrid) in English — a query in another "
            "language embeds far from that space and returns a plausible-looking wrong "
            "answer, not an error. Non-English text and exact strings (identifiers, "
            "literals, error messages) belong in search_text. Pagination is honest about "
            "its own limits: a page with page.truncated=true means the scan stopped early "
            "at an internal ceiling, so an empty or short page there is not evidence that "
            "nothing matches — narrow the filters or keep paging. It is equally honest "
            "about age: every response carrying entities re-hashes their files and reports "
            "meta.index_freshness, flagging an item whose file changed after indexing "
            "(\"stale\": true) or is gone from disk (\"file_missing\": true), so an "
            "unflagged item was checked and still matches the index. That block also "
            "reports live_worktrees: a checkout with live git worktrees has branch state "
            "that is not in its index at all, however fresh the index looks — ask the "
            "worktree directly by passing its path as directory. Search hits from "
            "test-like paths are marked \"is_test\": true (a path heuristic) and can be "
            "dropped with include_tests=false."
        ),
    )

    conn = MemgraphConnection(settings)
    repo = GraphRepository(conn, batch_size=settings.db_write_batch_size)
    llm = LLMClient(settings)
    builder = GraphBuilder(
        repo=repo,
        llm=llm,
        settings=settings,
        config=IndexerConfig.from_settings(settings),
    )
    rag_indexer = RAGIndexer(repo=repo, settings=settings)
    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag_indexer, settings=settings, conn=conn
    )
    watcher = FileWatcher(manager, settings, extensions=builder.extensions)

    # Register all tools
    register_indexing_tools(mcp, manager, watcher if enable_watcher else None)
    register_graph_tools(mcp, manager, repo)
    register_search_tools(mcp, manager, repo, settings)
    register_cypher_tools(mcp, manager, conn)

    logger.info("Inflorescence MCP server configured with MCP tools")
    return mcp, conn, llm, watcher


async def startup(
    conn: MemgraphConnection,
    *,
    allow_degraded: bool = False,
    repo: GraphRepository | None = None,
    settings: Settings | None = None,
    enable_janitor: bool = True,
) -> bool:
    """Verify Memgraph connectivity, create schema, start the worktree janitor.

    Returns whether the DB is healthy. The janitor lives here rather than in
    :func:`create_server` on purpose: ``create_server`` is also how the dashboard builds a
    read-only tools console, and that console never calls ``startup``. A viewer must not be
    able to delete anything, so the deleting code path is not reachable from the object it
    constructs. ``enable_janitor=False`` is the explicit off switch for an embedder that
    calls ``startup`` and still wants no deletions; ``settings.worktree_janitor_enabled`` is
    the user-facing one.
    """
    healthy = await conn.health_check()
    if healthy:
        logger.info("Memgraph connection OK")
        await setup_schema(conn)
        if enable_janitor:
            await start_worktree_janitor(conn, repo=repo, settings=settings)
    elif allow_degraded:
        logger.warning("Memgraph is not reachable — starting in degraded mode (--allow-degraded)")
    return healthy


async def start_worktree_janitor(
    conn: MemgraphConnection,
    *,
    repo: GraphRepository | None = None,
    settings: Settings | None = None,
) -> asyncio.Task[None] | None:
    """Sweep once now, then schedule periodic sweeps. Returns the periodic task, if any.

    Failures here are never fatal: the janitor is housekeeping, and a server that cannot
    start it must still serve queries. The first sweep runs inline so a restart reaps
    promptly (a worktree deleted while the server was down is the common case).

    The probe read before it is not redundant. :func:`janitor.sweep` deliberately swallows
    everything — it is a background chore — so a sweep returning is no proof that the
    repository works. The probe is: if the janitor's only read cannot be issued at all, the
    periodic task is not scheduled, because whatever broke it would break every later sweep
    too and a loop logging the same failure every half hour helps nobody.
    """
    try:
        settings = settings or Settings()
        if not settings.worktree_janitor_enabled:
            logger.info("Worktree janitor disabled by configuration — not scheduling sweeps")
            return None
        repo = repo or GraphRepository(conn, batch_size=settings.db_write_batch_size)
        await repo.list_projects()
        await janitor.sweep(repo, settings, conn=conn)
    except Exception:  # noqa: BLE001 — housekeeping must never block server startup
        logger.warning("Worktree janitor could not start — continuing without it", exc_info=True)
        return None

    task = asyncio.create_task(_janitor_loop(repo, settings, conn), name="worktree-janitor")
    _janitor_tasks.add(task)
    task.add_done_callback(_janitor_tasks.discard)
    return task


async def _janitor_loop(
    repo: GraphRepository, settings: Settings, conn: MemgraphConnection
) -> None:
    """Sweep every :data:`WORKTREE_JANITOR_INTERVAL_S` until cancelled at shutdown."""
    while True:
        await asyncio.sleep(WORKTREE_JANITOR_INTERVAL_S)
        try:
            await janitor.sweep(repo, settings, conn=conn)
        except Exception:  # noqa: BLE001 — a failed sweep must not end the loop
            logger.warning("Periodic worktree sweep failed", exc_info=True)


async def stop_worktree_janitor() -> None:
    """Cancel every periodic sweep this process started and wait for it to unwind."""
    tasks = list(_janitor_tasks)
    _janitor_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def shutdown(conn: MemgraphConnection, llm: LLMClient, watcher: FileWatcher) -> None:
    """Run shutdown tasks: stop the janitor, close connections, stop watchers."""
    watcher.stop_all()
    # Before the connection closes: a sweep mid-flight would otherwise fail on a dead driver.
    await stop_worktree_janitor()
    await llm.close()
    await conn.close()
    logger.info("Server shutdown complete")
