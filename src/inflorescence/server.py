"""FastMCP server: wires up all components and registers MCP tools."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from inflorescence import janitor
from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import IndexerConfig
from inflorescence.config import Settings
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import setup_schema
from inflorescence.feedback import FeedbackStore
from inflorescence.llm_client import LLMClient
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.indexer import RAGIndexer
from inflorescence.tools.cypher import register_cypher_tools
from inflorescence.tools.feedback import register_feedback_tools
from inflorescence.tools.graph import register_graph_tools
from inflorescence.tools.indexing import register_indexing_tools
from inflorescence.tools.search import register_search_tools
from inflorescence.watcher import FileWatcher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Components:
    """Everything a process needs to read or mutate an index, built once.

    Two entry points assemble the same stack for different jobs — the MCP server
    (:func:`create_server`) and the watch daemon
    (:class:`inflorescence.watch_daemon.WatchDaemon`) — and a third-party embedder assembles
    it directly. Sharing one factory is what keeps them from drifting apart: a dependency
    added for one is a dependency the other gets too, rather than a bug that only reproduces
    under the daemon.
    """

    settings: Settings
    conn: MemgraphConnection
    repo: GraphRepository
    llm: LLMClient
    builder: GraphBuilder
    rag: RAGIndexer
    manager: ProjectManager
    watcher: FileWatcher
    feedback: FeedbackStore


def create_components() -> Components:
    """Build the connection, repository, LLM client, indexer, manager and watcher."""
    settings = Settings()
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
    watcher = FileWatcher(
        manager, settings, extensions=builder.extensions, filenames=builder.indexable_filenames
    )
    feedback = FeedbackStore(settings.feedback_path)
    return Components(
        settings=settings,
        conn=conn,
        repo=repo,
        llm=llm,
        builder=builder,
        rag=rag_indexer,
        manager=manager,
        watcher=watcher,
        feedback=feedback,
    )


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
    components = create_components()
    settings = components.settings

    # Server-level instructions are injected into consuming agents' context — this is
    # the only place an agent learns what the server is FOR before picking a tool.
    mcp = FastMCP(
        "inflorescence",
        instructions=(
            "Code intelligence over indexed repositories: a code graph in Memgraph "
            "(functions/classes/modules/directories with CALLS / IMPORTS / INHERITS / "
            "IMPLEMENTS / CONTAINS edges), LLM-written summaries for every entity, and "
            "embeddings for search.\n\n"
            "The index is not only code. Non-code files are indexed whole as `document` "
            "entities — prose (.md, .markdown, .mdx, .rst, .txt, .text, .adoc, .asciidoc, "
            ".org), stylesheets (.css, .scss, .sass, .less, .styl), markup (.html, .htm), "
            "configuration (.json, .jsonc, .yaml, .yml, .toml, .ini, .cfg, .conf), shell "
            "and operator scripts (.sh, .bash, .zsh, .fish, .ps1) and the build files named "
            "rather than suffixed (Makefile, GNUmakefile, Dockerfile, Containerfile, "
            "Justfile, Procfile, Rakefile, Brewfile, .env.example/.sample/.template/.dist) — "
            "with their body split into overlapping windows, so a CSS media query, an "
            "index.html meta tag, a build-config key or the exact flags a wrapper script "
            "passes to a runner is findable by its own text (search_text) and by meaning "
            "(search_semantic). Shell scripts matter more than their size suggests: the "
            "operational recipe — which env vars, which flags, and the comment saying why — "
            "usually exists only there. They are not parsed for structure, so they own no "
            "symbols and no CALLS edges of their own — but code "
            "that imports one by path (`import './index.css'`, `import cfg from "
            "'./config.json'`) does give them an incoming IMPORTS edge, so "
            "get_related_entities(direction=\"incoming\") answers \"who pulls in this "
            "stylesheet\". One consequence to keep in mind: a module invoked only from a "
            "shell wrapper (`python -m scripts.runner`) has NO incoming CALLS edge, so "
            "operator-facing entry points look callerless — never read an empty incoming "
            "CALLS list as dead code without a search_text for the module name. Filter "
            "documents with node_types=[\"document\"] or exclude them "
            "the same way. Dependency lock files (package-lock.json, pnpm-lock.yaml, "
            "go.sum, …) and a real .env are never indexed.\n\n"
            "Use this server instead of grepping the filesystem when the question is about "
            "code structure, relationships, or purpose: find code by meaning "
            "(search_semantic), by exact identifier / string literal / error text "
            "(search_text), by name (search_entities), or by code-vector similarity "
            "(search_code, search_hybrid). Deep-dive a found entity with "
            "get_entity_context / get_entity_structure, trace callers/callees and imports "
            "with get_related_entities (which takes either a node_id or a name= substring, "
            "so \"who calls X\" is one call), and answer arbitrary structural questions with "
            "a read-only cypher_query (call graph_schema first).\n\n"
            "WHEN YOU DO NOT KNOW WHERE SOMETHING LIVES, WALK DOWN THE SUMMARIES instead of "
            "searching blind or listing everything. Every directory has an LLM summary of what "
            "its subtree is for — purpose, capabilities, entry points — written from the "
            "summaries of what it contains, so a few cheap calls narrow a repository to one "
            "file: get_entity_structure(directory) shows the root's summary and its top-level "
            "files and directories; get_entity_structure(directory, file_path=\"src/pkg\") "
            "does the same one level down (file_path takes a directory as well as a file); "
            "get_entity_structure(directory, file_path=\"src/pkg/providers/openai.py\") lists "
            "that file's classes and functions with line ranges and one-line summaries — then "
            "read only those lines. Each step is one small page, where "
            "list_entities(node_types=[\"module\"]) over a big package is one page of every "
            "file at once (a field report measured ~17k tokens for 56 modules). When the tree "
            "is too wide to read, jump by meaning instead: search_semantic(query, "
            "node_types=[\"directory\"]) ranks directories against directories (\"which "
            "part of this repo handles X\"), node_types=[\"module\"] ranks whole files; then "
            "scope the fine-grained search to that subtree with include_paths=[\"<path>\"].\n\n"
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
            "An index is a snapshot, and nothing about indexing keeps it matching the code. "
            "Only a watch daemon does — a process outside this session that follows "
            "registered projects, reconciles each one against disk when it starts, and "
            "re-indexes what is saved (`inflorescence watch add <path>`, then `watch run`, "
            "or `watch install` on macOS to run it at login). Without one, an index is "
            "exactly as fresh as the last index_directory call: the watcher that call "
            "leaves behind lives in this server process and dies with this session. Every "
            "response that carries entities says which of those is true in "
            "meta.index_freshness.watcher, and index_directory says it in `watching`. "
            "Treat a page whose files all hash fresh but whose watcher is `none` as a "
            "snapshot that is drifting: the flags only ever checked the handful of files "
            "that answer returned.\n\n"
            "Summaries and embeddings are English-only, so phrase semantic queries "
            "(search_semantic, search_code, search_hybrid) in English — a query in another "
            "language embeds far from that space and returns a plausible-looking wrong "
            "answer, not an error. Non-English text and exact strings (identifiers, "
            "literals, error messages) belong in search_text. Pagination is honest about "
            "its own limits: a page with page.truncated=true means the scan stopped early "
            "at an internal ceiling, so an empty or short page there is not evidence that "
            "nothing matches — narrow the filters or keep paging. It is equally honest "
            "about age: every response carrying entities names the tree it answered from "
            "in meta.answered_from (path, branch @ short commit, and the checkout a git "
            "worktree belongs to) — plus meta.nested_checkouts when the files on the page "
            "belong to an independent checkout inside that tree (a submodule or vendored "
            "repository on its own branch), whose HEAD, not answered_from, is the code you "
            "are reading — re-hashes their files and reports "
            "meta.index_freshness, flagging an item whose file changed after indexing "
            "(\"stale\": true) or is gone from disk (\"file_missing\": true), so an "
            "unflagged item was checked and still matches the index. That block also "
            "carries live_worktrees, a sentence saying how many git worktrees of this "
            "checkout hold branch state that is not in its index at all, however fresh the "
            "index looks — ask such a worktree directly by passing its path as directory. "
            "Vector searches are honest about distance too: a hit that matched only weakly "
            "carries \"weak\": true, and meta.weak_results=true means every hit on the page "
            "was weak — that is the closest thing to \"this code does not exist here\" a "
            "vector search can say, so confirm with search_text on exact terms rather than "
            "trusting the top hit.\n\n"
            "EXCLUDE TEST CODE WHEN YOU ARE LOOKING FOR AN IMPLEMENTATION — this is the "
            "single most effective thing you can do to the ranking of a semantic search. A "
            "test's summary restates the vocabulary of the code it exercises (\"Verifies "
            "that retrying a failed upload reuses the original storage key\"), so tests "
            "out-embed the implementation on exactly the "
            "\"what does X do\" queries you use to find it: a field report measured 11 of 15 "
            "hits from tests/ while the runner named in the query never reached page 1. "
            "Hits from test-like paths carry \"is_test\": true (a path heuristic over "
            "tests/, spec/, test_*.py, *_test.go, conftest.py, *.test.ts …). Pass "
            "include_tests=false to drop them: "
            "search_semantic(directory=\"/repo\", query=\"job runner start stop\", "
            "include_tests=false). For a layout the heuristic does not catch, name the "
            "directories: exclude_paths=[\"tests/**\", \"fixtures/**\"]. Search the tests "
            "deliberately (include_tests=true, the default) when the test IS what you want — "
            "usage examples, expected behaviour, a repro.\n\n"
            "AN EMPTY RESULT IS NOT PROOF OF ABSENCE, and the response now says which it is: "
            "an empty first page carries meta.index_coverage, which re-checks the "
            "directories this index knows against the filesystem and either names files that "
            "are on disk but NOT indexed (a coverage gap — read them directly, and "
            "index_directory to make them queryable) or states that nothing under that scope "
            "is missing, which is what makes \"it does not exist\" a safe conclusion. Index "
            "coverage is not staleness: a file created after the last index run has no node "
            "at all, so index_freshness cannot flag it and a live watcher line does not cover "
            "it either. Likewise get_entity_structure(file_path=…) distinguishes a path that "
            "exists on disk but was never indexed from one that does not exist.\n\n"
            "FINALLY, AND PLEASE DO NOT SKIP THIS: send_feedback is how this server gets "
            "better, and you are the only one who can see it from the inside. When you have "
            "finished working with these tools — after a search, an index, or a whole task "
            "— call send_feedback(kind=\"experience_report\") and say what using them was "
            "actually like: which tool description sent you somewhere wrong, what you "
            "expected a result to contain, how many calls a question took, what you fell "
            "back to grep for, and what worked well enough that it should not be changed. "
            "Report anything broken or surprising the moment you hit it, without waiting "
            "for the end of the task (kind=\"bug\", \"wrong_result\", "
            "\"missing_capability\", \"confusing_docs\", \"performance\"). It is free, "
            "needs no LLM, no index and not even a reachable database — feedback about a "
            "database that is down is exactly the feedback worth having — and it is "
            "appended to a local file on the user's machine, never uploaded anywhere. "
            "Concrete beats polite: name the call, the result you got and the result you "
            "expected."
        ),
    )

    conn, repo, manager, watcher = (
        components.conn,
        components.repo,
        components.manager,
        components.watcher,
    )

    # Register all tools
    register_indexing_tools(mcp, manager, watcher if enable_watcher else None, repo)
    register_graph_tools(mcp, manager, repo)
    register_search_tools(mcp, manager, repo, settings)
    register_cypher_tools(mcp, manager, conn)
    register_feedback_tools(mcp, components.feedback)

    logger.info("Inflorescence MCP server configured with MCP tools")
    return mcp, conn, components.llm, watcher


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
        await setup_schema(conn, dimension=(settings or Settings()).embedding_dimension)
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
