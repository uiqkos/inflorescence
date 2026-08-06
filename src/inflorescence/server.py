"""FastMCP server: wires up all components and registers MCP tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

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
            "A project must be indexed before anything else works: call "
            "index_directory(path, preview=true) to see the estimated dollar cost, then "
            "preview=false to build. Indexing persists incrementally — entities are "
            "name-searchable right after parsing, summaries stream in batch by batch, and "
            "an interrupted run resumes with no repeated LLM spend. Costs real money "
            "(LLM summarization): never index without a preview-approved estimate."
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


async def startup(conn: MemgraphConnection, *, allow_degraded: bool = False) -> bool:
    """Verify Memgraph connectivity and create schema. Returns whether the DB is healthy."""
    healthy = await conn.health_check()
    if healthy:
        logger.info("Memgraph connection OK")
        await setup_schema(conn)
    elif allow_degraded:
        logger.warning("Memgraph is not reachable — starting in degraded mode (--allow-degraded)")
    return healthy


async def shutdown(conn: MemgraphConnection, llm: LLMClient, watcher: FileWatcher) -> None:
    """Run shutdown tasks: close connections, stop watchers."""
    watcher.stop_all()
    await llm.close()
    await conn.close()
    logger.info("Server shutdown complete")
