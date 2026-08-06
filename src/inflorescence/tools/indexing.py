"""Indexing MCP tool."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from inflorescence.cost import IndexCostExceededError
from inflorescence.project_manager import IndexPreflightError, ProjectManager
from inflorescence.tools.responses import error_response
from inflorescence.watcher import FileWatcher


def register_indexing_tools(mcp: FastMCP, manager: ProjectManager, watcher: FileWatcher | None = None) -> None:
    """Register indexing tools with the MCP server."""

    @mcp.tool()
    async def index_directory(
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        preview: bool = False,
        max_cost_usd: float | None = None,
        force_rebuild: bool = False,
    ) -> dict:
        """Index or re-index a code directory (parses, builds a code graph, summarizes, embeds).

        RECOMMENDED FLOW: first call with preview=true to see how many files/bytes and roughly
        how many nodes/tokens would be processed and the estimated dollar cost (no DB writes, no
        LLM calls). Review the estimate, then call again with preview=false to actually index.
        Re-indexing only re-summarizes and re-embeds the files/nodes that actually changed.

        The index is persisted incrementally: the full graph skeleton (entities + edges,
        searchable by name) lands right after parsing, and summaries stream into the DB in
        batches as they are generated. A run that is killed mid-summarization loses nothing —
        call index_directory again and it resumes, reusing every stored summary by content
        hash at zero additional LLM cost. While a run is in progress, summary-based and
        chunk-based search cover only what has already landed (see the search tools' notes).

        Args:
            path: Absolute path to the directory to index.
            include_patterns: Optional relative paths/globs to include. If omitted, include all supported files.
            exclude_patterns: Optional relative paths, directory names, or globs to exclude.
            preview: If true, return only a dry-run estimate and perform no writes or LLM calls.
            max_cost_usd: If set, abort a first-time index whose estimated cost exceeds this many
                dollars (returns an ``index_cost_exceeded`` error). Incremental re-indexes are not gated.
            force_rebuild: Re-parse and re-resolve the whole project even when no file changed —
                needed to pick up indexer upgrades (e.g. new CALLS resolution / node properties)
                on an unchanged project, which the checksum-based incremental path would skip.
                Summaries and embeddings are still reused; no LLM re-spend. NOT a data wipe.

        Returns:
            Indexing statistics (nodes, edges, chunks, ...), or a dry-run estimate when preview=true.
        """
        if preview:
            return await manager.preview_directory(
                path, include_patterns=include_patterns, exclude_patterns=exclude_patterns
            )
        try:
            result = await manager.index_directory(
                path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_cost_usd=max_cost_usd,
                force_rebuild=force_rebuild,
            )
        except IndexCostExceededError as exc:
            return error_response("index_cost_exceeded", str(exc))
        except IndexPreflightError as exc:
            # A keyless/misconfigured server must refuse to index (and must not start
            # a watcher) instead of silently degrading the stored graph.
            return error_response("llm_unavailable", str(exc))
        if watcher is not None:
            watcher.start(str(result["project"]), path)
        return result
