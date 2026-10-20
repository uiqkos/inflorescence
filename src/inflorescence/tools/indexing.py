"""Indexing MCP tool."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from inflorescence.config import LOCKFILE_NAMES
from inflorescence.cost import IndexCostExceededError
from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import IndexPreflightError, ProjectManager
from inflorescence.tools import descriptions as doc
from inflorescence.tools.responses import error_response
from inflorescence.watcher import FileWatcher


async def _start_watching(
    repo: GraphRepository | None,
    watcher: FileWatcher | None,
    project: str,
    path: str,
) -> str:
    """Decide who follows this tree from now on, and say so in one sentence.

    A finished index used to leave a watcher behind unconditionally, inside whichever process
    served the call — for an MCP server, a stdio child of one session, which dies with it. The
    caller was never told any of that, so "indexed" read as "kept current from now on" and
    stayed wrong from the moment the window closed.

    Now the answer depends on who is already watching. A live watch daemon owns the tree, and
    a second watcher in this process would only duplicate every flush (they serialize on the
    project lease, so the cost is wasted scans, not corruption) — so we step aside. Otherwise
    the session watcher still starts, because a watcher for the length of this session beats
    none, but the sentence says exactly how long that lasts and what makes it permanent.
    """
    state = await repo.get_watch_state(project) if repo is not None else {}
    if state.get("live"):
        return "a watch daemon is following this tree — it stays current after this session ends"
    if watcher is None:
        return "nothing is watching this tree; `inflorescence watch add <path>` follows it permanently"
    watcher.start(project, path)
    if state.get("watched"):
        return (
            "this session only — the tree is registered for the watch daemon, but no daemon is "
            "running; start it with `inflorescence watch run`"
        )
    return (
        "this session only — the watcher dies with it; `inflorescence watch add "
        f"{path}` registers the tree with the watch daemon, which does not"
    )


# Composed rather than written as the docstring: the admitted document types are listed from
# config (doc.DOCUMENT_TYPES, LOCKFILE_NAMES), so a type added there is announced here without an edit.
INDEX_DIRECTORY_DESCRIPTION = """Index or re-index a code directory (parses, builds a code graph, summarizes, embeds).

RECOMMENDED FLOW: first call with preview=true to see how many files/bytes and roughly
how many nodes/tokens would be processed and the estimated dollar cost (no DB writes, no
LLM calls). Review the estimate, then call again with preview=false to actually index.
Re-indexing only re-summarizes and re-embeds the files/nodes that actually changed. The
preview's ``largest_files`` names the biggest files it priced, each with its own
``estimated_cost_usd`` — when a few generated tables, fixtures or specs dominate the quote,
add them to ``.inflorescenceignore`` (or pass ``exclude_patterns``) and preview again.

WHAT IS INDEXED: source files for the supported languages (Python, JS/TS, Go, Rust),
plus non-code files stored whole as ``document`` entities — by default DOCUMENT_TYPES.
Documents are chunked and summarized but never parsed for
structure. Always skipped: dependency lock files by name (LOCKFILE_NAMES), paths matched by a project-local
``.inflorescenceignore`` (gitignore syntax, honored in any directory and regardless of
RESPECT_GITIGNORE — the durable way a repo keeps a tree out of the index without
hiding it from git), .gitignore'd paths, files above
MAX_FILE_SIZE_BYTES (reported as ``skipped_large_files`` in both the preview and the
index result), and generated output (*.min.js, *.min.css, protobuf stubs). Set
INDEX_DOCUMENTS=false, or narrow DOCUMENT_EXTENSIONS / DOCUMENT_FILENAMES, to index
less than that — which also lowers what preview=true quotes.

The index is persisted incrementally: the full graph skeleton (entities + edges,
searchable by name) lands right after parsing, and summaries stream into the DB in
batches as they are generated. A run that is killed mid-summarization loses nothing —
call index_directory again and it resumes, reusing every stored summary by content
hash at zero additional LLM cost. While a run is in progress, summary-based and
chunk-based search cover only what has already landed (see the search tools' notes).

GIT WORKTREES ARE NEARLY FREE AND FAST, AND USUALLY NEED NO CALL AT ALL. The first
index of a worktree whose main checkout is already indexed reuses that checkout's
summaries and embeddings by content hash, so LLM spend is ~$0 — only files that differ
from the donor are paid for. The reuse happens as a copy *inside the database*, so it
also finishes in a fraction of the time a normal first index takes; the response
reports ``worktree_of`` and ``cloned`` (the copied node/chunk/edge counts). You
normally never call this for a worktree: the read tools
(search_*, list/get_*, cypher_query) index it themselves on the first query.
preview=true on such a worktree reports ``worktree_of`` (the donor project) plus
``donor_covered_files`` / ``uncovered_files``, and prices only the uncovered files,
so the estimate is small on purpose rather than by mistake.

Args:
    path: Absolute path to the directory to index.
    include_patterns: Optional relative paths/globs to include. If omitted, include all supported files.
    exclude_patterns: Optional relative paths, directory names, or globs to exclude. This
        keeps those paths out of THIS scan; it does not delete what an earlier, wider
        index already stored for them (an excluded file is still on disk, so a re-index
        cannot tell a narrowed scope from a failed scan and never deletes on that guess).
        Removing already-indexed paths is a separate, free operation the user runs:
        ``inflorescence prune <path> --exclude <pattern>`` (add ``--dry-run`` to see it
        first). Pass the same exclusions here on every index of this project, or the next
        run re-admits them.
    preview: If true, return only a dry-run estimate and perform no writes or LLM calls.
    max_cost_usd: If set, abort an index whose estimated cost exceeds this many dollars
        (returns an ``index_cost_exceeded`` error), overriding MAX_INDEX_COST_USD for this
        call. A first-time index is priced over the whole tree; a re-index of an already
        indexed project is priced over only the files whose content changed since the last
        run — which includes every file a widened extension set newly admits. Every run is
        then priced again after its parse, before any write or LLM request, over the exact
        summaries it will regenerate (entities an upgraded parser newly recognizes in
        unchanged files included), counted on the tree as it stands then — a file saved
        while the run is still summarizing can add the entities it touches; the error says
        how many it counted.
    force_rebuild: Re-parse and re-resolve the whole project even when no file changed —
        needed to pick up indexer upgrades (e.g. new CALLS resolution / node properties)
        on an unchanged project, which the checksum-based incremental path would skip.
        Summaries and embeddings are reused by content hash; what it pays for is the
        summaries of entities the upgrade newly recognizes or reads differently, and of the
        containers above them — priced and capped like any run. NOT a data wipe.

STAYING CURRENT AFTERWARDS. Indexing is a snapshot; nothing about it keeps the graph
matching the code. Every response says who follows the tree from here in ``watching``,
and there are only two answers that survive this conversation. A **watch daemon**
(``inflorescence watch add <path>``, then ``inflorescence watch run``, or
``inflorescence watch install`` for launchd) follows registered projects from its own
process: it reconciles each one against disk when it starts and re-indexes what is
saved, so an index stays current across sessions and reboots. Failing that, this call
leaves a watcher inside the current server process, which dies when the session does —
useful now, worthless tomorrow. Every read tool reports the same state in
``meta.index_freshness.watcher``, so an agent can tell a fresh-looking answer from a
genuinely maintained one.

Returns:
    Indexing statistics — ``nodes``/``edges``/``chunks``/``summaries`` count what this
    run wrote (an incremental run writes only what changed); ``db`` carries the project
    totals; ``watching`` names whatever will keep this index current from now on. With
    preview=true, a dry-run estimate instead.
""".replace("DOCUMENT_TYPES", doc.DOCUMENT_TYPES).replace(
    "(LOCKFILE_NAMES)", f"({', '.join(sorted(LOCKFILE_NAMES))})"
)


def register_indexing_tools(
    mcp: FastMCP,
    manager: ProjectManager,
    watcher: FileWatcher | None = None,
    repo: GraphRepository | None = None,
) -> None:
    """Register indexing tools with the MCP server."""

    @mcp.tool(description=INDEX_DIRECTORY_DESCRIPTION)
    async def index_directory(
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        preview: bool = False,
        max_cost_usd: float | None = None,
        force_rebuild: bool = False,
    ) -> dict:
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
        result["watching"] = await _start_watching(repo, watcher, str(result["project"]), path)
        return result
