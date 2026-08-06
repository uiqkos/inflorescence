"""Multi-project management: index, rebuild, and track indexed directories."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.config import Settings
from inflorescence.cost import IndexCostExceededError, estimate_index_cost
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.lease import project_lease
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import missing_search_indexes
from inflorescence.progress import ProgressReporter
from inflorescence.rag.indexer import RAGIndexer

logger = logging.getLogger(__name__)


class IndexPreflightError(RuntimeError):
    """A dependency the run needs (embeddings, LLM, or search circuit) is unavailable.

    Indexing must not proceed — INV-2 requires aborting before the first write rather than
    discovering the failure on the 800th node.
    """


class SearchUnavailableError(IndexPreflightError):
    """The vector search indexes don't exist — indexing would pay for unsearchable embeddings.

    A subclass of :class:`IndexPreflightError` so existing "refuse to index" handlers catch
    it, while callers that care can distinguish a broken search circuit from bad credentials.
    """


def _project_name(path: Path) -> str:
    """Derive a stable project identifier from an absolute path."""
    resolved = str(path.resolve())
    short_hash = hashlib.md5(resolved.encode()).hexdigest()[:8]  # noqa: S324
    return f"{path.name}_{short_hash}"


class ProjectManager:
    """Orchestrates full and change-triggered project indexing."""

    def __init__(
        self,
        repo: GraphRepository,
        graph_builder: GraphBuilder,
        rag_indexer: RAGIndexer,
        settings: Settings,
        conn: MemgraphConnection | None = None,
    ) -> None:
        self._repo = repo
        self._builder = graph_builder
        self._rag = rag_indexer
        self._settings = settings
        # Connection used only for the cross-process project lease. Real entry points
        # (server, CLI) always pass one; pure in-process unit tests may omit it, in which
        # case the per-project asyncio.Lock still serializes within the process.
        self._conn = conn
        # path -> project name mapping
        self._projects: dict[str, str] = {}
        # Two layers serialize a mutation of one project (INV-4, "by construction"):
        #   1. an in-process asyncio.Lock — cheap mutual exclusion between tasks here
        #      (watcher flush racing a manual re-index, or two flushes back to back);
        #   2. a cross-process advisory lease in Memgraph (see db/lease.py) — the only
        #      layer that also stops a *second process* (CLI index while the server's
        #      watcher flushes) from interleaving reconcile phases and destroying writes.
        # Under both, a duplicate mutation degrades to a cheap checksum no-op, never a race.
        self._project_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _preflight(self) -> None:
        """Fail closed before any graph mutation: verify every dependency the run needs.

        A server launched without a working key used to run the whole pipeline anyway: every
        call failed, summaries were overwritten with fallbacks, and embedding stores wrote
        nothing. The check now covers *all three* paid dependencies — both embedding endpoints
        (summary and code) and the LLM chat model — so a dead LLM_MODEL or a broken code-embed
        key aborts up front instead of on the 800th node (audit finding 6 / INV-2). Finally, unless
        disabled, it refuses to index when the vector search circuit is missing, so embeddings
        aren't paid for on a DB that can never search them (audit finding DB-5).
        """
        try:
            self._rag.preflight()
            await self._builder.preflight()
        except Exception as exc:
            raise IndexPreflightError(
                "Dependency pre-flight failed — aborting before touching the index. Verify "
                "OPENROUTER_API_KEY, LLM_MODEL, and any separate code-embedding endpoint/key "
                f"for the server process. Cause: {exc}"
            ) from exc
        await self._check_search_circuit()

    async def _check_search_circuit(self) -> None:
        """Refuse to index when the vector search indexes are absent (DB-5 / INV-2).

        No-op without a connection (pure in-process unit tests) or when the operator opted
        out via ``require_search_indexes=False`` (graph-only use on a search-less Memgraph).
        """
        if self._conn is None or not self._settings.require_search_indexes:
            return
        missing = await missing_search_indexes(self._conn)
        if missing:
            raise SearchUnavailableError(
                f"Vector search indexes are missing ({', '.join(missing)}) — refusing to index "
                "because embeddings would be paid for but never searchable. Use the Memgraph "
                "MAGE image, or set REQUIRE_SEARCH_INDEXES=false to index anyway (search stays "
                "degraded and the missing indexes are logged at ERROR)."
            )

    async def _persist_scope(
        self,
        project: str,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> None:
        """Record this explicit index's include/exclude so a watcher update reproduces it."""
        await self._repo.set_project_scope(project, include_patterns or [], exclude_patterns or [])

    def _gate_update_cost(self, root_path: Path, changed_files: list[str]) -> None:
        """Abort a background watcher update whose changed set would exceed the cost cap.

        A ``git checkout`` touching thousands of files fires one debounced flush that would
        re-summarize and re-embed them all. INV-10 forbids the background from silently spending
        more than an explicit index, so the same cap gates it — estimated over just the
        changed files (deletes cost nothing). No cap configured (``None``) means no gate.
        """
        limit = self._settings.max_index_cost_usd
        if limit is None:
            return
        preview = self._builder.estimate_paths(root_path, changed_files)
        estimate = estimate_index_cost(preview, self._settings)
        if estimate["usd"] > limit:
            logger.error(
                "Watcher update for project rooted at %s would cost ~$%.4f (limit $%.2f) over "
                "%d changed path(s) — refusing to auto-spend; re-index explicitly to raise the cap",
                root_path, estimate["usd"], limit, len(changed_files),
            )
            raise IndexCostExceededError(estimate["usd"], limit)

    def get_project(self, path: str) -> str:
        """Return the project identifier for a directory path.

        This only *derives* an id from the path — it never checks whether that project was
        actually indexed. A typo'd or adjacent directory yields a valid-but-unindexed id, so
        callers that answer user queries should gate on :meth:`project_is_indexed` to avoid
        returning an empty page that looks like "no matches" (audit finding T5 / INV-7).
        """
        resolved = str(Path(path).resolve())
        if resolved not in self._projects:
            self._projects[resolved] = _project_name(Path(path))
        return self._projects[resolved]

    async def project_is_indexed(self, project: str) -> bool:
        """True when *project* has any indexed data (cheap probe; see the repository)."""
        return await self._repo.project_is_indexed(project)

    async def index_directory(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_cost_usd: float | None = None,
        progress: ProgressReporter | None = None,
        force_rebuild: bool = False,
    ) -> dict[str, object]:
        """Index a directory: full build or incremental update.

        A first-time (full) index is gated by a cost cap: ``max_cost_usd`` (per call) or
        ``Settings.max_index_cost_usd``. If the pre-flight estimate exceeds it, raises
        :class:`IndexCostExceededError` before spending anything. Incremental re-indexes reuse
        unchanged summaries and are not gated.

        ``force_rebuild`` re-parses, re-resolves and reconciles the whole project even when
        no file content changed. The incremental path short-circuits on content checksums,
        so an indexer upgrade (new edge semantics, new node properties) would otherwise
        never reach an already-indexed unchanged project. Summaries are still reused by
        content hash and embeddings survive the reconcile — no LLM re-spend.

        Returns statistics about the indexed project.
        """
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        project = self.get_project(path)
        logger.info("Indexing directory=%s as project=%s", root, project)

        async with self._project_locks[project], project_lease(self._conn, project):
            await self._preflight()
            existing_stats = await self._repo.get_project_stats(project)
            stored_chunk_embeddings: dict[str, list[float]] = {}
            if existing_stats["nodes"] > 0:
                stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings(project)
                if force_rebuild:
                    logger.info("Force rebuild requested for project=%s", project)
                    stored_summaries = await self._repo.get_stored_summaries(project)
                    result = await self._builder.build(
                        project,
                        root,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                        stored_summaries=stored_summaries,
                        progress=progress,
                    )
                else:
                    result = await self._builder.update(
                        project,
                        root,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                        progress=progress,
                    )
                nodes, edges = result.nodes, result.edges
                if not nodes and not edges:
                    await self._repo.set_project_root_path(project, str(root))
                    await self._persist_scope(project, include_patterns, exclude_patterns)
                    repaired = await self._repair_incomplete_index(
                        project, root, stored_chunk_embeddings, progress
                    )
                    stats: dict[str, object] = {
                        "project": project,
                        "directory": str(root),
                        **existing_stats,
                        "reused": True,
                        **repaired,
                    }
                    logger.info("Project already indexed and unchanged, reusing existing graph: %s", stats)
                    return stats
            else:
                logger.info("First-time indexing for project=%s", project)
                limit = max_cost_usd if max_cost_usd is not None else self._settings.max_index_cost_usd
                if limit is not None:
                    preview = self._builder.preview(
                        root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
                    )
                    estimate = estimate_index_cost(preview, self._settings)
                    logger.info("Estimated first-time index cost: $%.4f (limit $%.2f)", estimate["usd"], limit)
                    if estimate["usd"] > limit:
                        raise IndexCostExceededError(estimate["usd"], limit)
                result = await self._builder.build(
                    project,
                    root,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    progress=progress,
                )
                nodes, edges = result.nodes, result.edges

            # Record the absolute root so the dashboard can read source files from disk, and
            # persist this explicit index's scan scope so a later watcher update reproduces it
            # instead of widening the scanned universe (audit finding 5c).
            await self._repo.set_project_root_path(project, str(root))
            await self._persist_scope(project, include_patterns, exclude_patterns)

            # RAG: chunk + embed code, embed summaries
            chunks_stored = await self._rag.index_code(
                project, root, nodes,
                stored_chunk_embeddings=stored_chunk_embeddings,
                prune_stale=True,
                reconcilable_files=result.reconcilable_files,
                progress=progress,
            )
            summaries_stored = await self._rag.index_summaries(
                project, nodes,
                stored_summaries=result.stored_summaries,
                dirty_ids=result.dirty_summary_ids,
                progress=progress,
            )

            # Report what the database actually holds now, not just what this run
            # submitted — a partially failed run is visible in the delta.
            db_stats = await self._repo.get_project_stats(project)
            stats = {
                "project": project,
                "directory": str(root),
                "nodes": len(nodes),
                "edges": len(edges),
                "chunks": chunks_stored,
                "summaries": summaries_stored,
                "db": db_stats,
                "skipped_large_files": list(result.skipped_large_files),
                "reused": False,
            }
            logger.info("Indexing complete: %s", stats)
            return stats

    async def _repair_incomplete_index(
        self,
        project: str,
        root: Path,
        stored_chunk_embeddings: dict[str, list[float]],
        progress: ProgressReporter | None = None,
    ) -> dict[str, int]:
        """Heal state a killed index run left incomplete, without a rebuild.

        The unchanged fast path never reaches the summarize/chunk/embed phases, so a
        run killed between them would otherwise leave gaps that no re-run repairs.
        Returns {stat_key: count} for the repairs performed (empty when complete).
        """
        repaired: dict[str, int] = {}

        generated = await self._builder.repair_missing_summaries(project, root)
        if generated:
            repaired["summaries_repaired"] = len(generated)

        # Files whose chunks are missing, partial (killed mid-window), or stale content.
        # Re-chunk them and prune the old chunks in the same scoped pass, so a file that
        # changed but never got fresh chunks stops serving old content (audit finding 8b).
        stale_chunk_files = await self._repo.get_files_with_stale_chunks(project)
        if stale_chunk_files:
            nodes = await self._repo.get_nodes_for_files(project, stale_chunk_files)
            if nodes:
                logger.info(
                    "Repairing chunks for %d file(s) with missing/stale chunks for project=%s",
                    len(stale_chunk_files), project,
                )
                chunks_stored = await self._rag.index_code(
                    project, root, nodes,
                    stored_chunk_embeddings=stored_chunk_embeddings,
                    prune_stale=True,
                    reconcilable_files=set(stale_chunk_files),
                    progress=progress,
                )
                if chunks_stored:
                    repaired["chunks_repaired"] = chunks_stored

        # Runs after summary repair so freshly generated summaries are embedded too. Now
        # also catches summaries whose vector went stale (regenerated text, dropped embed),
        # not just missing ones (audit finding 8a).
        unembedded = await self._repo.get_stale_summary_embedding_nodes(project)
        if unembedded:
            logger.info(
                "Backfilling %d missing/stale summary embeddings for project=%s",
                len(unembedded), project,
            )
            embeddings_stored = await self._rag.index_summaries(project, unembedded, progress=progress)
            if embeddings_stored:
                repaired["summary_embeddings_repaired"] = embeddings_stored

        # A run killed after deleting the structural edges but before re-MERGEing them can
        # leave the tree edgeless; rebuild the CONTAINS backbone from parent_id so
        # traversals/structure aren't silently empty (audit finding 9).
        missing_edges = await self._repo.get_nodes_missing_contains_edges(project)
        if missing_edges:
            logger.info(
                "Repairing %d missing CONTAINS edge(s) for project=%s",
                len(missing_edges), project,
            )
            edges_restored = await self._repo.repair_contains_edges(project, missing_edges)
            if edges_restored:
                repaired["contains_edges_repaired"] = edges_restored

        return repaired

    async def preview_directory(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, object]:
        """Dry-run estimate for a directory — no DB writes, no LLM calls."""
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        project = self.get_project(path)
        preview = self._builder.preview(
            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )
        estimate = estimate_index_cost(preview, self._settings)
        return {
            "project": project,
            "directory": str(root),
            "preview": True,
            "estimated_cost_usd": estimate["usd"],
            **preview,
        }

    async def update_project(self, project: str, changed_files: list[str]) -> dict[str, object]:
        """Update a project after receiving changed-file notifications."""
        # Find root path for this project
        root_path: Path | None = None
        for path_str, proj in self._projects.items():
            if proj == project:
                root_path = Path(path_str)
                break

        if root_path is None:
            raise ValueError(f"Unknown project: {project}")

        logger.info(
            "Applying watcher update for project=%s: %d changed path(s) under %s",
            project,
            len(changed_files),
            root_path,
        )

        async with self._project_locks[project], project_lease(self._conn, project):
            # Reproduce the persisted include/exclude scope of the explicit index that
            # created the project (finding 5c) — cheap, and needed to detect a no-op against
            # the same scanned universe.
            include, exclude = await self._repo.get_project_scope(project)
            # A no-op flush (unchanged save, gitignored/untracked path, editor touch-event)
            # must cost nothing: detect it with a cheap scan+checksum diff BEFORE paying for
            # the preflight embedding or dumping every stored vector (INV-5, audit finding:
            # no-op flush costs a paid embedding + O(project) unload).
            if not await self._builder.has_pending_changes(
                project, root_path, changed_files,
                include_patterns=include or None, exclude_patterns=exclude or None,
            ):
                logger.info("Update was a no-op for project=%s (no indexable changes)", project)
                return {"project": project, "nodes": 0, "edges": 0, "chunks": 0,
                        "summaries": 0, "changed_files": len(changed_files), "noop": True}

            await self._preflight()
            # A background update obeys the same cost cap (finding 5b) and the same scan scope
            # (finding 5c) as the explicit index that created the project: gate the changed set,
            # then reproduce the persisted include/exclude so vendor/** excluded at index time is
            # not silently re-scanned, parsed, summarized and embedded on the first file save.
            self._gate_update_cost(root_path, changed_files)
            stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings(project)
            result = await self._builder.update(
                project,
                root_path,
                changed_files,
                include_patterns=include or None,
                exclude_patterns=exclude or None,
            )
            nodes, edges = result.nodes, result.edges

            if not nodes and not edges:
                logger.info("Update was a no-op for project=%s", project)
                return {"project": project, "nodes": 0, "edges": 0, "chunks": 0,
                        "summaries": 0, "changed_files": len(changed_files), "noop": True}

            chunks_stored = await self._rag.index_code(
                project, root_path, nodes,
                stored_chunk_embeddings=stored_chunk_embeddings,
                prune_stale=True,
                reconcilable_files=result.reconcilable_files,
            )
            summaries_stored = await self._rag.index_summaries(
                project, nodes,
                stored_summaries=result.stored_summaries,
                dirty_ids=result.dirty_summary_ids,
            )

            db_stats = await self._repo.get_project_stats(project)
            stats: dict[str, object] = {
                "project": project,
                "nodes": len(nodes),
                "edges": len(edges),
                "chunks": chunks_stored,
                "summaries": summaries_stored,
                "db": db_stats,
                "skipped_large_files": list(result.skipped_large_files),
                "changed_files": len(changed_files),
            }
            logger.info("Update complete: %s", stats)
            return stats
