"""Multi-project management: index, rebuild, and track indexed directories."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from inflorescence.code_indexer.graph_builder import BuildResult, GraphBuilder
from inflorescence.config import Settings
from inflorescence.cost import (
    IndexCostExceededError,
    estimate_index_cost,
    estimate_rechunk_cost,
    price_largest_files,
)
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.lease import project_lease
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import missing_search_indexes
from inflorescence.progress import ProgressReporter
from inflorescence.rag.indexer import RAGIndexer
from inflorescence.worktree import head_sha, worktree_main_root

logger = logging.getLogger(__name__)

# Why a directory cannot be queried, in the words a read tool passes on to its caller. Two
# shapes, because they call for two different actions: nothing was attempted, so index it —
# or an automatic attempt was made and failed, so fix that cause (INV-7: an unqueryable
# directory says so, and says why; it never answers with an empty page).
_NOT_INDEXED_REASON = "index it with index_directory first"
_AUTO_INDEX_FAILED_PREFIX = "auto-index attempted and failed: "


@dataclass(frozen=True)
class _DonorCoverage:
    """How a worktree's files differ from the donor project's index.

    ``uncovered`` — root-relative paths whose content the donor does not hold (edited by the
    branch, or new); ``deleted`` — paths the donor indexed that this checkout no longer has;
    ``scanned`` — how many files the walk saw, which is what makes a near-zero cost estimate
    explainable rather than merely surprising.
    """

    uncovered: list[str]
    deleted: list[str]
    scanned: int


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


def _relative_to(path: str, root: Path) -> str:
    """*path* relative to *root* when it is absolute and inside it (watcher events are absolute)."""
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(candidate)


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
        worktree_resolver: Callable[[Path], Path | None] = worktree_main_root,
    ) -> None:
        self._repo = repo
        self._builder = graph_builder
        self._rag = rag_indexer
        self._settings = settings
        # "Is this directory a linked git worktree, and of which checkout?" Injectable so
        # tests can exercise the seeding branch without laying out a real git repository;
        # production always uses the filesystem-archaeology implementation in worktree.py.
        self._worktree_resolver = worktree_resolver
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

    async def preflight_dependencies(self) -> None:
        """Public entry to the dependency pre-flight, for callers that need it up front.

        Indexing calls :meth:`_preflight` itself, so nothing here is required for correctness.
        The watch daemon uses it at startup for a different reason: it is a background process
        with no user watching stderr, and a daemon that cannot summarize must fail at boot with
        one clear message rather than come up "live", claim every project, and log an identical
        failure on every file anybody saves.
        """
        await self._preflight()

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

    async def _rechunk_backlog(
        self, project: str, root: Path, exclude: Iterable[str]
    ) -> tuple[list[str], float]:
        """Unchanged files this run will re-chunk, and what embedding their new chunks costs.

        A run re-chunks every file whose coverage marker is stale — including files whose
        bytes never moved but whose chunks were cut by an older chunker (``CHUNK_FORMAT``).
        The checksum-based gates never see those files, so without this the first run after
        a chunker upgrade would embed new chunks across the whole project unpriced — small
        money (embeddings only), but spent by a background flush that priced one file, which
        is exactly what INV-10 forbids. *exclude* holds files already priced as changed.
        """
        excluded = {str(Path(p)) for p in exclude}
        stale = await self._repo.get_files_with_stale_chunks(project)
        backlog = sorted(path for path in stale if path not in excluded)
        if not backlog:
            return [], 0.0
        preview = self._builder.estimate_paths(root, backlog)
        return backlog, estimate_rechunk_cost(preview, self._settings)["usd"]

    async def _gate_update_cost(self, project: str, root_path: Path, changed_files: list[str]) -> None:
        """Abort a background watcher update whose changed set would exceed the cost cap.

        A ``git checkout`` touching thousands of files fires one debounced flush that would
        re-summarize and re-embed them all. INV-10 forbids the background from silently spending
        more than an explicit index, so the same cap gates it — estimated over just the
        changed files (deletes cost nothing), plus the unchanged files whose chunks this run
        will re-cut (:meth:`_rechunk_backlog`). No cap configured (``None``) means no gate.
        """
        limit = self._settings.max_index_cost_usd
        if limit is None:
            return
        preview = self._builder.estimate_paths(root_path, changed_files)
        estimate = estimate_index_cost(preview, self._settings)["usd"]
        changed_rel = [_relative_to(p, root_path) for p in changed_files]
        backlog, backlog_usd = await self._rechunk_backlog(project, root_path, changed_rel)
        total = round(estimate + backlog_usd, 4)
        if total > limit:
            logger.error(
                "Watcher update for project rooted at %s would cost ~$%.4f (limit $%.2f) over "
                "%d changed path(s) and %d re-chunked file(s) — refusing to auto-spend; re-index "
                "explicitly to raise the cap",
                root_path, total, limit, len(changed_files), len(backlog),
            )
            raise IndexCostExceededError(
                total, limit, changed_files=len(changed_files), rechunk_files=len(backlog)
            )

    async def _gate_reindex_cost(
        self,
        project: str,
        root: Path,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        limit: float | None,
    ) -> None:
        """Abort an incremental re-index whose changed set would exceed the cost cap.

        The gate on the first-time branch prices the whole tree; that is the wrong question
        here, where almost every node is already paid for and reused by content hash — it
        would reject every routine re-index of a large repo. What this run pays for is the
        files whose checksum moved, so those are what gets priced (same shape as
        :meth:`_gate_update_cost`).

        Without this, INV-10 held only for a project's *first* index. Widening what the
        indexer admits (a new entry in ``DOCUMENT_EXTENSIONS``) makes thousands of files
        newly billable on the next ordinary re-index of an already-indexed project — a
        tree-sized bill behind a call that reads like a cheap refresh. Measured on a real
        repo: adding data formats turned a re-index into an ungated ~$57 run.

        Costs one directory walk + MD5 pass that the build below repeats. That is the same
        price the watcher path already pays, and it buys the only number worth gating on.
        """
        if limit is None:
            return
        changed = await self._builder.pending_change_paths(
            project, root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )
        backlog, backlog_usd = await self._rechunk_backlog(project, root, changed)
        if not changed and not backlog:
            return
        preview = self._builder.estimate_paths(root, changed) if changed else {}
        total = round(estimate_index_cost(preview, self._settings)["usd"] + backlog_usd, 4)
        logger.info(
            "Estimated re-index cost for project=%s: $%.4f (limit $%.2f) over %d changed file(s) "
            "and %d re-chunked file(s)",
            project, total, limit, len(changed), len(backlog),
        )
        if total > limit:
            raise IndexCostExceededError(
                total, limit, changed_files=len(changed), rechunk_files=len(backlog)
            )

    # ------------------------------------------------------------------
    # Git worktrees: index as a separate project, pay only for what differs
    # ------------------------------------------------------------------

    def _worktree_main_root(self, root: Path) -> Path | None:
        """The main checkout *root* is a linked worktree of, or ``None`` if it is not one.

        :func:`~inflorescence.worktree.worktree_main_root` is total by contract — damaged or
        unreadable git metadata answers "not a worktree" rather than raising. This wrapper
        holds that line for an *injected* resolver too: a detection failure may cost the
        worktree its cache reuse, but it must never be able to abort an index.
        """
        try:
            return self._worktree_resolver(root)
        except Exception:  # noqa: BLE001 — detection is advisory; indexing must survive it
            logger.warning(
                "Worktree detection failed for %s — indexing it as a standalone project",
                root, exc_info=True,
            )
            return None

    async def _stamp_worktree_meta(self, project: str, main_root: Path | None) -> None:
        """Record "this project is a worktree of <main checkout>" on the project root node.

        Written on every index of a detected worktree — including one whose main checkout was
        never indexed, so no cache seeding happened. The marker is a fact about the directory,
        not a receipt for the seeding, and a future janitor's ability to reap the index of a
        deleted worktree (and to leave every other project alone) rests on it being there.
        Ordinary projects are never marked.
        """
        if main_root is None:
            return
        await self._repo.set_project_worktree_meta(
            project, self.get_project(str(main_root)), str(main_root)
        )

    async def _stamp_indexed_head(self, project: str, root: Path) -> None:
        """Record the commit the index now matches the disk at. Never fails the run.

        Called only where the run leaves every scanned file indexed as it is on disk, so the
        stamp is a true statement; read tools compare it with the checkout's HEAD to notice a
        tree that moved on without the index (``index_freshness.drift``).
        """
        try:
            await self._repo.set_indexed_head(project, head_sha(root))
        except Exception:  # noqa: BLE001 — provenance must not fail an index that succeeded
            logger.warning("Could not stamp indexed_head for project=%s", project, exc_info=True)

    async def _donor_project(self, project: str, main_root: Path | None) -> str | None:
        """The indexed main-checkout project whose cache this first index may reuse.

        ``None`` — no seeding — when the directory is not a worktree, when the operator turned
        the feature off, or when the main checkout has no index to donate. The order matters:
        the cheap local checks come first so an ordinary directory reaches the database for
        exactly the same calls it always did.
        """
        if main_root is None or not self._settings.worktree_auto_index:
            return None
        donor = self.get_project(str(main_root))
        if donor == project:
            # Defensive: a resolver pointing at the worktree itself would otherwise make the
            # project its own donor and seed it from the empty index it is about to write.
            return None
        if not await self._repo.project_is_indexed(donor):
            logger.info(
                "Worktree of %s has no indexed main checkout (project=%s) — indexing in full",
                main_root, donor,
            )
            return None
        return donor

    async def _donor_coverage(
        self,
        root: Path,
        donor_project: str,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
    ) -> _DonorCoverage:
        """How this tree differs from the donor's index: uncovered, deleted, and how many scanned.

        One walk of the tree answers everything the worktree path needs. The scanned total is
        what makes the uncovered count explainable in a preview — "2 of 4300 files will be
        paid for" rather than a bare, suspiciously small dollar figure. ``deleted`` (indexed
        by the donor, absent here) costs nothing to index but has to be reconciled after a
        clone, or the branch's removed files keep answering queries.

        Deliberately a single call: the cost gate and the post-clone reconcile both need this
        diff, and a second call would re-read and re-hash every file in the repository for
        the same answer.
        """
        donor_checksums = await self._repo.get_checksums(donor_project)
        current = self._builder.scan_checksums(
            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )
        return _DonorCoverage(
            uncovered=sorted(rel for rel, md5 in current.items() if donor_checksums.get(rel) != md5),
            deleted=sorted(set(donor_checksums) - set(current)),
            scanned=len(current),
        )

    async def _discard_unfinished_clone(self, project: str) -> bool:
        """Delete a project whose server-side clone never finished. Returns whether it went.

        The nodes such a project holds are a partial copy of *another* project: complete for
        some labels, missing for others, with no identity of its own (see
        :meth:`GraphRepository.clone_project`). They are not an index of this directory, so
        they are thrown away and the directory is indexed from scratch — never updated in
        place, which would reconcile a fresh parse against a graph that was never whole.
        Nothing paid-for is lost: the donor still holds every summary and vector.
        """
        source = await self._repo.get_clone_marker(project)
        if source is None:
            return False
        logger.warning(
            "Project=%s holds an unfinished clone of project=%s (a previous run was killed "
            "mid-copy) — deleting it and indexing from scratch",
            project, source,
        )
        await self._repo.delete_project(project)
        return True

    async def _clone_donor_graph(self, project: str, donor_project: str) -> dict[str, int] | None:
        """Copy the donor's graph into *project* inside Memgraph. ``None`` means "fall back".

        The fast half of the worktree fast path: the donor's summaries and vectors are
        already the right values under the wrong project id, and moving them out to the
        client and back is the single most expensive thing the old path did.

        A failure here is never fatal. The clone is deleted and ``None`` is returned, and the
        caller then runs the original client-side seeding path — slower, but proven, and
        equally free of LLM and embedding spend. That is why this catches everything: a new
        Memgraph quirk must cost a worktree its speed, not its index.
        """
        if not self._settings.worktree_clone:
            return None
        try:
            return await self._repo.clone_project(donor_project, project)
        except Exception as exc:  # noqa: BLE001 — any clone failure degrades to the slow path
            logger.warning(
                "Server-side clone of donor project=%s into project=%s failed (%s) — falling "
                "back to client-side seeding for this index",
                donor_project, project, exc, exc_info=True,
            )
            try:
                await self._repo.delete_project(project)
            except Exception:  # noqa: BLE001 — the marker keeps the leftovers repairable
                logger.error(
                    "Could not delete the partial clone in project=%s; it stays marked as an "
                    "unfinished clone and the next index run will remove it",
                    project, exc_info=True,
                )
            return None

    async def _index_cloned_worktree(
        self,
        project: str,
        root: Path,
        donor_project: str,
        main_root: Path | None,
        clone_counts: dict[str, int],
        coverage: _DonorCoverage,
        include_patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        progress: ProgressReporter | None,
    ) -> dict[str, object]:
        """Finish a first index that started as a server-side clone of the donor's graph.

        The clone left this project holding the donor's graph and none of its own identity.
        Two things remain: stamp the identity, and pay for the difference between the two
        checkouts — nothing more, because everything else is already correct in the database.

        Finding that difference needs no git: the donor's per-file checksums (which the clone
        copied along with everything else) versus a plain scan of this tree names exactly the
        files whose content differs, plus the ones the branch deleted — *coverage*, already
        computed for the cost gate. The existing incremental path then does the real work,
        and the RAG phases are scoped to those files: the unscoped versions would re-write
        every chunk and every summary vector in the project — one vector-index update each —
        which is precisely the cost the clone avoided.
        """
        # Identity first: the update below reconciles against a project that must already
        # know where it lives and what scope it was indexed under.
        await self._repo.set_project_root_path(project, str(root))
        await self._stamp_worktree_meta(project, main_root)
        await self._persist_scope(project, include_patterns, exclude_patterns)

        touched = set(coverage.uncovered) | set(coverage.deleted)

        result = BuildResult(nodes=[], edges=[])
        chunks_stored = 0
        summaries_stored = 0
        if touched:
            logger.info(
                "Cloned worktree project=%s differs from donor project=%s in %d file(s) — "
                "reconciling just those", project, donor_project, len(touched),
            )
            result = await self._builder.update(
                project,
                root,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                progress=progress,
            )
            # Scope both paid phases to the files that actually differ. `reconcilable_files`
            # is intersected rather than replaced so a file the build failed to re-parse
            # keeps its cloned chunks instead of being emptied (INV-1/INV-3, finding 4).
            reconcilable = touched & result.reconcilable_files
            chunk_nodes = [n for n in result.nodes if n.file_path in touched]
            # Summaries additionally cover every node this run regenerated: a container's
            # summary moves when a child's does, and its vector must follow.
            summary_nodes = [
                n for n in result.nodes
                if n.file_path in touched or n.id in result.dirty_summary_ids
            ]
            if chunk_nodes or reconcilable:
                stored_chunk_embeddings = (
                    await self._repo.get_stored_chunk_embeddings_for_files(project, sorted(touched))
                )
                chunks_stored = await self._rag.index_code(
                    project, root, chunk_nodes,
                    stored_chunk_embeddings=stored_chunk_embeddings,
                    prune_stale=True,
                    reconcilable_files=reconcilable,
                    progress=progress,
                )
            if summary_nodes:
                summaries_stored = await self._rag.index_summaries(
                    project, summary_nodes,
                    stored_summaries=result.stored_summaries,
                    dirty_ids=result.dirty_summary_ids,
                    progress=progress,
                )
        else:
            logger.info(
                "Cloned worktree project=%s is byte-identical to donor project=%s — nothing "
                "left to index", project, donor_project,
            )

        await self._stamp_indexed_head(project, root)
        db_stats = await self._repo.get_project_stats(project)
        stats: dict[str, object] = {
            "project": project,
            "directory": str(root),
            # What this project holds now, which for a clone is the whole graph — reporting
            # only what the tail-end update touched would read as "indexed almost nothing".
            "nodes": db_stats["nodes"],
            "edges": db_stats["edges"],
            "chunks": db_stats["chunks"],
            "summaries": db_stats["summaries"],
            "db": db_stats,
            "skipped_large_files": list(result.skipped_large_files),
            "reused": False,
            "worktree_of": donor_project,
            # Present only on this path, so a caller can tell a server-side clone from the
            # client-side seeding that reports `seeded_summaries` instead.
            "cloned": dict(clone_counts),
            "changed_files": len(touched),
            "chunks_reindexed": chunks_stored,
            "summaries_reembedded": summaries_stored,
        }
        logger.info("Indexing complete (cloned worktree): %s", stats)
        return stats

    async def _persist_seeded_summaries(self, project: str, result: BuildResult) -> int:
        """Write the donor-reused summaries into the new project. **Not optional.**

        ``build`` persists each *generated* batch through its own callback and deliberately
        skips reused nodes — "they already match the DB". For a donor-seeded project that is
        false: the skeleton upsert wrote empty summaries for this project, and the reused text
        exists only on the in-memory nodes. Left unwritten, the project ends the run with an
        empty ``summary`` on every unchanged node, which is precisely what
        ``get_nodes_missing_summaries`` looks for — the repair pass would then regenerate,
        with real LLM spend, the summaries this whole path exists to avoid paying for.

        Only non-dirty nodes with non-empty summaries are written: dirty ones were already
        persisted batch-by-batch, and an empty summary must never be stored over a stored one
        (``store_summaries`` SETs unconditionally — INV-1).
        """
        items = [
            {
                "node_id": node.id,
                "summary": node.summary,
                "summary_input_hash": node.summary_input_hash,
                "summary_prompt_version": node.summary_prompt_version,
            }
            for node in result.nodes
            if node.summary and node.id not in result.dirty_summary_ids
        ]
        if items:
            await self._repo.store_summaries(project, items)
            logger.info(
                "Seeded %d reused summar(ies) into worktree project=%s from its main checkout",
                len(items), project,
            )
        return len(items)

    def get_project(self, path: str) -> str:
        """Return the project identifier for a directory path.

        This only *derives* an id from the path — it never checks whether that project was
        actually indexed. A typo'd or adjacent directory yields a valid-but-unindexed id, so
        callers that answer user queries must go through :meth:`ensure_indexed_for_query`
        (or at least :meth:`project_is_indexed`) to avoid returning an empty page that looks
        like "no matches" (audit finding T5 / INV-7).
        """
        resolved = str(Path(path).resolve())
        if resolved not in self._projects:
            self._projects[resolved] = _project_name(Path(path))
        return self._projects[resolved]

    async def project_is_indexed(self, project: str) -> bool:
        """True when *project* has any indexed data (cheap probe; see the repository)."""
        return await self._repo.project_is_indexed(project)

    async def ensure_indexed_for_query(self, directory: str) -> tuple[str, str | None]:
        """Resolve *directory* to a project a read tool may query, indexing it if that is free.

        Returns ``(project, reason)``. ``reason`` is ``None`` exactly when the project holds
        an index and the query may proceed; otherwise it is a human-readable clause the tool
        appends to its "no index found" error. Read tools call nothing else — the decision of
        what may be indexed on someone's behalf, and what that costs, lives here.

        The one case that indexes instead of refusing is a linked worktree of an **indexed**
        main checkout with ``worktree_auto_index`` on. That index is donor-seeded: every
        byte-identical file reuses the donor's summary and chunk vectors, so it costs parsing
        and writes but no LLM spend (see :meth:`index_directory`). Paying that on the first
        query is strictly better than answering "not indexed" to someone who just switched
        branches in a worktree they created seconds ago. It runs **synchronously** — the
        query that triggered it must see the finished index, not race it.

        Everything else refuses: an ordinary directory, a worktree whose main checkout was
        never indexed, the feature switched off. Each of those means summarizing a whole tree
        for real money, and INV-10 forbids spending it without an explicit, preview-approved
        index.

        Concurrency needs no extra machinery here: :meth:`index_directory` is serialized per
        project by an asyncio lock and a cross-process lease, so N parallel first queries
        produce one build and N-1 checksum no-ops.

        A failed attempt is **reported, never raised**: the tool contract is a structured
        error, so a dead API key or a tripped cost cap comes back as "auto-index attempted
        and failed: <cause>" instead of an exception the MCP layer renders as a crash, and
        never as an empty page that reads like "no matches" (INV-7).
        """
        project = self.get_project(directory)
        if await self.project_is_indexed(project):
            return project, None

        root = Path(directory)
        try:
            if not root.is_dir():
                # A typo'd or deleted path: nothing to index, and index_directory would only
                # raise FileNotFoundError at the caller.
                return project, _NOT_INDEXED_REASON
        except OSError:
            return project, _NOT_INDEXED_REASON

        donor = await self._donor_project(project, self._worktree_main_root(root))
        if donor is None:
            return project, _NOT_INDEXED_REASON

        # Inherit the donor's persisted scan scope. Without it the worktree would scan a
        # *wider* universe than the checkout it is seeded from, so a vendor/** the operator
        # excluded there gets parsed, summarized and embedded here — paid for, by a query.
        include, exclude = await self._repo.get_project_scope(donor)
        logger.info(
            "First query against unindexed worktree project=%s — indexing inline, seeded from "
            "donor project=%s", project, donor,
        )
        try:
            await self.index_directory(
                directory,
                include_patterns=include or None,
                exclude_patterns=exclude or None,
            )
        except Exception as exc:  # noqa: BLE001 — every failure becomes a structured error
            logger.warning("Auto-index of worktree %s failed", directory, exc_info=True)
            return project, f"{_AUTO_INDEX_FAILED_PREFIX}{exc}"

        if not await self.project_is_indexed(project):
            # The run succeeded but stored nothing (an empty tree, or one the inherited scope
            # excludes entirely). Querying it would return the empty page this whole method
            # exists to avoid, so it is still an error — with a different cause.
            return project, (
                f"{_AUTO_INDEX_FAILED_PREFIX}the index run stored no entities for this directory"
            )
        return project, None

    async def prune_out_of_scope(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Drop what an indexed project holds for files a narrowed scope no longer covers.

        Indexing never does this on its own: an excluded file is still on disk, so a
        re-index cannot tell "the operator narrowed the scope" from "the scan failed", and
        must not turn destructive on a guess (INV-1/INV-3 — a failure may leave data stale,
        never delete it). Narrowing is therefore its own explicit, free operation: no walk,
        no parse, no LLM, no embedding — pure path matching against the stored node index.

        The new excludes are **merged** into the project's persisted scan scope rather than
        replacing it, so a background watcher update stops re-admitting the pruned paths and
        the narrowing survives the next explicit index of the same scope. It does not survive
        an explicit index that passes a different scope — that call defines the scope, as it
        always has.

        With *dry_run* nothing is written and the same counts are returned, so the caller can
        show what would go before it goes.
        """
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        project = self.get_project(path)

        async with self._project_locks[project], project_lease(self._conn, project):
            if not await self._repo.project_is_indexed(project):
                raise ValueError(f"Project is not indexed: {path}")

            stored_include, stored_exclude = await self._repo.get_project_scope(project)
            merged_exclude = list(dict.fromkeys([*stored_exclude, *(exclude_patterns or [])]))
            effective_include = include_patterns if include_patterns is not None else (
                stored_include or None
            )

            plan = await self._builder.plan_scope_prune(
                project,
                include_patterns=effective_include,
                exclude_patterns=merged_exclude or None,
            )
            stats: dict[str, object] = {
                "project": project,
                "directory": str(root),
                "dry_run": dry_run,
                "files": len(plan.files),
                "nodes": len(plan.node_ids),
                "directories": len(plan.directory_ids),
                "chunks": len(plan.chunk_ids),
                "sample": sorted(plan.files)[:10],
                "scope": {"include": effective_include or [], "exclude": merged_exclude},
            }
            if dry_run or not plan.node_ids:
                logger.info("Scope prune plan for project=%s: %s", project, stats)
                return stats

            await self._repo.delete_nodes_by_ids(project, sorted(plan.node_ids))
            if plan.chunk_ids:
                await self._repo.delete_chunks_by_ids(project, sorted(plan.chunk_ids))
            # Persist last: if a delete failed, the scope is not yet narrowed and re-running
            # the command re-plans against the same store rather than a half-applied one.
            await self._repo.set_project_scope(project, effective_include or [], merged_exclude)
            stats["db"] = await self._repo.get_project_stats(project)
            logger.info("Scope prune applied for project=%s: %s", project, stats)
            return stats

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

        Every run that would spend is gated by a cost cap: ``max_cost_usd`` (per call) or
        ``Settings.max_index_cost_usd``. If the pre-flight estimate exceeds it, raises
        :class:`IndexCostExceededError` before spending anything. A first-time index is priced
        over the whole tree; an incremental re-index reuses unchanged summaries by content hash
        and is priced over just the files whose checksum moved — which is also what catches an
        indexer change that newly admits thousands of files into an existing project.

        ``force_rebuild`` re-parses, re-resolves and reconciles the whole project even when
        no file content changed. The incremental path short-circuits on content checksums,
        so an indexer upgrade (new edge semantics, new node properties) would otherwise
        never reach an already-indexed unchanged project. Summaries are still reused by
        content hash and embeddings survive the reconcile — no LLM re-spend.

        A directory that is a **linked git worktree** is indexed as its own project (it is a
        different tree), but its first index is seeded from the main checkout's project when
        that one is indexed: summaries and chunk vectors are reused for every byte-identical
        file, so only what the branch actually changed is paid for — and the cost cap is
        applied to exactly that set, not to the whole tree. A detected worktree is also
        marked as such on its project root node whether or not seeding happened; see
        :meth:`_stamp_worktree_meta`.

        That first index normally runs as a **server-side clone** (``worktree_clone``): the
        donor's graph is copied inside Memgraph, so no summary and no vector travels to the
        client and back, and only the files this checkout differs on are re-parsed and
        re-embedded. On a repository of a few thousand nodes that is tens of seconds rather
        than minutes — the residual cost is Memgraph's own vector-index insert, which no path
        avoids. If the clone fails or fails its verification it is deleted and the original
        client-side seeding runs instead; both cost nothing in LLM or embedding calls.

        Returns statistics about the indexed project. A seeded worktree index additionally
        reports ``worktree_of`` (the donor project), plus either ``cloned`` (the copied
        node/chunk/external/edge counts, server-side path) or ``seeded_summaries`` (the
        client-side path).
        """
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        project = self.get_project(path)
        # Filesystem-only detection (no `git` subprocess), so it is cheap enough to run on
        # every index and is what decides both the marker below and cache seeding.
        main_root = self._worktree_main_root(root)
        logger.info("Indexing directory=%s as project=%s", root, project)

        async with self._project_locks[project], project_lease(self._conn, project):
            await self._preflight()
            # Before anything reads this project's state: a killed server-side clone left
            # nodes that belong to another project. They must not be mistaken for an index
            # of this directory and updated in place.
            await self._discard_unfinished_clone(project)
            existing_stats = await self._repo.get_project_stats(project)
            seeded_from: str | None = None
            seeded_summaries = 0
            stored_chunk_embeddings: dict[str, list[float]] = {}
            if existing_stats["nodes"] > 0:
                # INV-10 covers re-indexes too, not only a project's first one: price the
                # files whose content moved before any of them is summarized or embedded.
                await self._gate_reindex_cost(
                    project,
                    root,
                    include_patterns,
                    exclude_patterns,
                    max_cost_usd if max_cost_usd is not None else self._settings.max_index_cost_usd,
                )
                if force_rebuild:
                    logger.info("Force rebuild requested for project=%s", project)
                    # Same project, so the stored vectors stay where they are: only the
                    # summary text and its hashes are needed to reuse paid work (the light
                    # read skips the per-node 1536-float payload entirely).
                    stored_summaries = await self._repo.get_stored_summaries_light(project)
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
                # Chunk vectors this project might reuse, read *after* the build so the read
                # can be scoped to the files that actually moved (an update re-chunks the
                # whole project but rewrites almost none of it — see RAGIndexer.index_code).
                # `changed_files is None` means the run narrowed nothing, i.e. a force
                # rebuild: an empty map is correct there too, because same-project reuse is
                # decided per chunk id inside index_code and needs no vectors on the wire.
                if result.changed_files:
                    stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings_for_files(
                        project, sorted(result.changed_files)
                    )
                if not nodes and not edges:
                    await self._repo.set_project_root_path(project, str(root))
                    await self._stamp_worktree_meta(project, main_root)
                    await self._persist_scope(project, include_patterns, exclude_patterns)
                    repaired = await self._repair_incomplete_index(project, root, progress)
                    await self._stamp_indexed_head(project, root)
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
                # A linked worktree is a different tree, hence its own project — but its files
                # are the main checkout's minus one branch's edits. Seeding this first index
                # with the main project's caches makes every unchanged node a hash hit:
                # summaries are reused by input hash, chunk vectors by content-derived id, so
                # no LLM call and no embedding is paid for content that is already indexed.
                donor_project = await self._donor_project(project, main_root)
                donor_summaries: dict[str, dict[str, object]] = {}
                # One scan of the tree answers both "what would this cost?" and, after a
                # clone, "what still has to be indexed?" — see _donor_coverage.
                coverage: _DonorCoverage | None = None
                limit = max_cost_usd if max_cost_usd is not None else self._settings.max_index_cost_usd
                if donor_project is not None and (limit is not None or self._settings.worktree_clone):
                    coverage = await self._donor_coverage(
                        root, donor_project, include_patterns, exclude_patterns
                    )
                if limit is not None:
                    if coverage is not None:
                        # Pricing the whole tree here would reject a free index: a large repo's
                        # full preview clears any sane cap, while the actual spend is limited to
                        # the files the donor does not already cover (INV-10 gates *spend*).
                        preview = self._builder.estimate_paths(root, coverage.uncovered)
                        logger.info(
                            "Worktree of an indexed checkout: pricing %d file(s) not covered by "
                            "donor project=%s", len(coverage.uncovered), donor_project,
                        )
                    else:
                        preview = self._builder.preview(
                            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
                        )
                    estimate = estimate_index_cost(preview, self._settings)
                    logger.info("Estimated first-time index cost: $%.4f (limit $%.2f)", estimate["usd"], limit)
                    if estimate["usd"] > limit:
                        raise IndexCostExceededError(estimate["usd"], limit)
                if donor_project is not None and coverage is not None:
                    # The fast path: copy the donor's graph inside Memgraph, then pay only
                    # for the files this checkout differs on. `None` means the copy could not
                    # be made or could not be trusted — the client-side seeding below then
                    # runs exactly as it did before, at the same (zero) LLM cost.
                    clone_counts = await self._clone_donor_graph(project, donor_project)
                    if clone_counts is not None:
                        return await self._index_cloned_worktree(
                            project, root, donor_project, main_root, clone_counts, coverage,
                            include_patterns, exclude_patterns, progress,
                        )
                if donor_project is not None:
                    donor_summaries = await self._repo.get_stored_summaries(donor_project)
                    stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings(donor_project)
                    logger.info(
                        "Seeding first index of project=%s from donor project=%s "
                        "(%d stored summar(ies), %d chunk vector(s))",
                        project, donor_project, len(donor_summaries), len(stored_chunk_embeddings),
                    )
                result = await self._builder.build(
                    project,
                    root,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    stored_summaries=donor_summaries or None,
                    progress=progress,
                )
                nodes, edges = result.nodes, result.edges
                if donor_project is not None:
                    # Must run before anything reads this project's summaries back — see
                    # _persist_seeded_summaries for why the build's own callback isn't enough.
                    seeded_from = donor_project
                    seeded_summaries = await self._persist_seeded_summaries(project, result)

            # Record the absolute root so the dashboard can read source files from disk, and
            # persist this explicit index's scan scope so a later watcher update reproduces it
            # instead of widening the scanned universe (audit finding 5c).
            await self._repo.set_project_root_path(project, str(root))
            await self._stamp_worktree_meta(project, main_root)
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

            await self._stamp_indexed_head(project, root)
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
            if seeded_from is not None:
                # Only present when the worktree fast path actually ran, so an ordinary
                # project's stats keep exactly the shape callers already parse.
                stats["worktree_of"] = seeded_from
                stats["seeded_summaries"] = seeded_summaries
            logger.info("Indexing complete: %s", stats)
            return stats

    async def _repair_incomplete_index(
        self,
        project: str,
        root: Path,
        progress: ProgressReporter | None = None,
    ) -> dict[str, int]:
        """Heal state a killed index run left incomplete, without a rebuild.

        The unchanged fast path never reaches the summarize/chunk/embed phases, so a
        run killed between them would otherwise leave gaps that no re-run repairs.
        Returns {stat_key: count} for the repairs performed (empty when complete).

        Reads nothing up front. The chunk half repairs a *named* set of files, so the vectors
        it can reuse are fetched for those files only — this used to receive a dump of every
        vector in the project, paid for on every unchanged re-index whether or not anything
        needed repairing.
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
                stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings_for_files(
                    project, stale_chunk_files
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
        """Dry-run estimate for a directory — no DB writes, no LLM calls.

        For a linked worktree of an indexed checkout the estimate covers **only the files the
        donor's index does not already hold**, because those are the only ones that will ever
        be paid for (see :meth:`index_directory`). Pricing the whole tree there quoted ~$50
        for an index that actually costs ~$0 — a preview that scares a caller away from a
        free operation is worse than no preview. The response then carries ``worktree_of``
        and the covered/uncovered file counts, so the small number is explainable instead of
        merely surprising.

        A non-worktree, and a worktree whose main checkout was never indexed, are unchanged:
        with nothing to seed from, the whole tree really will be paid for.
        """
        root = Path(path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")
        project = self.get_project(path)
        donor = await self._donor_project(project, self._worktree_main_root(root))
        if donor is not None:
            coverage = await self._donor_coverage(root, donor, include_patterns, exclude_patterns)
            uncovered, scanned = coverage.uncovered, coverage.scanned
            paths_preview = price_largest_files(self._builder.estimate_paths(root, uncovered), self._settings)
            paths_estimate = estimate_index_cost(paths_preview, self._settings)
            return {
                "project": project,
                "directory": str(root),
                "preview": True,
                "estimated_cost_usd": paths_estimate["usd"],
                # "files" keeps one meaning across both shapes: what this index would pay to
                # process. For a seeded worktree that is the uncovered set, not the tree.
                "files": len(uncovered),
                **paths_preview,
                "worktree_of": donor,
                "donor_covered_files": scanned - len(uncovered),
                "uncovered_files": len(uncovered),
            }
        preview = price_largest_files(
            self._builder.preview(root, include_patterns=include_patterns, exclude_patterns=exclude_patterns),
            self._settings,
        )
        estimate = estimate_index_cost(preview, self._settings)
        return {
            "project": project,
            "directory": str(root),
            "preview": True,
            "estimated_cost_usd": estimate["usd"],
            **preview,
        }

    async def scan_uncovered(
        self,
        directory: str,
        indexed_paths: Iterable[str],
        scope: Sequence[str] | None = None,
        max_examples: int | None = None,
    ) -> dict[str, object]:
        """Which files under *directory* the index does not hold — see
        :meth:`GraphBuilder.scan_uncovered`. *max_examples* raises how many are named
        (``find_text`` reads every one it is given).

        Read tools call this when their answer is empty, to keep "nothing matched" apart from
        "that file was never indexed". Off the event loop, because it is `scandir` I/O and
        this process also serves a dashboard whose health endpoint must keep answering
        (the same reason ``_scan_for_changes`` threads its walk).
        """
        if max_examples is None:
            return await asyncio.to_thread(
                self._builder.scan_uncovered, Path(directory), indexed_paths, scope
            )
        return await asyncio.to_thread(
            partial(self._builder.scan_uncovered, max_examples=max_examples),
            Path(directory),
            indexed_paths,
            scope,
        )

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
            await self._gate_update_cost(project, root_path, changed_files)
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

            # Scoped to the files that actually moved, and read only once the update has said
            # which those are. The whole-project dump this replaces cost seconds and a
            # project's worth of vectors on the wire on every single file save, to serve a
            # handful of chunks (INV-5); everything else an update can reuse is decided per
            # chunk id inside index_code, without moving a vector at all.
            stored_chunk_embeddings: dict[str, list[float]] = {}
            if result.changed_files:
                stored_chunk_embeddings = await self._repo.get_stored_chunk_embeddings_for_files(
                    project, sorted(result.changed_files)
                )

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

            await self._stamp_indexed_head(project, root_path)
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
