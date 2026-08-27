"""Orchestrator: scan files -> parse -> resolve -> summarize -> store in Memgraph."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pathspec

from inflorescence.code_indexer.models import (
    CodeNode,
    Edge,
    EdgeType,
    ExternalRef,
    FileChecksum,
    FileFacts,
    IndexerConfig,
    NodeType,
)
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.document_parser import DocumentParser
from inflorescence.code_indexer.parser.llm_parser import parse_file_with_llm
from inflorescence.code_indexer.parser.registry import ParserRegistry
from inflorescence.code_indexer.parser.resolver import resolve_edges
from inflorescence.code_indexer.scip_semantic import SemanticCalls, semantic_calls_pass
from inflorescence.code_indexer.summarizer import (
    node_source,
    summarize_hierarchy,
    summarize_single,
    summary_input_hash,
)
from inflorescence.config import LOCKFILE_NAMES, Settings
from inflorescence.db.repository import GraphRepository
from inflorescence.llm_client import LLMClient
from inflorescence.progress import ProgressReporter, resolve

logger = logging.getLogger(__name__)

# Cap on the in-memory LLM-fallback parse cache (audit finding 7). Keys are content-addressed
# ({rel}:{md5}), so entries only accumulate for distinct broken files across their edits — a
# handful in practice. The cap is a runaway backstop, not a working limit.
_LLM_PARSE_CACHE_MAX = 512


@dataclass
class BuildResult:
    """Per-run output of a build/update.

    Carries the summary-reuse state (stored summaries, dirty ids, skipped large files) that
    used to live on ``GraphBuilder`` instance fields. Returning it per call — instead of
    stashing it on ``self`` — is what lets two concurrent runs share one builder without
    overwriting each other's state between ``await`` points (INV-4, audit finding 2). Unpacks as
    ``(nodes, edges)`` so existing ``nodes, edges = await builder.build(...)`` sites are unchanged.
    """

    nodes: list[CodeNode]
    edges: list[Edge]
    stored_summaries: dict[str, dict[str, object]] = field(default_factory=dict)
    dirty_summary_ids: set[str] = field(default_factory=set)
    skipped_large_files: list[str] = field(default_factory=list)
    # Files this run authoritatively reconciled (re-parsed clean or confirmed gone).
    # Chunk pruning is scoped to these so a read/parse failure that hid a good file
    # from the scan can't strip its stored chunks (INV-1/INV-3, audit finding 4).
    reconcilable_files: set[str] = field(default_factory=set)
    # Root-relative paths whose content actually moved: changed on disk, or deleted. Set by
    # :meth:`GraphBuilder.update` — the only caller that *has* narrowed anything. ``None``
    # means "this run did not narrow" (a full build), and a consumer must not read it as "no
    # file changed": the distinction is what keeps a scoped read from silently becoming an
    # empty one. Everything else in this result still covers the whole project, because an
    # update re-parses and re-resolves all of it (a one-line edit changes cross-file CALLS).
    changed_files: set[str] | None = None

    def __iter__(self) -> Iterator[Any]:
        # Backward-compatible unpacking: `nodes, edges = await builder.build(...)`. Production
        # code reads the typed .nodes/.edges attributes; this is only for terse call sites.
        yield self.nodes
        yield self.edges


@dataclass
class _ReconcilePlan:
    """What a reconcile may delete, and what the store already holds.

    ``stored_edge_keys`` is the set of ``(source, TYPE, target)`` relationships that existed
    in the database when the plan was computed — read once, and used for both halves of the
    reconcile: the stale-edge diff, and deciding which of this run's edges are already there
    and need no MERGE at all.
    """

    stale_node_ids: set[str]
    stale_edges: list[dict[str, str]]
    reconcilable_files: set[str]
    stored_edge_keys: set[tuple[str, str, str]]


def _build_default_registry(config: IndexerConfig | None = None) -> ParserRegistry:
    """Create a registry with all available parsers."""
    config = config or IndexerConfig()
    registry = ParserRegistry()
    registry.register(PythonAstParser())
    if config.index_documents:
        # Registered like a language parser on purpose: everything downstream — the scan,
        # the watcher's extension filter, the cost preview, and the reconcile's
        # "is this file's parser still available?" check — keys off the registry, so
        # documents are covered by all of it without a parallel code path.
        registry.register(DocumentParser(config.document_extensions))
    try:
        from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser

        registry.register(JavaScriptParser())
    except ImportError:
        logger.debug("tree-sitter JS/TS not available, skipping JavaScript parser")
    try:
        from inflorescence.code_indexer.parser.go_parser import GoParser

        registry.register(GoParser())
    except ImportError:
        logger.debug("tree-sitter Go not available, skipping Go parser")
    try:
        from inflorescence.code_indexer.parser.rust_parser import RustParser

        registry.register(RustParser())
    except ImportError:
        logger.debug("tree-sitter Rust not available, skipping Rust parser")
    return registry


def _merge_semantic_calls(
    nodes: list[CodeNode],
    edges: list[Edge],
    semantic: SemanticCalls,
) -> list[Edge]:
    """Replace heuristic CALLS with semantic ones for the files SCIP covered.

    Per covered file the compiler's answer is authoritative for function/method
    callers: their heuristic CALLS edges are dropped, semantic edges added, external
    lists replaced and unresolved lists cleared. Module-node CALLS (top-level code)
    stay heuristic — SCIP references at module level can't be told apart from import
    statements. Uncovered files keep the full heuristic result (degradation, INV-3).
    """
    node_by_id = {n.id: n for n in nodes}
    covered = semantic.covered_files

    def keep(e: Edge) -> bool:
        if e.edge_type != EdgeType.CALLS:
            return True
        src = node_by_id.get(e.source)
        if src is None or src.file_path not in covered:
            return True
        return src.node_type == NodeType.MODULE

    merged = [e for e in edges if keep(e)]
    existing = {(e.source, e.target) for e in merged if e.edge_type == EdgeType.CALLS}
    merged.extend(e for e in semantic.edges if (e.source, e.target) not in existing)

    for n in nodes:
        if n.file_path not in covered:
            continue
        if n.node_type == NodeType.MODULE:
            n.calls_provenance = semantic.provenance.get(n.file_path, "semantic")
        elif n.node_type in (NodeType.FUNCTION, NodeType.METHOD):
            n.external_calls = semantic.external_calls.get(n.id, [])
            n.unresolved_calls = []
    return merged


def _confirmed_absent(path: Path) -> bool:
    """True only when *path* is positively gone from disk (ENOENT/ENOTDIR).

    ``Path.exists()`` collapses "unreadable" into "absent": it also returns False on
    ``PermissionError``, so a directory that lost read permission (chmod, dropped ACL,
    detached mount) would have its files counted as confirmed-deleted and their indexed
    data destroyed. Any error other than a definite not-found means "can't tell" — the
    reconcile must keep the file stale instead of deleting it.
    """
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError:
        return False
    return False


def _classify_changed_files(
    changed_files: list[str], current_paths: set[str], root: Path, indexed_paths: set[str]
) -> tuple[list[Path], list[str]]:
    """Split watcher-reported paths into (still-present, gone), keyed root-relative.

    Paths may arrive **absolute** (the file watcher emits ``event.src_path``) or
    **root-relative** (direct callers, tests). Both are normalized to the same
    root-relative key used by the scan (``str(f.relative_to(root))``) so a modified
    file is classified as *changed* rather than silently treated as *deleted*.
    Paths outside *root* are ignored.

    A path absent from the scan counts as *deleted* only if it was actually indexed
    before (*indexed_paths*). Events for paths the scan never covers — editor temp
    files, ignored or generated files — are dropped instead of masquerading as
    deletions and triggering a project rebuild.
    """
    changed: list[Path] = []
    deleted: list[str] = []
    for fp in changed_files:
        candidate = Path(fp)
        abs_path = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            rel = str(abs_path.relative_to(root))
        except ValueError:
            logger.debug("Ignoring change outside project root %s: %s", root, fp)
            continue
        if rel in current_paths and abs_path.exists():
            changed.append(abs_path)
        elif rel in indexed_paths:
            deleted.append(rel)
        else:
            logger.debug("Ignoring change to never-indexed path: %s", rel)
    return changed, deleted


class GraphBuilder:
    """Builds and incrementally updates a code graph stored in Memgraph."""

    def __init__(
        self,
        repo: GraphRepository | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        registry: ParserRegistry | None = None,
        config: IndexerConfig | None = None,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._settings = settings or Settings()
        # Config first: the default registry is derived from it (which document
        # extensions to scan, whether to scan them at all).
        self._config = config or IndexerConfig()
        self._registry = registry or _build_default_registry(self._config)
        # LLM-fallback parse results keyed by "{rel}:{md5}". Content-addressed, so a
        # concurrent build of another project writes different keys (never a clobber, unlike
        # the mutable per-run fields audit finding 2 removed), and re-parsing an unchanged
        # broken .py is a cache hit — no repeated LLM spend on every flush (INV-5, finding 7).
        self._llm_parse_cache: dict[str, tuple[list[CodeNode], list[Edge]]] = {}

    @property
    def extensions(self) -> set[str]:
        """File extensions the registered parsers can index (e.g. for watcher filtering)."""
        return set(self._registry.get_all_extensions())

    async def preflight(self) -> None:
        """Verify the LLM chat model answers before any graph mutation (INV-2, finding 6).

        A dead or misnamed ``LLM_MODEL`` raises ``NotFoundError`` on every summarize / fallback
        parse call, so a run would pay to build the graph and embed, then write a docstring
        surrogate for every summary. One tiny generation up front turns that into a clean
        abort. Skipped when no LLM is wired or neither summaries nor the LLM fallback parser
        is enabled — then the chat model is genuinely not a dependency of the run.
        """
        if self._llm is None:
            return
        if not (self._config.use_llm_summaries or self._config.use_llm_fallback_parser):
            return
        await self._llm.generate("ping", system_prompt="Reply with the single word: ok")

    async def build(
        self,
        project: str,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        stored_summaries: dict[str, dict[str, object]] | None = None,
        progress: ProgressReporter | None = None,
        unchanged_files: set[str] | None = None,
    ) -> BuildResult:
        """Full build: scan -> parse -> resolve -> directories -> store skeleton -> summarize -> checksums.

        *unchanged_files* is the incremental path's contribution: root-relative paths whose
        on-disk checksum still equals the stored one. Everything is still parsed and resolved
        across the whole project (CALLS edges are cross-file, so nothing less is correct) —
        the set only narrows what is **written back**, since a node whose file did not change
        and whose cross-file call facts still match the store is already exactly what this run
        would MERGE. ``None`` (a plain build) writes everything, unconditionally.
        """
        root = root_path.resolve()
        bar = resolve(progress)
        files, skipped_large_files = self._scan_files(
            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )
        logger.info("Scanned %d files for project=%s", len(files), project)

        all_nodes: list[CodeNode] = []
        all_edges: list[Edge] = []
        all_facts: dict[str, FileFacts] = {}

        for i, f in enumerate(files, 1):
            rel = f.relative_to(root)
            logger.info("[%d/%d] Parsing %s", i, len(files), rel)
            nodes, edges, facts = await self._parse_single_file(f, root)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            if facts is not None:
                all_facts[facts.file_path] = facts
            bar.update("parse", i, len(files))

        logger.info("Parsed %d nodes, %d edges", len(all_nodes), len(all_edges))
        resolution = resolve_edges(all_nodes, all_edges, all_facts)
        all_edges = resolution.edges
        node_by_id = {n.id: n for n in all_nodes}
        for node_id, calls in resolution.external_calls.items():
            if node_id in node_by_id:
                node_by_id[node_id].external_calls = calls
        for node_id, calls in resolution.unresolved_calls.items():
            if node_id in node_by_id:
                node_by_id[node_id].unresolved_calls = calls
        for n in all_nodes:
            if n.node_type == NodeType.MODULE:
                n.calls_provenance = "heuristic"
        logger.info(
            "Resolved edges: %d total (%d callers with external, %d with unresolved calls)",
            len(all_edges), len(resolution.external_calls), len(resolution.unresolved_calls),
        )

        # Semantic CALLS enrichment: SCIP indexers upgrade the heuristic edges wherever
        # the project actually builds; any failure (no binary, broken build, timeout)
        # leaves the heuristic result untouched. Runs in a thread — it is subprocess-bound.
        if self._config.use_scip_semantic:
            semantic = await asyncio.to_thread(semantic_calls_pass, root, all_nodes, self._config)
            if semantic is not None and semantic.covered_files:
                all_edges = _merge_semantic_calls(all_nodes, all_edges, semantic)
                logger.info(
                    "SCIP semantic pass covered %d file(s), %d semantic CALLS edges",
                    len(semantic.covered_files), len(semantic.edges),
                )

        _build_directory_nodes(all_nodes, all_edges)

        # Store in Memgraph (UNWIND-batched writes; each batch commits incrementally).
        # This is a RECONCILE, not a wipe-and-recreate: MERGE-based upserts keep extra
        # properties (summary_embedding) on surviving nodes, then only *provably* gone
        # nodes/edges are deleted. "Absent from this scan" is NOT "deleted": a missing
        # language parser or a transient read/parse failure yields zero nodes for a file
        # that still exists, and its data must survive as stale, never vanish (INV-1/INV-3,
        # audit findings 3, 4). On a fresh project the stale sets are empty and this
        # degrades to a plain build.
        #
        # The skeleton is stored BEFORE summarization: the graph becomes readable
        # (dashboard, search-by-name) while summaries stream in per batch below, and a
        # run killed mid-summarize keeps every already-persisted summary. The reconcile
        # plan must be computed before the first upsert — it reads the stored node index
        # as "what the previous run left", which the early upsert would overwrite.
        current_paths = {str(f.relative_to(root)) for f in files}
        produced_ids = {n.id for n in all_nodes}
        all_edges, external_refs = _resolve_external_edges(all_edges, all_nodes, produced_ids)
        if external_refs:
            produced_ids |= {ref.id for ref in external_refs}  # so their edges reconcile
            logger.info(
                "%d external dependenc(ies) recorded: %s",
                len(external_refs),
                dict(Counter(ref.kind for ref in external_refs)),
            )
        plan = await self._plan_reconcile(
            project, root, all_nodes, all_edges, current_paths, produced_ids
        )

        # What still has to be written. On a plain build that is everything; on an
        # incremental update the rows already provably identical in the store are dropped —
        # see _narrow_writes. The reconcile above ran on the FULL fresh graph, so what may be
        # deleted is decided from the same world as before (INV-1/INV-3).
        nodes_to_upsert, edges_to_upsert = all_nodes, all_edges
        if unchanged_files is not None:
            nodes_to_upsert, edges_to_upsert = await self._narrow_writes(
                project, all_nodes, all_edges, unchanged_files, plan.stored_edge_keys
            )

        # Report the store phase across nodes then edges so the bar advances monotonically.
        # Nodes carry summary='' here; the upsert query preserves stored non-empty
        # summaries (INV-1/INV-3), so a rebuild never wipes paid work.
        store_total = len(all_nodes) + len(all_edges)
        await self._repo.upsert_nodes(project, nodes_to_upsert)
        if external_refs:
            await self._repo.upsert_external_nodes(project, external_refs)
        bar.update("store", len(all_nodes), store_total)
        # Edges are MERGEd in BEFORE any deletion, so the graph never passes through an
        # edgeless window — a concurrent reader (or a crash) sees old∪new edges, then the
        # exact set, never "no edges" (audit finding 9). Skipping an edge already present in
        # the store keeps that property: the edge stays, it is simply not re-MERGEd. Stale
        # edges are then removed one by one; a run killed mid-prune leaves edges no worse
        # than stale.
        await self._repo.upsert_edges(project, edges_to_upsert)
        if plan.stale_node_ids:
            await self._repo.delete_nodes_by_ids(project, sorted(plan.stale_node_ids))
        if plan.stale_edges:
            await self._repo.delete_edges_by_keys(project, plan.stale_edges)
        # An :External node carries no file and no checksum, so the reconcile above — which
        # decides staleness from re-parsed files — can never reach it. A dependency the
        # project stopped using would linger forever; sweep the ones nothing points at.
        await self._repo.delete_orphan_external_nodes(project)
        bar.update("store", store_total, store_total)
        # Register the on-disk root as soon as the skeleton exists (the target dir:. node
        # is only just created by the upsert above) so a live dashboard can read sources
        # mid-run. project_manager re-stamps it after the run — idempotent.
        await self._repo.set_project_root_path(project, str(root))

        resolved_stored = stored_summaries or {}
        if self._config.use_llm_summaries and self._llm:
            # Every generated batch is persisted immediately: an interrupted run leaves
            # its spend in the DB, and the retry (which re-enters via update() because
            # nodes now exist) reuses those summaries by input hash at zero LLM cost (INV-8/INV-10).
            # Reused nodes never reach the callback — they already match the DB (INV-5).
            async def _persist_summary_batch(batch: Sequence[CodeNode]) -> None:
                items = [
                    {
                        "node_id": n.id,
                        "summary": n.summary,
                        "summary_input_hash": n.summary_input_hash,
                    }
                    for n in batch
                    # Only non-empty summaries (success or kept-stale, the latter with
                    # hash='' as the retry marker). A node that failed with nothing to
                    # keep stays ''/'' from the skeleton upsert — already the state the
                    # missing-summaries repair detects; writing it would risk clobbering
                    # a stored summary via store_summaries' unconditional SET (INV-1).
                    if n.summary
                ]
                if items:
                    await self._repo.store_summaries(project, items)

            dirty_summary_ids = await summarize_hierarchy(
                all_nodes, all_edges, root, self._llm,
                batch_size=self._config.batch_size,
                max_source_chars=self._settings.summary_max_source_chars,
                stored=resolved_stored,
                progress=bar,
                on_batch=_persist_summary_batch,
            )
        else:
            logger.info("Skipping LLM summarization")
            dirty_summary_ids = {n.id for n in all_nodes}

        # Update checksums (batched). A file that turned unreadable since the scan is
        # skipped rather than crashing the run — its checksum stays as-is, so the next run
        # sees a mismatch and re-processes it (self-healing, INV-3).
        checksum_items = [
            {"node_id": str(f.relative_to(root)), "checksum": ck.md5}
            for f in files
            if (ck := FileChecksum.try_compute(f, root)) is not None
        ]
        await self._repo.update_checksums(project, checksum_items)

        logger.info("Build complete: %d nodes, %d edges stored", len(all_nodes), len(all_edges))
        return BuildResult(
            nodes=all_nodes,
            edges=all_edges,
            stored_summaries=resolved_stored,
            dirty_summary_ids=dirty_summary_ids,
            skipped_large_files=skipped_large_files,
            reconcilable_files=plan.reconcilable_files,
        )

    async def _narrow_writes(
        self,
        project: str,
        all_nodes: list[CodeNode],
        all_edges: list[Edge],
        unchanged_files: set[str],
        stored_edge_keys: set[tuple[str, str, str]],
    ) -> tuple[list[CodeNode], list[Edge]]:
        """Drop the upserts that provably cannot change the stored graph.

        An incremental update re-parses the whole project — it has to, since CALLS edges
        depend on every other file — and then used to re-MERGE all of it: thousands of writes
        of unchanged data for a one-file edit (measured: ~2s of the ~3s the store phase took
        on 3k nodes). Two removals, each with its own proof:

        **Nodes.** A row is skipped only when (a) its file's checksum is unchanged, so
        everything the parser derives from that file's bytes — type, name, lines, signature,
        docstring, parent — is what the store already holds; (b) the store actually has a row
        for that id (a node the last run never wrote is not in ``stored_call_facts`` and is
        always written); and (c) the three properties that do *not* come from the file's own
        bytes — ``external_calls``, ``unresolved_calls``, ``calls_provenance``, all products
        of the whole-project resolution — still equal what this run computed. Nothing else in
        the row can differ: ``summary``/``summary_input_hash`` are sent empty here and the
        upsert query preserves the stored ones anyway (INV-1).

        What this deliberately does NOT cover is a *code* change: a new indexer version that
        would parse the same bytes into different rows never reaches an unchanged file this
        way. That is precisely what ``force_rebuild`` exists for (see
        ``ProjectManager.index_directory``), and it writes everything unconditionally.

        **Edges.** A relationship whose ``(source, TYPE, target)`` key is already in the store
        is not re-MERGEd — except CALLS, whose MERGE also refreshes ``resolution`` and
        ``callee_text``, and those can change while the key does not (a heuristic edge
        upgraded by a semantic pass). Every other edge type MERGEs no properties at all, so
        re-writing an existing key is a provable no-op. Existing edges are left in place, so
        the "never edgeless" property is strengthened, not weakened.
        """
        stored_facts: dict[str, dict[str, object]] = {}
        if unchanged_files:
            stored_facts = await self._repo.get_stored_call_facts(project, sorted(unchanged_files))

        def already_stored(node: CodeNode) -> bool:
            facts = stored_facts.get(node.id)
            return (
                facts is not None
                and facts.get("file_path") == node.file_path
                and node.file_path in unchanged_files
                and list(facts.get("external_calls") or []) == list(node.external_calls)
                and list(facts.get("unresolved_calls") or []) == list(node.unresolved_calls)
                and str(facts.get("calls_provenance") or "") == node.calls_provenance
            )

        nodes = [n for n in all_nodes if not already_stored(n)]
        edges = [
            e for e in all_edges
            if e.edge_type is EdgeType.CALLS
            or (e.source, e.edge_type.value.upper(), e.target) not in stored_edge_keys
        ]
        logger.info(
            "Incremental store: %d/%d node(s) and %d/%d edge(s) need writing for project=%s "
            "(%d unchanged file(s))",
            len(nodes), len(all_nodes), len(edges), len(all_edges), project, len(unchanged_files),
        )
        return nodes, edges

    async def _plan_reconcile(
        self,
        project: str,
        root: Path,
        all_nodes: list[CodeNode],
        all_edges: list[Edge],
        current_paths: set[str],
        produced_ids: set[str],
    ) -> _ReconcilePlan:
        """Decide what a reconcile may delete without destroying data a failure only hid.

        Returns a :class:`_ReconcilePlan` — what to delete, which files this run may prune
        chunks for, and (read in the same pass) which edges the store already holds.

        A stored node is stale only when its file was handled *authoritatively* this run:
        either re-parsed clean (its id is in the fresh node set) or confirmed gone from
        disk with its language still supported. Files a missing parser or a read/parse
        failure dropped from the scan are left entirely untouched — every parser emits at
        least a module node for a file it reads, so "scanned but produced nothing" means
        failure, not an empty file. ``reconcilable_files`` scopes chunk pruning the same way.
        """
        supported_ext = self.extensions
        produced_files = {n.file_path for n in all_nodes if n.node_type != NodeType.DIRECTORY}

        stored = await self._repo.get_stored_node_index(project)
        stored_nondir = [(nid, fp) for nid, fp, ntype in stored if ntype != NodeType.DIRECTORY.value]
        indexed_files = {fp for _, fp in stored_nondir if fp}

        # Files that were indexed but are absent from this scan. Split by *why*: a
        # confirmed on-disk deletion (still-supported language) is safe to remove; a file
        # whose language parser vanished (registry degraded) is kept stale, not deleted.
        missing_from_scan = indexed_files - current_paths
        confirmed_deleted = {
            fp for fp in missing_from_scan
            if Path(fp).suffix in supported_ext and _confirmed_absent(root / fp)
        }
        orphaned_language = {fp for fp in missing_from_scan if Path(fp).suffix not in supported_ext}
        if orphaned_language:
            logger.warning(
                "Reconcile kept %d indexed file(s) whose language parser is unavailable "
                "(e.g. %s) as stale rather than deleting them (project=%s)",
                len(orphaned_language), ", ".join(sorted(orphaned_language)[:3]), project,
            )

        reconcilable_files = produced_files | confirmed_deleted

        stale_node_ids: set[str] = set()
        for nid, fp in stored_nondir:
            if fp in confirmed_deleted:
                stale_node_ids.add(nid)  # whole file gone from disk
            elif fp in produced_files and nid not in produced_ids:
                stale_node_ids.add(nid)  # file re-parsed, this sub-node no longer exists

        # Directory nodes: prune only when no surviving file remains beneath them, so a
        # directory still holding an orphaned-language module (kept above) is never removed
        # and its subtree left dangling.
        surviving_files = (indexed_files - confirmed_deleted) | produced_files
        needed_dirs = {""}
        for fp in surviving_files:
            parts = fp.split("/")
            for i in range(1, len(parts)):
                needed_dirs.add("/".join(parts[:i]))
        for nid, fp, ntype in stored:
            if ntype == NodeType.DIRECTORY.value and nid not in produced_ids:
                dp = nid[len("dir:"):] if nid.startswith("dir:") else fp
                if (dp if dp != "." else "") not in needed_dirs:
                    stale_node_ids.add(nid)

        # Edges reconcile like nodes: fetch stored, diff in Python, delete only the ones
        # the fresh build dropped — and only when BOTH endpoints were re-parsed this run,
        # so an edge into a kept (orphaned-language / failed-parse) node survives.
        stored_edges = await self._repo.get_code_edges(project)
        produced_edge_keys = {(e.source, e.edge_type.value.upper(), e.target) for e in all_edges}
        stale_edges = [
            {"source": s, "type": ty, "target": t}
            for (s, ty, t) in stored_edges
            if (s, ty, t) not in produced_edge_keys and s in produced_ids and t in produced_ids
        ]
        return _ReconcilePlan(
            stale_node_ids=stale_node_ids,
            stale_edges=stale_edges,
            reconcilable_files=reconcilable_files,
            stored_edge_keys=set(stored_edges),
        )

    async def _scan_for_changes(
        self,
        project: str,
        root: Path,
        changed_files: list[str] | None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        full_checksums: bool = False,
    ) -> tuple[list[Path], list[str], set[str]]:
        """Scan + checksum diff only — no parsing, LLM, or embedding.

        Returns ``(changed_paths, deleted_rel_paths, unchanged_rel_paths)``. This is the cheap
        part of an update: a directory walk and per-file MD5, nothing billable.
        ``has_pending_changes`` uses it so a watcher flush can detect a no-op *before* paying
        for a preflight embedding or an O(project) vector dump (INV-5, audit finding: no-op
        flush costs an API call + full dump).

        The third element is only populated with *full_checksums*, and is what the write phase
        may treat as untouched. It is defined by MD5 alone, never by "the watcher did not
        mention it": a file that changed on disk without an event must still be rewritten, so
        it must not land in this set. That costs the watcher path an MD5 pass over the tree,
        which the caller (an update that is about to read and parse every one of those files
        anyway) can afford and ``has_pending_changes`` — which must stay cheap — does not pay.
        """
        files, _ = self._scan_files(root, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        current_paths = {str(f.relative_to(root)) for f in files}
        stored_checksums = await self._repo.get_checksums(project)

        changed: list[Path] = []
        deleted: list[str] = []
        unchanged: set[str] = set()
        if changed_files is not None:
            # Watcher-reported paths arrive absolute; callers/tests may pass root-relative
            # paths. Normalize both to a root-relative key before classifying.
            reported_changed, deleted = _classify_changed_files(
                changed_files, current_paths, root, indexed_paths=set(stored_checksums)
            )
            # Drop no-op modify events (editors and tools fire them liberally): only a
            # content change — or a file the index has never seen — justifies a rebuild.
            # A file unreadable at checksum time (vanished mid-flush) yields None and is
            # dropped from `changed` — never crashing the run; if it was truly deleted it
            # is already in `deleted`, otherwise it stays stale and is retried next flush.
            current: dict[str, str] = {}
            if full_checksums:
                current = {
                    ck.file_path: ck.md5
                    for f in files
                    if (ck := FileChecksum.try_compute(f, root)) is not None
                }
                unchanged = {
                    rel for rel, md5 in current.items() if stored_checksums.get(rel) == md5
                }
            for f in reported_changed:
                rel = str(f.relative_to(root))
                # The whole-tree pass, when it ran, already holds this file's digest —
                # hashing it a second time would only buy a different answer if the file
                # changed mid-run, and then `changed` should still describe one consistent
                # scan rather than a mix of two.
                md5 = current.get(rel) if full_checksums else (
                    ck.md5 if (ck := FileChecksum.try_compute(f, root)) is not None else None
                )
                if md5 is not None and stored_checksums.get(rel) != md5:
                    changed.append(f)
        else:
            deleted = sorted(set(stored_checksums) - current_paths)
            for f in files:
                rel = str(f.relative_to(root))
                ck = FileChecksum.try_compute(f, root)
                if ck is None:
                    continue
                if stored_checksums.get(rel) != ck.md5:
                    changed.append(f)
                else:
                    unchanged.add(rel)
        return changed, deleted, unchanged

    async def has_pending_changes(
        self,
        project: str,
        root_path: Path,
        changed_files: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """Cheaply answer "would an update do any work?" without billable calls.

        A watcher flush for an unchanged/ignored file (a no-op modify event, a gitignored
        path, a save that changed nothing) must not cost a preflight embedding or dump every
        stored vector before discovering there is nothing to do (INV-5). The caller gates on
        this before the expensive preflight/embedding-load path.
        """
        root = root_path.resolve()
        changed, deleted, _ = await self._scan_for_changes(
            project, root, changed_files,
            include_patterns=include_patterns, exclude_patterns=exclude_patterns,
        )
        return bool(changed or deleted)

    async def update(
        self,
        project: str,
        root_path: Path,
        changed_files: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        progress: ProgressReporter | None = None,
    ) -> BuildResult:
        """Update a project after detecting file changes.

        The graph is re-assembled in memory (parsing is cheap; summaries are reused by
        content hash) and then **reconciled** against the store: nodes are upserted in
        place, only stale nodes/chunks are deleted, and summary embeddings on unchanged
        nodes survive untouched. Nothing is wiped up front, so a failed run degrades to
        stale data instead of data loss.

        Two things make this cheaper than a build without making it narrower. The summary
        reuse map is fetched **without the vectors** (they are already on the nodes; nothing
        here has to move them), and the files whose checksum did not move are handed to
        :meth:`build` so their unchanged rows and already-present edges are not re-written.
        Both are write/wire economies only: the scan, the parse and the call resolution still
        cover the whole project, because a one-line edit in one file can change what another
        file is recorded as calling.

        The result carries ``changed_files`` — the paths this scan found different — so the
        caller can scope its own reads to them without re-deriving the diff.
        """
        root = root_path.resolve()
        changed, deleted, unchanged = await self._scan_for_changes(
            project, root, changed_files,
            include_patterns=include_patterns, exclude_patterns=exclude_patterns,
            full_checksums=True,
        )

        if not changed and not deleted:
            logger.info(
                "No indexable changes for project=%s (%d path(s) reported)",
                project,
                len(changed_files) if changed_files is not None else 0,
            )
            return BuildResult(nodes=[], edges=[], changed_files=set())

        logger.info(
            "Detected %d changed and %d deleted file(s) for project=%s; reconciling project",
            len(changed),
            len(deleted),
            project,
        )
        if changed:
            logger.debug("Changed: %s", ", ".join(sorted(str(p.relative_to(root)) for p in changed)))
        if deleted:
            logger.debug("Deleted: %s", ", ".join(sorted(deleted)))
        # Vector-free: this project's own summary vectors never have to leave the database
        # for a same-project reuse decision, and dragging ~1.5k floats per node over Bolt was
        # the single most expensive read an update did (INV-5).
        stored_summaries = await self._repo.get_stored_summaries_light(project)
        result = await self.build(
            project, root,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            stored_summaries=stored_summaries,
            progress=progress,
            unchanged_files=unchanged,
        )
        result.changed_files = {str(p.relative_to(root)) for p in changed} | set(deleted)
        return result

    async def repair_missing_summaries(self, project: str, root_path: Path) -> list[CodeNode]:
        """Regenerate summaries a killed run left missing, without a full rebuild.

        Leaves are re-summarized from source; containers from their children's
        stored summaries, children-first so a repaired child feeds its parent.
        Persists summary + input hash and returns the repaired nodes.
        """
        if not (self._config.use_llm_summaries and self._llm):
            return []
        missing = await self._repo.get_nodes_missing_summaries(project)
        if not missing:
            return []
        root = root_path.resolve()
        logger.info("Repairing %d missing summaries for project=%s", len(missing), project)

        repaired: list[CodeNode] = []

        async def _generate(node: CodeNode, children: list[CodeNode] | None) -> None:
            try:
                node.summary = await summarize_single(
                    node, root, self._llm, self._settings.summary_max_source_chars, children=children
                )
            except Exception as exc:  # noqa: BLE001 — mirror summarize_hierarchy's per-node resilience
                node.summary = node.docstring or ""
                # Retry marker: an empty input hash forces the next healthy run to
                # re-summarize this node. The old code wrote a *valid* hash even on
                # failure, so a transient LLM error permanently locked in the docstring
                # surrogate — the node was never re-summarized (audit finding 8c). This now
                # truly mirrors summarize_hierarchy, which clears the hash on failure.
                node.summary_input_hash = ""
                logger.warning("Failed to repair summary: %s: %s", node.id, exc)
            else:
                node.summary_input_hash = summary_input_hash(
                    node, children or [], source=node_source(node, root) if not children else ""
                )
            # An empty result means there is nothing to summarize (e.g. an empty
            # __init__.py), not damage — don't store or count it as repaired.
            if node.summary:
                repaired.append(node)

        leaves = [n for n, child_count in missing if child_count == 0]
        for i in range(0, len(leaves), self._config.batch_size):
            batch = leaves[i : i + self._config.batch_size]
            await asyncio.gather(*[_generate(n, None) for n in batch])
        await self._store_repaired(project, [n for n in leaves if n.summary])

        # Containers children-first, so a repaired child's summary is stored before
        # its parent reads it. CONTAINS is a tree, so the passes terminate.
        remaining = [n for n, child_count in missing if child_count > 0]
        while remaining:
            children_by_id = {
                n.id: await self._repo.get_child_summary_rows(project, n.id) for n in remaining
            }
            unresolved = {n.id for n in remaining}
            ready = [
                n for n in remaining
                if not any(c.id in unresolved for c in children_by_id[n.id])
            ] or remaining
            for i in range(0, len(ready), self._config.batch_size):
                batch = ready[i : i + self._config.batch_size]
                await asyncio.gather(*[_generate(n, children_by_id[n.id]) for n in batch])
            await self._store_repaired(project, [n for n in ready if n.summary])
            ready_ids = {n.id for n in ready}
            remaining = [n for n in remaining if n.id not in ready_ids]

        logger.info(
            "Repaired %d of %d missing summaries for project=%s (%d had nothing to summarize)",
            len(repaired), len(missing), project, len(missing) - len(repaired),
        )
        return repaired

    async def _store_repaired(self, project: str, nodes: list[CodeNode]) -> None:
        items = [
            {"node_id": n.id, "summary": n.summary, "summary_input_hash": n.summary_input_hash}
            for n in nodes
        ]
        if items:
            await self._repo.store_summaries(project, items)

    async def _parse_single_file(
        self, f: Path, root: Path
    ) -> tuple[list[CodeNode], list[Edge], FileFacts | None]:
        """Parse a file using the appropriate parser from the registry."""
        parser = self._registry.get_parser(f)
        if parser:
            nodes, edges, facts = parser.parse_file_with_facts(f, root)
        else:
            nodes, edges, facts = [], [], None

        # LLM fallback for Python files only
        if not nodes and f.suffix == ".py" and self._config.use_llm_fallback_parser and self._llm:
            rel = str(f.relative_to(root))
            # An unreadable file yields no checksum; skip the content-keyed cache entirely
            # (the LLM parse below will itself return nothing for it) rather than crash.
            ck = FileChecksum.try_compute(f, root)
            key = f"{rel}:{ck.md5}" if ck is not None else None
            cached = self._llm_parse_cache.get(key) if key is not None else None
            if cached is not None:
                logger.debug("LLM fallback cache hit (unchanged broken file): %s", rel)
                # Hand out fresh copies: downstream mutates nodes (parent_id, summary), and the
                # cache must stay pristine so the next hit isn't polluted by this run's state.
                c_nodes, c_edges = cached
                return (
                    [n.model_copy(deep=True) for n in c_nodes],
                    [e.model_copy(deep=True) for e in c_edges],
                    None,
                )
            logger.info("AST failed, LLM fallback: %s", rel)
            nodes, edges = await parse_file_with_llm(f, root, self._llm)
            # Cache only a real result keyed by content checksum, so an unchanged broken .py
            # costs one LLM call per content version, not one per flush (INV-5, finding 7). An
            # empty result is a transient read/LLM failure — leave it uncached so it's retried.
            if nodes and key is not None:
                self._cache_llm_parse(key, nodes, edges)

        return nodes, edges, facts

    def _cache_llm_parse(self, key: str, nodes: list[CodeNode], edges: list[Edge]) -> None:
        """Store a pristine deep copy of an LLM-fallback parse, bounded by a runaway cap."""
        if len(self._llm_parse_cache) >= _LLM_PARSE_CACHE_MAX:
            # Content-addressed entries are equal value for equal key, so eviction order is
            # immaterial to correctness; drop the oldest half to keep the map bounded.
            for stale in list(self._llm_parse_cache)[: _LLM_PARSE_CACHE_MAX // 2]:
                del self._llm_parse_cache[stale]
        self._llm_parse_cache[key] = (
            [n.model_copy(deep=True) for n in nodes],
            [e.model_copy(deep=True) for e in edges],
        )

    def _scan_files(
        self,
        root: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> tuple[list[Path], list[str]]:
        """Find files with registered extensions, honoring excludes, .gitignore, size, and nested worktrees.

        Returns ``(files, skipped_large_files)``. The skipped list is returned rather than
        stored on the instance so concurrent scans on a shared builder can't clobber it.
        """
        extensions = set(self._registry.get_all_extensions())
        includes = _normalize_patterns(include_patterns if include_patterns is not None else self._config.include_patterns)
        excludes = _normalize_patterns([*self._config.exclude_patterns, *(exclude_patterns or [])])
        gitignore = _load_ignore_specs(root, self._config.respect_gitignore)
        max_size = self._config.max_file_size_bytes
        files: list[Path] = []
        skipped_large: list[str] = []
        skipped_generated = 0

        for current_root, dirnames, filenames in os.walk(root):
            current_path = Path(current_root)
            kept_dirs: list[str] = []
            for dirname in dirnames:
                dir_path = current_path / dirname
                rel_dir = dir_path.relative_to(root).as_posix()
                if _matches_patterns(rel_dir, dirname, excludes):
                    continue
                if gitignore and _gitignored(rel_dir, gitignore, is_dir=True):
                    continue
                if _is_git_worktree(dir_path):
                    logger.info("Skipping linked git worktree: %s", rel_dir)
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs

            for filename in filenames:
                f = current_path / filename
                if f.suffix not in extensions:
                    continue
                if filename in LOCKFILE_NAMES:
                    # By name, unconditionally — before includes, so not even an explicit
                    # include_patterns can pull a resolver transcript into the index.
                    continue
                rel = f.relative_to(root).as_posix()
                if _matches_patterns(rel, filename, excludes):
                    continue
                if gitignore and _gitignored(rel, gitignore):
                    continue
                if includes and not _matches_patterns(rel, filename, includes):
                    continue
                try:
                    if f.stat().st_size > max_size:
                        skipped_large.append(rel)
                        logger.info("Skipping large file (>%d bytes): %s", max_size, rel)
                        continue
                except OSError:
                    continue
                if self._config.skip_generated and _is_generated_file(f, filename, self._config.generated_patterns):
                    skipped_generated += 1
                    logger.info("Skipping generated file: %s", rel)
                    continue
                files.append(f)

        if skipped_large:
            logger.info("Skipped %d file(s) over %d bytes: %s", len(skipped_large), max_size, ", ".join(skipped_large))
        if skipped_generated:
            logger.info("Skipped %d generated file(s)", skipped_generated)
        return sorted(set(files)), sorted(skipped_large)

    def preview(
        self,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        top_n: int = 10,
    ) -> dict[str, object]:
        """Estimate what would be indexed — no DB writes, no LLM calls.

        Runs the same scan pipeline (after .gitignore + size limit + excludes).
        """
        root = root_path.resolve()
        files, skipped_large_files = self._scan_files(
            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )

        languages: dict[str, dict[str, int]] = {}
        sizes: list[tuple[str, int]] = []
        total_bytes = 0
        total_lines = 0
        for f in files:
            try:
                data = f.read_bytes()
            except OSError:
                continue
            size = len(data)
            total_bytes += size
            total_lines += data.count(b"\n") + 1
            rel = f.relative_to(root).as_posix()
            sizes.append((rel, size))
            entry = languages.setdefault(_language_for_suffix(f.suffix), {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += size

        sizes.sort(key=lambda item: item[1], reverse=True)
        return {
            "files": len(files),
            "total_bytes": total_bytes,
            "languages": languages,
            "largest_files": [{"path": p, "bytes": b} for p, b in sizes[:top_n]],
            "skipped_large_files": list(skipped_large_files),
            "estimated_nodes": total_lines // 10,
            "estimated_tokens": total_bytes // 4,
        }

    def scan_checksums(
        self,
        root_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, str]:
        """Return ``{root-relative path: md5}`` for every file this scan would index.

        The same directory walk + MD5 pass :meth:`_scan_for_changes` runs, but diffed against
        a caller-supplied checksum map instead of the project's own stored one. That is what
        lets a *never-indexed* project (a fresh worktree) be compared against another
        project's checksums — the donor's — to decide which files are genuinely new content.
        Nothing billable happens here: no parse, no LLM, no embedding.

        A file unreadable at checksum time is omitted rather than raising: it is then treated
        as content the donor does not cover, i.e. priced, which is the safe direction (INV-3).
        """
        root = root_path.resolve()
        files, _ = self._scan_files(
            root, include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )
        checksums: dict[str, str] = {}
        for f in files:
            ck = FileChecksum.try_compute(f, root)
            if ck is not None:
                checksums[ck.file_path] = ck.md5
        return checksums

    def estimate_paths(self, root_path: Path, paths: list[str]) -> dict[str, object]:
        """Preview-style token/node stats for a specific set of changed paths.

        Lets a background watcher update be cost-gated (INV-10, finding 5b) without a full-project
        scan: only the named files that still exist and carry an indexable extension are read.
        Deleted files simply don't exist and contribute nothing. Accepts absolute or
        root-relative paths (the watcher emits absolute) and mirrors :meth:`preview`'s
        token/node proxy so :func:`estimate_index_cost` reads the result the same way.
        """
        root = root_path.resolve()
        extensions = self.extensions
        total_bytes = 0
        total_lines = 0
        seen: set[str] = set()
        for p in paths:
            candidate = Path(p)
            f = (candidate if candidate.is_absolute() else root / candidate).resolve()
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                continue
            if rel in seen or f.suffix not in extensions or f.name in LOCKFILE_NAMES:
                continue
            seen.add(rel)
            try:
                data = f.read_bytes()
            except OSError:
                continue
            total_bytes += len(data)
            total_lines += data.count(b"\n") + 1
        return {"estimated_tokens": total_bytes // 4, "estimated_nodes": total_lines // 10}


_PREVIEW_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    # Documents, grouped by kind rather than by extension, so the pre-flight breakdown
    # reads "markdown: 40 files" instead of splitting the same prose across .md/.mdx.
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".rst": "restructuredtext",
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".org": "org",
    ".txt": "text",
    ".text": "text",
    # Styles, markup and configuration — same grouping rule: variants of one format count
    # together, so the estimate reads "yaml: 12 files", not ".yml: 7" plus ".yaml: 5".
    ".css": "css",
    ".scss": "sass",
    ".sass": "sass",
    ".less": "less",
    ".styl": "stylus",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}


def _language_for_suffix(suffix: str) -> str:
    return _PREVIEW_LANGUAGES.get(suffix.lower(), suffix.lstrip(".").lower() or "other")


def _load_ignore_specs(root: Path, respect_gitignore: bool) -> dict[str, pathspec.PathSpec]:
    """Compile per-directory ignore specs under *root*, keyed by posix directory rel-path ('' for root).

    A project-local ``.inflorescenceignore`` is always honored; ``.gitignore`` only when
    *respect_gitignore* is True. Patterns from both files in the same directory are merged
    into a single PathSpec.
    """
    names = [".inflorescenceignore"] + ([".gitignore"] if respect_gitignore else [])
    per_dir: dict[str, list[str]] = defaultdict(list)
    for name in names:
        for f in root.rglob(name):
            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            rel_dir = f.parent.relative_to(root).as_posix()
            per_dir["" if rel_dir == "." else rel_dir].extend(lines)
    return {k: pathspec.PathSpec.from_lines("gitwildmatch", v) for k, v in per_dir.items()}


def _gitignored(rel_path: str, specs: dict[str, pathspec.PathSpec], is_dir: bool = False) -> bool:
    """Return True if *rel_path* (posix, relative to root) is ignored by any applicable ``.gitignore``."""
    for base, spec in specs.items():
        if base == "":
            candidate = rel_path
        elif rel_path.startswith(base + "/"):
            candidate = rel_path[len(base) + 1 :]
        else:
            continue
        if not candidate:
            continue
        if spec.match_file(candidate) or (is_dir and spec.match_file(candidate + "/")):
            return True
    return False


def _is_generated_file(path: Path, filename: str, patterns: list[str]) -> bool:
    """True if the file is machine-generated — matched by a known generated-file glob, or
    carrying a canonical generated marker in its head. Protobuf/gRPC Go stubs carry
    ``// Code generated ... DO NOT EDIT.``; JS/TS/Python grpc stubs are matched by name."""
    if any(fnmatch.fnmatch(filename, pat) for pat in patterns):
        return True
    try:
        head = path.read_bytes()[:2048].decode("utf-8", errors="ignore")
    except OSError:
        return False
    if "@generated" in head:
        return True
    return "Code generated" in head and "DO NOT EDIT" in head


def _is_git_worktree(dir_path: Path) -> bool:
    """Return True only for a *linked git worktree*.

    A linked worktree is a ``.git`` **file** whose gitdir points into ``.../worktrees/...``
    — a second checkout of the same repository, so indexing it would double-count the code.
    An independent nested repository (``.git`` **directory**) or a submodule
    (``.git`` file pointing into ``.../modules/...``) is NOT a worktree and must be indexed.
    """
    git_marker = dir_path / ".git"
    if not git_marker.is_file():
        return False
    try:
        content = git_marker.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return content.startswith("gitdir:") and "/worktrees/" in content


def _normalize_patterns(patterns: list[str] | None) -> list[str]:
    if not patterns:
        return []
    return [_normalize_path_pattern(p) for p in patterns if p.strip()]


def _normalize_path_pattern(pattern: str) -> str:
    normalized = pattern.replace("\\", "/").strip().rstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _matches_patterns(rel_path: str, name: str, patterns: list[str]) -> bool:
    normalized = _normalize_path_pattern(rel_path)
    parts = normalized.split("/")
    for pattern in patterns:
        if (
            normalized == pattern
            or normalized.startswith(pattern + "/")
            or fnmatch.fnmatch(normalized, pattern)
            or fnmatch.fnmatch(name, pattern)
            or any(fnmatch.fnmatch(part, pattern) for part in parts)
        ):
            return True
    return False


# Node types that stand for a whole file, and so hang directly off a directory node.
# A document is one of them: it is a file with no symbols inside it, not a symbol.
_FILE_NODE_TYPES = frozenset({NodeType.MODULE, NodeType.DOCUMENT})


_LANGUAGE_BY_SUFFIX = {
    ".go": "go", ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".rs": "rust",
}


def _external_root(name: str, language: str) -> str:
    """The distributable a specifier belongs to, for "what does this project depend on".

    ``pydantic.v1.main`` -> ``pydantic``; ``@scope/pkg/sub`` -> ``@scope/pkg``;
    ``github.com/x/y/internal/z`` -> ``github.com/x/y``. A Go stdlib path (no dot in its
    first segment) is its own root, since ``net`` alone would name nothing.
    """
    if language in ("typescript", "javascript"):
        parts = name.split("/")
        if name.startswith("@"):
            return "/".join(parts[:2]) if len(parts) >= 2 else name
        return parts[0]
    if language == "go":
        parts = name.split("/")
        if "." not in parts[0]:
            return name  # stdlib: net/http, encoding/json
        return "/".join(parts[:3]) if len(parts) >= 3 else name
    return name.split(".")[0]  # python, rust, and any base-class name


def _resolve_external_edges(
    edges: list[Edge], nodes: list[CodeNode], produced_ids: set[str]
) -> tuple[list[Edge], list[ExternalRef]]:
    """Rewrite edges leaving the project so they point at :External nodes.

    An ``IMPORTS``/``INHERITS`` target that is not a produced node is not a mistake — it is a
    dependency: ``net/http``, ``@electron-forge/maker-zip``, ``pydantic.BaseModel``. Those
    edges used to be submitted with a target no node carried, so the write layer's MATCHes
    discarded them silently (29% of all edges on a Go service, 24% on an Electron app) and
    the shortfall was logged as an ERROR indistinguishable from real data loss.

    Here each such target is turned into an ``:External`` node and the edge is repointed at
    it, so the dependency becomes a first-class, traversable part of the graph. CALLS never
    reaches this path — the resolution ladder already keeps unresolved callees on the caller
    as ``external_calls``.
    """
    language_by_file = {n.file_path: _LANGUAGE_BY_SUFFIX.get(Path(n.file_path).suffix, "") for n in nodes}
    source_files = {n.id: n.file_path for n in nodes}

    resolved: list[Edge] = []
    refs: dict[str, ExternalRef] = {}
    for edge in edges:
        if edge.source in produced_ids and edge.target in produced_ids:
            resolved.append(edge)
            continue
        if edge.source not in produced_ids or edge.edge_type not in (EdgeType.IMPORTS, EdgeType.INHERITS):
            # A source that does not exist is a genuine anomaly, and any other edge type
            # leaving the project is not something this function knows how to name.
            continue
        kind = "module" if edge.edge_type is EdgeType.IMPORTS else "class"
        language = language_by_file.get(source_files.get(edge.source, ""), "")
        ref_id = f"ext:{kind}:{edge.target}"
        refs.setdefault(
            ref_id,
            ExternalRef(
                id=ref_id,
                name=edge.target,
                root=_external_root(edge.target, language),
                kind=kind,
                language=language,
            ),
        )
        resolved.append(edge.model_copy(update={"target": ref_id}))
    return resolved, list(refs.values())


def _build_directory_nodes(nodes: list[CodeNode], edges: list[Edge]) -> None:
    """Create DIRECTORY nodes and CONTAINS edges for the directory hierarchy.

    Mutates *nodes* and *edges* in place.
    """
    dir_paths: set[str] = set()
    module_dir: dict[str, str] = {}
    for n in nodes:
        if n.node_type in _FILE_NODE_TYPES:
            parts = n.file_path.split("/")
            module_dir[n.file_path] = "/".join(parts[:-1]) if len(parts) > 1 else ""
            for i in range(1, len(parts)):
                dir_paths.add("/".join(parts[:i]))

    if dir_paths or any(v == "" for v in module_dir.values()):
        dir_paths.add("")

    dir_nodes: dict[str, CodeNode] = {}
    for dp in sorted(dir_paths):
        name = dp.split("/")[-1] if dp else "."
        node_id = f"dir:{dp}" if dp else "dir:."
        parent_dp = "/".join(dp.split("/")[:-1]) if "/" in dp else ("" if dp else None)
        parent_id: str | None = None
        if parent_dp is not None:
            parent_id = f"dir:{parent_dp}" if parent_dp else "dir:."
        dn = CodeNode(
            id=node_id,
            name=name,
            node_type=NodeType.DIRECTORY,
            file_path=dp if dp else ".",
            line_start=0,
            line_end=0,
            parent_id=parent_id,
        )
        dir_nodes[dp] = dn

    nodes.extend(dir_nodes.values())

    for n in nodes:
        if n.node_type in _FILE_NODE_TYPES and n.file_path in module_dir:
            dp = module_dir[n.file_path]
            if dp in dir_nodes:
                n.parent_id = dir_nodes[dp].id
                edges.append(Edge(source=dir_nodes[dp].id, target=n.id, edge_type=EdgeType.CONTAINS))

    for dn in dir_nodes.values():
        if dn.parent_id and dn.parent_id.startswith("dir:"):
            edges.append(Edge(source=dn.parent_id, target=dn.id, edge_type=EdgeType.CONTAINS))

    logger.info("Built %d directory nodes", len(dir_nodes))
