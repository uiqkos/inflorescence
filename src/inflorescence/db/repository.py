"""High-level async repository wrapping Cypher queries for Memgraph."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from time import monotonic, time
from typing import Any

from inflorescence.code_indexer.models import CodeNode, Edge, ExternalRef, NodeType
from inflorescence.db import lease, queries
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.schema import LEVEL_SUMMARY_INDEXES

logger = logging.getLogger(__name__)

# Default number of rows per UNWIND write. Bounds transaction size and peak
# memory per query while keeping round-trips ~1/batch_size of the row count.
DEFAULT_WRITE_BATCH_SIZE = 500

# Memgraph doesn't support parameterized relationship types, so callers group
# rows by type and format the type in; every row in $rows shares it. RETURN count(*)
# reports the edges the DB actually wrote — a row whose source or target id doesn't
# resolve to a node (e.g. an unresolved CALLS target like `len`) is silently dropped by
# the MATCHes, so counting matched rows tells the truth instead of "all submitted" (INV-7).
# The target may be an :External dependency (an imported package, an inherited base class
# that lives outside the tree); a :Code-only MATCH here is what used to discard those edges
# silently — 29% of all edges on a Go service. CALLS never points outside, so its template
# below stays :Code-scoped.
#
# The "is this an allowed target" test is written over `labels(t)` rather than as the
# `t:Code OR t:External` disjunction it replaces. Memgraph 3.12.0 miscompiles a batched
# (UNWIND) write whose WHERE is a disjunction of *pure label tests* whenever the planner can
# serve the MATCH from a label-property index: the write reaches only the first row, and
# `RETURN count(*)` reports that truncated number, so the shortfall is invisible (this is the
# MARK_CHUNKS_COVERED defect — see queries.py). This query only escaped it because its target
# MATCH carries no label, so no label-property index applies; probed live, adding a label to
# that MATCH drops the same query from 3 written edges to 0. That immunity is an invisible
# accident one innocent edit would undo, and the failure would be silent edge loss, so the
# shape is removed rather than commented around. The `labels()` form is live-validated to
# write every row with and without an index in play.
_UPSERT_EDGES_BATCH_TEMPLATE = (
    "UNWIND $rows AS row "
    "MATCH (s:Code {{id: row.source, project: $project}}) "
    "MATCH (t {{id: row.target, project: $project}}) "
    "WHERE ANY(label IN labels(t) WHERE label IN ['Code', 'External']) "
    "MERGE (s)-[:{edge_type}]->(t) "
    "RETURN count(*) AS written"
)

# CALLS edges carry provenance: which ladder level (or SCIP indexer) bound the target
# and the call text as written. MERGE keeps the edge unique per (s, t); SET refreshes
# provenance so a semantic pass upgrades a heuristic edge in place.
_UPSERT_CALLS_EDGES_TEMPLATE = (
    "UNWIND $rows AS row "
    "MATCH (s:Code {id: row.source, project: $project}) "
    "MATCH (t:Code {id: row.target, project: $project}) "
    "MERGE (s)-[r:CALLS]->(t) "
    "SET r.resolution = row.resolution, r.callee_text = row.callee_text "
    "RETURN count(*) AS written"
)

_NODE_TYPE_LABEL_BY_VALUE = {
    NodeType.DIRECTORY.value: "Directory",
    NodeType.MODULE.value: "Module",
    NodeType.CLASS.value: "Class",
    NodeType.FUNCTION.value: "Function",
    NodeType.METHOD.value: "Method",
    NodeType.INTERFACE.value: "Interface",
    NodeType.ENUM.value: "Enum",
    NodeType.STRUCT.value: "Struct",
    NodeType.TRAIT.value: "Trait",
    # Prose files. Labelled apart from :Module so "list the modules" doesn't return
    # READMEs, and so the file-level queries below can address both explicitly.
    NodeType.DOCUMENT.value: "Document",
}
# Invariant (guarded by a test): every NodeType has an entry here. upsert_nodes indexes
# this map by value, so a type added without a label raises KeyError on its first node.
_NODE_TYPE_VALUE_BY_LABEL = {label: value for value, label in _NODE_TYPE_LABEL_BY_VALUE.items()}

# Rows per clone transaction. Far larger than DEFAULT_WRITE_BATCH_SIZE because nothing
# crosses the wire here: the batch exists only to bound how much a single Memgraph
# transaction has to hold (a chunk batch materialises ~batch_size × 1536 floats server-side).
# Measured on 5.5k nodes / 8.1k chunks, one query per label was the *fastest* variant — the
# per-batch re-scan costs more than the transaction size saves — so the batch is set high
# enough that ordinary projects finish in one or two round trips and only genuinely huge
# ones ever paginate.
CLONE_BATCH_SIZE = 5000

# How many clone statements are in flight at once.
#
# The node half of a clone is a set of *independent* writes — one per node-type label, plus
# chunks and externals, each further splittable into disjoint id ranges — but it used to be
# issued one statement at a time, so the client sat waiting on a round trip while Memgraph
# had nothing else to do.
#
# Four, because four is where it stops helping. Measured on a 3.8k-node / 8.1k-chunk /
# 14.3k-edge project, best of two runs each, same source and the same verified result:
#
#     serial (before)   33.6s      2 workers   19.6s
#     1 worker          32.5s      4 workers   13.6s
#                                  8 workers   18.4s
#
# Past four the writes contend — one storage engine, one shared vector index — and it gets
# *worse*, and every concurrent statement also holds a Bolt connection and makes Memgraph
# materialise its own window of copied vectors. The 1-worker row is the other half of the
# reading: it matches the serial time, so splitting the groups costs nothing when it cannot
# be exploited. Deliberately NOT a `Settings` knob: it describes these statement shapes, not
# the deployment, and nothing an operator can measure would tell them what to set it to.
CLONE_WORKERS = 4

# Which label a cloned edge endpoint should be matched by. A Code node also carries its
# node-type label, and the clone needs the *class* label that has a project index on it.
_EDGE_ENDPOINT_LABELS = ("Code", "CodeChunk", "External")


class ProjectCloneError(RuntimeError):
    """A server-side project clone could not be completed or could not be trusted.

    Raised for both halves of the same failure: the copy itself blew up, or it finished and
    the independent recount disagreed with the source. Callers treat the two identically —
    delete the partial copy and fall back to the client-side path — so they are one class.
    """


# Memgraph's vector_search.search has no filtered variant: it returns the GLOBAL
# top-k across every project sharing the index, and the project filter runs after.
# With several indexed projects a small top_k can be fully consumed by another
# project's nodes, silently starving the query. Over-fetch and truncate post-filter.
_VECTOR_OVERFETCH_FACTOR = 20
_VECTOR_MIN_FETCH = 256
# A label-scoped summary index (directories, modules) holds few nodes across every project,
# so fetching nearly all of it costs little and keeps a big neighbour from starving a small
# project the way the shared index can.
_LEVEL_MIN_FETCH = 2048


def _vector_fetch_k(top_k: int) -> int:
    return max(top_k * _VECTOR_OVERFETCH_FACTOR, _VECTOR_MIN_FETCH)


# ---------------------------------------------------------------------------
# Vector search: surviving Memgraph's post-deletion index lag
# ---------------------------------------------------------------------------

# Memgraph's vector indexes keep holding entries for nodes a DETACH DELETE removed until
# its garbage collector catches up — about 20 seconds on 3.12. Inside that window EVERY
# `vector_search.search` fails outright with "Trying to get a property from a deleted
# object": not just a search for the deleted project, but every search, for every project.
# Deleting a project is no longer only a deliberate user action — the worktree janitor
# issues delete_worktree_project on its own schedule — so an agent's search can start
# failing for a reason nothing in its own request explains, and the engine's bare message
# suggests neither a cause nor that waiting fixes it.
#
# The retry is deliberately narrow: this one error only (matched on the engine's message,
# since it arrives as a generic Memgraph.ClientError with no distinguishing code), a budget
# far shorter than the GC cycle so a tool call fails fast rather than hanging, and any other
# failure propagates untouched on the first attempt. A slow-but-correct search must not be
# turned into an eight-second stall, and a syntax error or a dead connection must not be
# retold as a deletion story.
_DELETED_OBJECT_MARKER = "deleted object"
_VECTOR_RETRY_BUDGET_S = 8.0
_VECTOR_RETRY_BASE_DELAY_S = 0.25
_VECTOR_RETRY_MAX_DELAY_S = 2.0

# Written for a calling agent, which sees this string and nothing else: it has to name the
# cause, say the wait is bounded, and say that re-indexing would be wasted effort.
_STALE_VECTOR_INDEX_MESSAGE = (
    "Vector search is temporarily unavailable: Memgraph's vector index still holds entries "
    "for nodes removed by a recent project deletion (an explicit delete, or the worktree "
    "janitor's sweep). Until its garbage collector clears them, every vector search fails — "
    "for every project, not only the deleted one. Retried for {waited:.1f}s without "
    "recovering; retry the same search in ~20 seconds. Nothing is corrupted and no "
    "re-indexing is needed. Underlying engine error: {original}"
)


class StaleVectorIndexError(RuntimeError):
    """Vector search kept failing on Memgraph's post-deletion vector-index lag.

    Carries the explanation a caller can act on; the raw engine error is the ``__cause__``.
    """


async def _vector_search_with_retry(
    run: Callable[[], Awaitable[list[Mapping[str, object]]]],
) -> list[Mapping[str, object]]:
    """Run a vector search, retrying only Memgraph's post-deletion "deleted object" error.

    Returns the search result as soon as one succeeds. Re-raises anything else immediately,
    and raises :class:`StaleVectorIndexError` if the budget runs out while the index is
    still stale.
    """
    started = monotonic()
    deadline = started + _VECTOR_RETRY_BUDGET_S
    delay = _VECTOR_RETRY_BASE_DELAY_S
    attempts = 0
    while True:
        attempts += 1
        try:
            return await run()
        except Exception as exc:  # noqa: BLE001 — re-raised below unless it is *that* error
            if _DELETED_OBJECT_MARKER not in str(exc):
                raise
            remaining = deadline - monotonic()
            if remaining <= 0:
                waited = monotonic() - started
                logger.warning(
                    "Vector search still blocked by a stale vector index after %d attempt(s)/%.1fs",
                    attempts, waited,
                )
                raise StaleVectorIndexError(
                    _STALE_VECTOR_INDEX_MESSAGE.format(waited=waited, original=exc)
                ) from exc
            logger.info(
                "Vector search hit a stale vector index (attempt %d), retrying in %.2fs",
                attempts, min(delay, remaining),
            )
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _VECTOR_RETRY_MAX_DELAY_S)


def _relative_tree_path(path: str) -> str:
    """Normalize a caller-typed project-relative path to the form the graph stores.

    Directory nodes store ``src/pkg`` and the root stores ``.``; agents type ``src/pkg/``,
    ``./src/pkg`` and ``./`` just as often, and each of those used to miss.
    """
    cleaned = path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.rstrip("/")
    return cleaned or "."


def _written(rows: Sequence[Mapping[str, object]]) -> int:
    """Read the ``written`` count a batched write's ``RETURN count(*)`` produced.

    A write with ``RETURN count(*) AS written`` reports the rows the DB actually matched
    (and therefore wrote), which is < the rows submitted when an id doesn't resolve to a
    node. This makes "Stored N" mean N rows in the DB, not N rows sent (INV-7, finding 14).
    """
    if rows and rows[0].get("written") is not None:
        return int(rows[0]["written"])  # type: ignore[arg-type]
    return 0


def _node_type_from_labels(labels: object) -> str | None:
    if not isinstance(labels, (list, tuple, set)):
        return None
    names = {str(label) for label in labels}
    for label in names:
        node_type = _NODE_TYPE_VALUE_BY_LABEL.get(label)
        if node_type:
            return node_type
    # A dependency node carries no NodeType — it is not code this project contains. Reported
    # under its own name so a caller sees "external" rather than a typeless entity; whether
    # it is an imported module or an inherited base class is already carried by the relation
    # (IMPORTS vs INHERITS) on the same row.
    if "External" in names:
        return "external"
    return None


def _code_node_from_row(row: Mapping[str, object]) -> CodeNode | None:
    """Rebuild a CodeNode from a stored-node query row; None if the row is unusable."""
    type_value = row.get("node_type") or _node_type_from_labels(row.get("labels"))
    if not row.get("id") or not type_value:
        return None
    try:
        node_type = NodeType(str(type_value))
    except ValueError:
        return None
    return CodeNode(
        id=str(row["id"]),
        name=str(row.get("name") or ""),
        node_type=node_type,
        file_path=str(row.get("file_path") or ""),
        line_start=int(row.get("line_start") or 1),
        line_end=int(row.get("line_end") or 1),
        signature=str(row.get("signature") or ""),
        docstring=str(row.get("docstring") or ""),
        summary=str(row.get("summary") or ""),
        summary_input_hash=str(row.get("summary_input_hash") or ""),
    )


def _serialize_entity(raw: Mapping[str, object] | None, prefix: str = "") -> dict[str, object] | None:
    if raw is None:
        return None
    p = f"{prefix}." if prefix else ""
    return {
        "id": raw.get(f"{p}id"),
        "name": raw.get(f"{p}name"),
        "node_type": _node_type_from_labels(raw.get(f"{p}labels")),
        "file_path": raw.get(f"{p}file_path"),
        "line_start": raw.get(f"{p}line_start"),
        "line_end": raw.get(f"{p}line_end"),
        "summary": raw.get(f"{p}summary"),
    }


class GraphRepository:
    """Async CRUD, vector, and graph-query operations against Memgraph."""

    def __init__(self, conn: MemgraphConnection, batch_size: int = DEFAULT_WRITE_BATCH_SIZE) -> None:
        self._conn = conn
        self._batch_size = max(1, batch_size)

    # ------------------------------------------------------------------
    # Node / edge CRUD
    # ------------------------------------------------------------------

    async def upsert_nodes(self, project: str, nodes: Sequence[CodeNode]) -> int:
        """Upsert Code nodes in UNWIND batches. Returns number of nodes upserted.

        Nodes are grouped by label (labels can't be parameterized) and each group is
        written batch_size rows per query, so a large repo costs ~N/batch_size round
        trips instead of one per node, and each batch commits incrementally.
        """
        by_label: dict[str, list[CodeNode]] = defaultdict(list)
        for node in nodes:
            by_label[_NODE_TYPE_LABEL_BY_VALUE[node.node_type.value]].append(node)

        total = sum(len(group) for group in by_label.values())
        done = 0
        for label, group in by_label.items():
            query = queries.UPSERT_NODES_BATCH_TEMPLATE.format(node_label=label)
            for start in range(0, len(group), self._batch_size):
                chunk = group[start : start + self._batch_size]
                rows = [
                    {
                        "id": node.id,
                        "node_type": node.node_type.value,
                        "name": node.name,
                        "file_path": node.file_path,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                        "signature": node.signature,
                        "docstring": node.docstring,
                        "summary": node.summary,
                        "summary_input_hash": node.summary_input_hash,
                        "parent_id": node.parent_id,
                        "external_calls": node.external_calls,
                        "unresolved_calls": node.unresolved_calls,
                        "calls_provenance": node.calls_provenance,
                    }
                    for node in chunk
                ]
                await self._conn.execute_write(query, {"project": project, "rows": rows})
                done += len(chunk)
                logger.info("Upserted %d/%d nodes for project=%s", done, total, project)
        return done

    async def upsert_edges(self, project: str, edges: Sequence[Edge]) -> int:
        """Upsert edges in UNWIND batches (grouped by relationship type).

        Returns the number of edges the DB actually wrote — an edge whose source or
        target id doesn't resolve to a node (a common case for unresolved CALLS targets)
        is silently dropped by the MATCHes, so the count reflects the graph, not the
        submission. A large shortfall is logged as ERROR (INV-7, audit finding 14).
        """
        by_type: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            by_type[edge.edge_type.value.upper()].append(edge)

        submitted = sum(len(group) for group in by_type.values())
        written = 0
        for edge_type, group in by_type.items():
            if edge_type == "CALLS":
                query = _UPSERT_CALLS_EDGES_TEMPLATE
            else:
                query = _UPSERT_EDGES_BATCH_TEMPLATE.format(edge_type=edge_type)
            for start in range(0, len(group), self._batch_size):
                chunk = group[start : start + self._batch_size]
                rows = [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "resolution": edge.resolution,
                        "callee_text": edge.callee_text,
                    }
                    for edge in chunk
                ]
                result = await self._conn.execute_write_query(query, {"project": project, "rows": rows})
                written += _written(result)
                logger.info("Upserted %d/%d edges for project=%s", written, submitted, project)
        if written < submitted:
            logger.error(
                "Edge upsert wrote %d of %d edges for project=%s — %d edge(s) had an "
                "unresolved endpoint at write time",
                written, submitted, project, submitted - written,
            )
        return written

    async def upsert_external_nodes(self, project: str, refs: Sequence[ExternalRef]) -> int:
        """Upsert the project's dependencies as :External nodes.

        Labelled apart from :Code on purpose — every other query here is :Code-scoped, so
        these stay out of the entity listing, the counts, search and summarization while
        remaining traversable and reachable from a project-scoped cypher_query.
        """
        written = 0
        for start in range(0, len(refs), self._batch_size):
            chunk = refs[start : start + self._batch_size]
            rows = [
                {"id": r.id, "name": r.name, "root": r.root, "kind": r.kind, "language": r.language}
                for r in chunk
            ]
            result = await self._conn.execute_write_query(
                queries.UPSERT_EXTERNAL_NODES, {"project": project, "rows": rows}
            )
            written += _written(result)
        logger.info("Upserted %d external dependency node(s) for project=%s", written, project)
        return written

    async def delete_orphan_external_nodes(self, project: str) -> int:
        """Remove :External nodes nothing points at any more (a dropped dependency)."""
        result = await self._conn.execute_write_query(
            queries.DELETE_ORPHAN_EXTERNAL_NODES, {"project": project}
        )
        deleted = int(result[0]["deleted"]) if result else 0
        if deleted:
            logger.info("Removed %d orphaned external dependenc(ies) for project=%s", deleted, project)
        return deleted

    async def delete_project(self, project: str) -> None:
        """Delete all graph and vector data for a project (including any clone marker)."""
        await self._conn.execute_write(queries.DELETE_PROJECT, {"project": project})
        logger.debug("Deleted all nodes for project=%s", project)

    # ------------------------------------------------------------------
    # Server-side project clone
    # ------------------------------------------------------------------

    async def get_clone_marker(self, project: str) -> str | None:
        """The source project of a clone that never finished, or ``None``.

        A non-``None`` answer means this project holds a partial copy of the named project
        and must be deleted before being (re-)indexed — see :meth:`clone_project` for the
        contract. Cheap: the marker label holds one node per clone in flight.
        """
        records = await self._conn.execute_query(queries.GET_CLONE_MARKER, {"project": project})
        if not records:
            return None
        source = records[0].get("source")
        return str(source) if source else None

    async def clone_project(self, source: str, target: str) -> dict[str, int]:
        """Copy every node, chunk, dependency and edge of *source* into *target*, inside the DB.

        Returns ``{"nodes", "chunks", "externals", "edges"}`` — the counts a *re-read* of the
        target found, not the counts the writes claimed.

        Why this exists: the client-side alternative pulls every summary and every 1536-float
        vector out of Memgraph and writes them all back under a new project id. The values are
        identical on both ends of that trip, so the trip is pure cost — minutes on a repository
        of a few thousand nodes, nearly all of it spent moving vectors the database already
        has. Here nothing but ids and counts crosses the wire.

        **Concurrency.** The node half runs on up to :data:`CLONE_WORKERS` connections at
        once: each node-type label, the chunks and the externals are independent writes, and
        each is further split into disjoint id ranges (:meth:`_clone_shards`) — 33.6s to
        13.6s on an 8.1k-chunk project, measured. The **edge
        phase starts only after every node statement has finished** — an edge is created by
        re-finding both endpoints by id *in the target*, so a relationship copied while its
        endpoint was still being written would simply not be created, and the recount below
        would refuse the clone. If any node statement fails, its siblings are cancelled and
        awaited before the error propagates, so no straggler is still writing into the target
        when the caller deletes the half-clone.

        **Atomicity contract.** There is no transaction spanning the whole copy, so the
        interruption story is explicit instead of hoped for:

        1. The target must be empty. A non-empty target raises rather than duplicating nodes.
        2. A ``:CloneMarker`` is written *before* the first node. While it is present
           :meth:`project_is_indexed` reports the project as **not indexed**, so no read tool
           can serve a half-copied graph, and ``ProjectManager.index_directory`` deletes the
           project and clones it again instead of updating it.
        3. The cloned root node's borrowed identity (the donor's ``root_path`` and scope) is
           erased immediately, so a half-clone is invisible to ``list_projects`` — the janitor
           and the dashboard never see a phantom project rooted at the donor's directory.
        4. The copy is verified by independent recounts of both projects (per label, per
           relationship type). Memgraph 3.12 is known to write fewer rows than its own
           ``count(*)`` reports for some shapes, so the writes' own numbers are not evidence.
        5. Only then is the marker removed. The caller stamps this project's real
           ``root_path``/scope/worktree marker afterwards.

        Anything that goes wrong — a failed statement or a recount that disagrees — raises
        :class:`ProjectCloneError` with the target left marked and therefore un-queryable.
        The caller is expected to delete the partial copy (``delete_project``) and fall back;
        if the process dies before that, step 2 makes the next index run do it.
        """
        existing = await self._count_one(queries.COUNT_CODE_NODES, target)
        existing += await self._count_one(queries.COUNT_CHUNKS, target)
        existing += await self._count_one(queries.COUNT_EXTERNALS, target)
        if existing:
            raise ProjectCloneError(
                f"Refusing to clone project={source} into project={target}: the target already "
                f"holds {existing} node(s). Cloning into a non-empty project would duplicate "
                "every node, since the copy CREATEs rather than MERGEs. Delete it first."
            )

        started = monotonic()
        await self._conn.execute_write(
            queries.SET_CLONE_MARKER, {"project": target, "source": source}
        )
        try:
            # (statement, label to shard by) for every independent node-ish copy.
            groups: list[tuple[str, str]] = [
                (queries.CLONE_NODES_TEMPLATE.format(node_label=label), f"Code:{label}")
                for label in _NODE_TYPE_LABEL_BY_VALUE.values()
            ]
            groups.append((queries.CLONE_CHUNKS, "CodeChunk"))
            groups.append((queries.CLONE_EXTERNALS, "External"))
            written = await self._clone_groups(source, target, groups)
            # Positional: the chunk and external groups were appended last, in that order.
            nodes = sum(written[:-2])
            chunks, externals = written[-2], written[-1]
            # Before any verification, so an interrupted clone cannot masquerade as a project
            # rooted at the donor's directory (contract step 3).
            await self._conn.execute_write(
                queries.CLEAR_CLONED_ROOT_IDENTITY, {"project": target}
            )
            # Every node exists by now — `_clone_groups` does not return until all of its
            # statements have completed — so the edge phase can re-find both endpoints.
            edges = await self._clone_edges(source, target)
            counts = await self._verify_clone(source, target)
        except ProjectCloneError:
            raise
        except Exception as exc:
            raise ProjectCloneError(
                f"Server-side clone of project={source} into project={target} failed: {exc}"
            ) from exc

        await self._conn.execute_write(queries.CLEAR_CLONE_MARKER, {"project": target})
        logger.info(
            "Cloned project=%s into project=%s in %.1fs (%d nodes, %d chunks, %d externals, "
            "%d edges; write counts: %d/%d/%d/%d)",
            source, target, monotonic() - started,
            counts["nodes"], counts["chunks"], counts["externals"], counts["edges"],
            nodes, chunks, externals, edges,
        )
        return counts

    async def _count_one(self, query: str, project: str) -> int:
        records = await self._conn.execute_query(query, {"project": project})
        return int(records[0]["count"]) if records else 0

    async def _edge_counts(self, project: str) -> dict[str, int]:
        records = await self._conn.execute_query(queries.COUNT_EDGES_BY_TYPE, {"project": project})
        return {str(r["type"]): int(r["count"]) for r in records}

    async def _clone_groups(
        self, source: str, target: str, groups: Sequence[tuple[str, str]]
    ) -> list[int]:
        """Copy every *(statement, label)* group concurrently. Returns rows written per group.

        Each group is first split into disjoint id ranges (:meth:`_clone_shards`) and every
        range becomes its own keyset-paginated worker, so a project whose nodes sit almost
        entirely under one label parallelises just as well as one spread over ten. At most
        :data:`CLONE_WORKERS` statements are ever in flight.

        The whole set is run inside a task group: the first failure cancels its siblings and
        **waits for them**, then propagates unwrapped, so the caller's "delete the half-clone"
        recovery can never race a statement that is still writing into the target. The result
        is positional — ``written[i]`` belongs to ``groups[i]`` — because the caller reports
        node, chunk and external counts separately.
        """
        semaphore = asyncio.Semaphore(CLONE_WORKERS)

        async def shards_for(label: str) -> list[tuple[str, str]]:
            # Under the same limit as the copies: splitting a group costs a sorted label scan
            # that materialises its ids server-side, and a dozen of those at once would spike
            # Memgraph's memory for no gain.
            async with semaphore:
                return await self._clone_shards(source, label)

        shard_lists = await asyncio.gather(*(shards_for(label) for _query, label in groups))
        written = [0] * len(groups)

        async def run(index: int, query: str, after: str, upper: str) -> None:
            async with semaphore:
                rows = await self._clone_keyset(query, source, target, after=after, upper=upper)
            # Only mutated after the await, and asyncio gives us no preemption inside this
            # statement, so a plain += is safe across the concurrent workers.
            written[index] += rows

        try:
            async with asyncio.TaskGroup() as group:
                for index, ((query, _label), shards) in enumerate(zip(groups, shard_lists, strict=True)):
                    for after, upper in shards:
                        group.create_task(run(index, query, after, upper))
        except BaseExceptionGroup as failures:
            # Re-raise the first real failure with its own type, so a lost cursor still
            # arrives at the caller as ProjectCloneError rather than as an ExceptionGroup.
            raise failures.exceptions[0] from failures
        return written

    async def _clone_shards(self, source: str, label: str) -> list[tuple[str, str]]:
        """Split one label group into up to :data:`CLONE_WORKERS` ``(after, upper]`` id ranges.

        ``''`` means unbounded, matching the keyset cursor's own convention (every id sorts
        above the empty string). The last range is always left open-ended, so anything written
        to the source after the boundaries were sampled is still copied — the copy stays a
        superset of the sample, and the recount, not the sample, decides whether it was exact.
        """
        records = await self._conn.execute_query(
            queries.CLONE_SHARD_BOUNDARIES_TEMPLATE.format(label=label),
            {"source": source, "shards": CLONE_WORKERS},
        )
        raw = list(records[0].get("boundaries") or []) if records else []
        boundaries: list[str] = []
        for value in raw:
            # Defensive: a null or a duplicate boundary would produce an empty or overlapping
            # range. Empty is merely wasteful; overlapping would duplicate nodes.
            text = str(value) if value is not None else ""
            if text and (not boundaries or text > boundaries[-1]):
                boundaries.append(text)

        shards: list[tuple[str, str]] = []
        after = ""
        for boundary in boundaries:
            shards.append((after, boundary))
            after = boundary
        shards.append((after, ""))
        return shards

    async def _clone_keyset(
        self, query: str, source: str, target: str, after: str = "", upper: str = ""
    ) -> int:
        """Run one keyset-paginated clone statement to exhaustion. Returns rows written.

        Copies the ``(after, upper]`` id range only; ``''`` is unbounded on either end, so the
        default arguments reproduce the whole-group behaviour this had before it was sharded.
        """
        written = 0
        while True:
            records = await self._conn.execute_write_query(
                query,
                {
                    "source": source, "target": target,
                    "after": after, "upper": upper, "limit": CLONE_BATCH_SIZE,
                },
            )
            batch = _written(records)
            if not batch:
                return written
            written += batch
            last = records[0].get("last")
            if last is None:
                # Rows were written but no cursor came back — continuing would re-clone the
                # same window forever. Better to fail the clone than to duplicate nodes.
                raise ProjectCloneError(
                    f"Clone of project={source} lost its pagination cursor after {written} row(s)"
                )
            after = str(last)

    async def _clone_edges(self, source: str, target: str) -> int:
        """Clone every relationship of the source, one (type, target label) pair at a time."""
        shapes = await self._conn.execute_query(queries.CLONE_EDGE_SHAPES, {"source": source})
        pairs: set[tuple[str, str]] = set()
        for row in shapes:
            labels = {str(label) for label in (row.get("target_labels") or [])}
            target_label = next((label for label in _EDGE_ENDPOINT_LABELS if label in labels), None)
            if target_label is None:
                # An endpoint the clone has no way to address by label. Not fatal here — the
                # recount below turns it into a refusal, which is the honest outcome.
                logger.error(
                    "Clone of project=%s found a %s edge into a node with labels %s it cannot "
                    "address; the clone will fail verification",
                    source, row.get("type"), sorted(labels),
                )
                continue
            pairs.add((str(row["type"]), target_label))

        written = 0
        after = ""
        while True:
            window = await self._conn.execute_query(
                queries.CLONE_SOURCE_ID_WINDOW,
                {"source": source, "after": after, "limit": CLONE_BATCH_SIZE},
            )
            if not window:
                return written
            upper = str(window[-1]["id"])
            for edge_type, target_label in sorted(pairs):
                query = queries.CLONE_EDGES_TEMPLATE.format(
                    edge_type=edge_type, target_label=target_label
                )
                records = await self._conn.execute_write_query(
                    query,
                    {
                        "source": source, "target": target,
                        "after": after, "upper": upper,
                    },
                )
                written += _written(records)
            after = upper

    async def _verify_clone(self, source: str, target: str) -> dict[str, int]:
        """Recount both projects independently; raise unless every count matches."""
        counts: dict[str, int] = {}
        mismatches: list[str] = []
        for key, query in (
            ("nodes", queries.COUNT_CODE_NODES),
            ("chunks", queries.COUNT_CHUNKS),
            ("externals", queries.COUNT_EXTERNALS),
        ):
            src = await self._count_one(query, source)
            dst = await self._count_one(query, target)
            counts[key] = dst
            if src != dst:
                mismatches.append(f"{key}: source={src} target={dst}")

        src_edges = await self._edge_counts(source)
        dst_edges = await self._edge_counts(target)
        counts["edges"] = sum(dst_edges.values())
        for edge_type in sorted(set(src_edges) | set(dst_edges)):
            src_n = src_edges.get(edge_type, 0)
            dst_n = dst_edges.get(edge_type, 0)
            if src_n != dst_n:
                mismatches.append(f"{edge_type} edges: source={src_n} target={dst_n}")

        if mismatches:
            raise ProjectCloneError(
                f"Clone of project={source} into project={target} is incomplete — "
                + "; ".join(mismatches)
            )
        return counts

    async def get_stored_chunk_embeddings_for_files(
        self, project: str, file_paths: Sequence[str]
    ) -> dict[str, list[float]]:
        """``{chunk_id: embedding}`` for the given files only (null embeddings dropped).

        The scoped counterpart to :meth:`get_stored_chunk_embeddings`, for callers that know
        which files they are about to re-chunk. Reuse is never lost by scoping: a chunk id
        hashes the owning node and file path, so a chunk of one file can never be reused for
        another.
        """
        file_paths = list(file_paths)
        if not file_paths:
            return {}
        records = await self._conn.execute_query(
            queries.GET_STORED_CHUNK_EMBEDDINGS_FOR_FILES,
            {"project": project, "file_paths": file_paths},
        )
        return {r["chunk_id"]: r["embedding"] for r in records if r["embedding"] is not None}

    async def get_stored_chunk_embeddings_for_ids(
        self, project: str, chunk_ids: Sequence[str]
    ) -> dict[str, list[float]]:
        """``{chunk_id: embedding}`` for the named chunks only (null embeddings dropped).

        The narrowest of the three reads: used to recover the vector of a chunk that is
        stored but has to be re-written anyway, so that re-writing it never costs a paid
        embedding call (see :meth:`RAGIndexer.index_code`).
        """
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return {}
        found: dict[str, list[float]] = {}
        for start in range(0, len(chunk_ids), self._batch_size):
            records = await self._conn.execute_query(
                queries.GET_STORED_CHUNK_EMBEDDINGS_FOR_IDS,
                {"project": project, "ids": chunk_ids[start : start + self._batch_size]},
            )
            found.update(
                {r["chunk_id"]: r["embedding"] for r in records if r["embedding"] is not None}
            )
        return found

    async def get_stored_chunk_facts(
        self, project: str, chunk_ids: Sequence[str]
    ) -> dict[str, dict[str, object]]:
        """``{chunk_id: {"language": str, "node_ids": set[str]}}`` for chunks already stored
        *with* an embedding.

        Answers "does the database already hold this exact chunk?" without moving a single
        vector: an id match already proves the node, file, line span, kind and content equal
        (they are all hashed into the id), so only the language and the owning node's id are
        left to compare — see ``queries.GET_STORED_CHUNK_FACTS`` for why those two, and why a
        vector-less chunk is deliberately reported as absent.

        Scoped by id and batched, never a project scan: an incremental update must cost in
        proportion to what it changed, not to how big the project is. A chunk id that does not
        appear in the result has no usable stored row and must be written.
        """
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return {}
        facts: dict[str, dict[str, object]] = {}
        for start in range(0, len(chunk_ids), self._batch_size):
            records = await self._conn.execute_query(
                queries.GET_STORED_CHUNK_FACTS,
                {"project": project, "ids": chunk_ids[start : start + self._batch_size]},
            )
            for record in records:
                chunk_id = record.get("id")
                if not chunk_id:
                    continue
                entry = facts.setdefault(
                    str(chunk_id), {"language": str(record.get("language") or ""), "node_ids": set()}
                )
                node_id = record.get("node_id")
                if node_id:
                    # A chunk should have exactly one owner, but the read does not assume it:
                    # the writer only needs to know whether *its* node is among them.
                    entry["node_ids"].add(str(node_id))  # type: ignore[union-attr]
        return facts

    async def get_node_ids(self, project: str) -> set[str]:
        """Return the ids of all stored Code nodes for a project."""
        records = await self._conn.execute_query(queries.GET_NODE_IDS, {"project": project})
        return {r["id"] for r in records}

    async def get_stored_node_index(self, project: str) -> list[tuple[str, str, str]]:
        """Return ``(id, file_path, node_type)`` for every stored Code node.

        Lets a reconcile scope deletion to files it actually re-processed (parsed
        clean) or confirmed gone from disk, instead of deleting every id absent from a
        scan that a missing language parser or a read failure silently truncated.
        """
        records = await self._conn.execute_query(queries.GET_NODE_INDEX, {"project": project})
        index: list[tuple[str, str, str]] = []
        for r in records:
            node_id = r.get("id")
            if not node_id:
                continue
            node_type = r.get("node_type") or _node_type_from_labels(r.get("labels")) or ""
            index.append((str(node_id), str(r.get("file_path") or ""), str(node_type)))
        return index

    async def get_chunk_ids(self, project: str) -> set[str]:
        """Return the ids of all stored CodeChunk nodes for a project."""
        records = await self._conn.execute_query(queries.GET_CHUNK_IDS, {"project": project})
        return {r["id"] for r in records}

    async def get_chunk_index(self, project: str) -> dict[str, str]:
        """Return ``{chunk_id: file_path}`` for all stored CodeChunk nodes."""
        records = await self._conn.execute_query(queries.GET_CHUNK_INDEX, {"project": project})
        return {r["id"]: str(r.get("file_path") or "") for r in records if r.get("id")}

    async def get_code_edges(self, project: str) -> list[tuple[str, str, str]]:
        """Return ``(source_id, TYPE, target_id)`` for every Code->Code relationship."""
        records = await self._conn.execute_query(queries.GET_CODE_EDGES, {"project": project})
        return [(str(r["source"]), str(r["type"]), str(r["target"])) for r in records]

    async def delete_nodes_by_ids(self, project: str, ids: Sequence[str]) -> int:
        """Delete the given Code nodes (DETACH). Returns the count submitted."""
        ids = list(ids)
        for start in range(0, len(ids), self._batch_size):
            batch = ids[start : start + self._batch_size]
            await self._conn.execute_write(
                queries.DELETE_NODES_BY_IDS, {"project": project, "ids": batch}
            )
        if ids:
            logger.info("Deleted %d stale node(s) for project=%s", len(ids), project)
        return len(ids)

    async def delete_chunks_by_ids(self, project: str, ids: Sequence[str]) -> int:
        """Delete the given CodeChunk nodes (DETACH). Returns the count submitted."""
        ids = list(ids)
        for start in range(0, len(ids), self._batch_size):
            batch = ids[start : start + self._batch_size]
            await self._conn.execute_write(
                queries.DELETE_CHUNKS_BY_IDS, {"project": project, "ids": batch}
            )
        if ids:
            logger.info("Deleted %d stale chunk(s) for project=%s", len(ids), project)
        return len(ids)

    async def delete_code_edges(self, project: str) -> None:
        """Delete all Code->Code relationships for a project (HAS_CHUNK links survive)."""
        await self._conn.execute_write(queries.DELETE_CODE_EDGES, {"project": project})
        logger.debug("Deleted code edges for project=%s", project)

    async def delete_edges_by_keys(
        self, project: str, edges: Sequence[Mapping[str, str]]
    ) -> int:
        """Delete specific Code->Code relationships identified by (source, type, target).

        The targeted counterpart to the wipe-all :meth:`delete_code_edges`: an update
        MERGEs the fresh edges first, then removes only these stale ones, so the graph
        never passes through an edgeless state and a crash leaves edges no worse than
        stale. Returns the number of edge keys submitted.
        """
        edges = list(edges)
        for start in range(0, len(edges), self._batch_size):
            batch = edges[start : start + self._batch_size]
            await self._conn.execute_write(
                queries.DELETE_EDGES_BY_KEYS,
                {"project": project, "rows": [dict(edge) for edge in batch]},
            )
        if edges:
            logger.info("Deleted %d stale edge(s) for project=%s", len(edges), project)
        return len(edges)

    # ------------------------------------------------------------------
    # Checksums
    # ------------------------------------------------------------------

    async def get_checksums(self, project: str) -> dict[str, str]:
        """Return {file_path: md5} for all modules in a project."""
        records = await self._conn.execute_query(
            queries.GET_CHECKSUMS,
            {"project": project},
        )
        return {r["file_path"]: r["checksum"] for r in records if r["checksum"]}

    async def project_is_indexed(self, project: str) -> bool:
        """True when the project has at least one indexed node (cheap, LIMIT 1 probe).

        Distinguishes a never-indexed project (typo'd/adjacent directory → valid-but-empty
        project id) from an indexed one with no matches, so tools stop "politely lying" with
        an empty page (audit finding T5 / INV-7).

        A project holding an unfinished server-side clone is reported as **not** indexed even
        though it has nodes: they are a partial copy of another project (see
        :meth:`clone_project`). Read tools therefore refuse it rather than serving half a
        graph, and the auto-index path they fall into is what repairs it.
        """
        records = await self._conn.execute_query(queries.PROJECT_IS_INDEXED, {"project": project})
        return bool(records)

    async def get_project_stats(self, project: str) -> dict[str, int]:
        """Return persisted graph/vector counts for a project."""
        records = await self._conn.execute_query(
            queries.GET_PROJECT_STATS,
            {"project": project},
        )
        if not records:
            return {"nodes": 0, "edges": 0, "chunks": 0, "summaries": 0}
        record = records[0]
        return {
            "nodes": int(record["nodes"] or 0),
            "edges": int(record["edges"] or 0),
            "chunks": int(record["chunks"] or 0),
            "summaries": int(record["summaries"] or 0),
        }

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        """Record the absolute root directory on the project's root node."""
        await self._conn.execute_write(
            queries.SET_PROJECT_ROOT_PATH,
            {"project": project, "root_path": root_path},
        )

    async def get_watch_state(self, project: str) -> dict[str, Any]:
        """Is anything keeping this project's index current? ``{watched, live, holder, daemon_live}``.

        ``live`` is about *this project*: a daemon holds its claim and re-indexes its saves.
        ``daemon_live`` is about the machine: a watch daemon is running at all. They are not the
        same question and conflating them produced advice that contradicted itself — "no daemon
        is running, start one" printed by a machine whose daemon was running fine and had simply
        not reached this project yet (it re-reads the registry every ``WATCH_REFRESH_SECONDS``),
        or had deliberately skipped it. With both, a caller can tell "nothing is running" from
        "running, but not following this one", and only the first is fixed by starting a daemon.

        Liveness on either is the claim's heartbeat, not the mere existence of a lease node: a
        daemon killed with SIGKILL leaves its claims behind to expire.

        Never raises. A caller is typically annotating an answer it has already computed, and
        an unreadable lock node is not a reason to fail their search.
        """
        blank = {"watched": False, "live": False, "holder": None, "daemon_live": False}
        try:
            records = await self._conn.execute_query(
                queries.GET_WATCH_STATE,
                {
                    "project": project,
                    "claim_key": lease.watch_claim_key(project),
                    "daemon_key": lease.daemon_claim_key(),
                },
            )
        except Exception:  # noqa: BLE001 — a status read must never break the caller's answer
            logger.debug("Could not read watch state for project=%s", project, exc_info=True)
            return blank
        if not records:
            return blank
        row = records[0]
        now_ms = int(time() * 1000)
        holder = row.get("holder")
        daemon_holder = row.get("daemon_holder")
        return {
            "watched": bool(row.get("watched")),
            "live": bool(holder) and int(row.get("expires_at") or 0) > now_ms,
            "holder": str(holder) if holder else None,
            "daemon_live": bool(daemon_holder)
            and int(row.get("daemon_expires_at") or 0) > now_ms,
        }

    async def set_project_watched(self, project: str, watched: bool) -> bool:
        """Add or remove a project from the watch registry; returns whether a project matched.

        The registry is what ``inflorescence watch`` reads to decide which trees to follow, so
        the write lands on the project root node — a node that exists only for an *indexed*
        project. That is the intended gate rather than an accident: the daemon updates an
        existing graph and is not allowed to conjure one (a first index costs real money and
        must stay an explicit, previewed act), so a path with no index cannot be watched, and
        ``False`` here is exactly that refusal, reported to the caller instead of silently
        registering a project nothing will ever update.
        """
        records = await self._conn.execute_write_query(
            queries.SET_PROJECT_WATCHED, {"project": project, "watched": bool(watched)}
        )
        return bool(records and int(records[0]["matched"] or 0))

    async def set_project_scope(
        self, project: str, include: Sequence[str], exclude: Sequence[str]
    ) -> None:
        """Persist the include/exclude scan scope of an explicit index on the root node.

        A background watcher update reads this back (see :meth:`get_project_scope`) and
        reproduces the same scan, so it never widens the scanned universe beyond what the
        user indexed (audit finding 5c). Empty lists mean "no extra restriction".
        """
        await self._conn.execute_write(
            queries.SET_PROJECT_SCOPE,
            {"project": project, "include": list(include), "exclude": list(exclude)},
        )

    async def get_project_scope(self, project: str) -> tuple[list[str], list[str]]:
        """Return the persisted ``(include, exclude)`` scan scope for a project.

        Both lists are empty when the project predates scope persistence or was indexed
        with no explicit include/exclude — the caller then falls back to config defaults.
        """
        records = await self._conn.execute_query(queries.GET_PROJECT_SCOPE, {"project": project})
        if not records:
            return [], []
        row = records[0]
        include = [str(p) for p in (row.get("include") or [])]
        exclude = [str(p) for p in (row.get("exclude") or [])]
        return include, exclude

    async def set_project_worktree_meta(
        self, project: str, worktree_of: str, main_root_path: str
    ) -> None:
        """Mark this project as the index of a linked git worktree of *worktree_of*.

        Stamped on every index of a directory detection identified as a worktree, whether or
        not the donor project turned out to be indexed: the marker records a fact about the
        directory, and a future janitor's safety depends on that fact being complete — a
        worktree index it cannot recognise is one it would either keep forever or, worse,
        confuse with a standalone project. Ordinary projects never carry it.
        """
        await self._conn.execute_write(
            queries.SET_PROJECT_WORKTREE_META,
            {"project": project, "worktree_of": worktree_of, "main_root_path": main_root_path},
        )

    async def get_project_worktree_meta(self, project: str) -> dict[str, str] | None:
        """Return ``{"worktree_of", "main_root_path"}`` for a worktree project, else ``None``.

        ``None`` covers both "not a worktree" and "root node absent" (never indexed), because
        neither gives a caller anything to act on. A half-written marker — the donor id present
        but the path missing, or vice versa — is also reported as absent rather than handed on:
        a partial answer here would send a janitor to look up the wrong tree.
        """
        records = await self._conn.execute_query(
            queries.GET_PROJECT_WORKTREE_META, {"project": project}
        )
        if not records:
            return None
        row = records[0]
        worktree_of = row["worktree_of"]
        main_root_path = row["main_root_path"]
        if not worktree_of or not main_root_path:
            return None
        return {"worktree_of": str(worktree_of), "main_root_path": str(main_root_path)}

    async def list_projects(self) -> list[dict[str, Any]]:
        """Return one row per indexed project: identity, root path, worktree marker, watch flag.

        ``worktree_of`` / ``main_root_path`` are ``None`` for an ordinary project — the marker
        is written only for a directory proved to be a linked worktree. ``watched`` is the
        watch-daemon registry flag (:meth:`set_project_watched`), ``False`` for every project
        never registered. Read from the project root nodes, so the cost is one row per project
        rather than a graph scan.
        """
        records = await self._conn.execute_query(queries.LIST_PROJECTS, {})
        projects: list[dict[str, Any]] = []
        for r in records:
            project = r.get("project")
            if not project:
                continue
            projects.append(
                {
                    "project": str(project),
                    "root_path": str(r.get("root_path") or ""),
                    "worktree_of": str(r["worktree_of"]) if r.get("worktree_of") else None,
                    "main_root_path": str(r["main_root_path"]) if r.get("main_root_path") else None,
                    "watched": bool(r.get("watched")),
                }
            )
        return projects

    async def delete_worktree_project(self, project: str) -> bool:
        """Delete a project's index **only** if it is marked as a worktree index. Returns whether it went.

        The guarded counterpart to :meth:`delete_project`, and the only deleter the janitor is
        allowed to call. Deleting a project throws away paid-for LLM summaries and embeddings
        and cannot be undone, so the marker is verified twice and the second check is not
        skippable by any caller: once here, and once *inside the delete statement itself*
        (:data:`queries.DELETE_WORKTREE_PROJECT`), which matches nothing when the root node
        carries no marker. A bug in the caller's candidate logic therefore ends in a refusal
        rather than in an ordinary project losing its index.

        Returns ``False`` — having written nothing — when the project has no marker, has no
        root node at all, or lost its marker between the two checks.
        """
        meta = await self.get_project_worktree_meta(project)
        if meta is None:
            logger.error(
                "Refusing to delete project=%s: it carries no worktree marker. Only an index "
                "created for a linked git worktree may be reaped.",
                project,
            )
            return False

        result = await self._conn.execute_write_query(
            queries.DELETE_WORKTREE_PROJECT, {"project": project}
        )
        if not result:
            # The guard matched nothing: the marker was removed between the read above and
            # this write (a re-index of the same directory as a standalone project). Nothing
            # was deleted, which is the outcome we want.
            logger.warning(
                "Delete of worktree project=%s matched no marked root node — nothing deleted",
                project,
            )
            return False
        deleted = int(result[0].get("deleted") or 0)
        logger.warning(
            "Deleted worktree project=%s (%d node(s)) — was a worktree of project=%s at %s",
            project, deleted, meta["worktree_of"], meta["main_root_path"],
        )
        return True

    async def update_checksums(self, project: str, items: Sequence[Mapping[str, object]]) -> int:
        """Set MD5 checksums on module nodes in UNWIND batches.

        Each item is ``{"node_id": ..., "checksum": ...}``. Returns the number the DB
        actually wrote (matched rows), logging ERROR on a shortfall (INV-7, finding 14).
        """
        items = list(items)
        submitted = len(items)
        written = 0
        for start in range(0, submitted, self._batch_size):
            batch = items[start : start + self._batch_size]
            result = await self._conn.execute_write_query(
                queries.UPDATE_CHECKSUMS_BATCH,
                {"project": project, "rows": [dict(item) for item in batch]},
            )
            written += _written(result)
        if written < submitted:
            logger.error(
                "Checksum update wrote %d of %d rows for project=%s — %d node(s) missing at write time",
                written, submitted, project, submitted - written,
            )
        logger.debug("Updated %d checksums for project=%s", written, project)
        return written

    async def mark_files_chunks_covered(
        self, project: str, items: Sequence[Mapping[str, object]]
    ) -> int:
        """Stamp the chunk-coverage marker on module nodes in UNWIND batches.

        Each item is ``{"file_path", "checksum"}``. Records that every chunk of the file's
        current content is stored, so a later run can tell "fully chunked" from "stale or
        partially chunked" (audit finding 8b). Returns the number of module nodes stamped.
        """
        items = list(items)
        written = 0
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            result = await self._conn.execute_write_query(
                queries.MARK_CHUNKS_COVERED,
                {"project": project, "rows": [dict(item) for item in batch]},
            )
            written += _written(result)
        return written

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        """Return {node_id: {summary, summary_input_hash, summary_embedding, summary_embedding_hash}} for reuse.

        ``summary_embedding_hash`` lets the caller reuse a stored embedding only when it is
        still fresh for the current summary (audit finding 8a).

        Every vector travels over Bolt here, which is only worth paying when the vectors are
        going to be **copied into another project** — the client-side worktree seeding. A
        caller that reuses summaries *within* the same project wants
        :meth:`get_stored_summaries_light` instead: same reuse decisions, none of the payload.
        """
        records = await self._conn.execute_query(queries.GET_STORED_SUMMARIES, {"project": project})
        return {
            r["id"]: {
                "summary": r["summary"],
                "summary_input_hash": r["summary_input_hash"],
                "summary_embedding": r["summary_embedding"],
                "summary_embedding_hash": r["summary_embedding_hash"],
            }
            for r in records
        }

    async def get_stored_summaries_light(self, project: str) -> dict[str, dict[str, object]]:
        """Vector-free summary reuse map: ``{node_id: {summary, summary_input_hash,
        summary_embedding_hash, has_embedding}}``.

        The same reuse decisions as :meth:`get_stored_summaries` at a fraction of the wire
        cost: summarization reuses by ``summary_input_hash``, and the only thing the stored
        *vector* was ever inspected for on this path is whether one exists and is still fresh
        — which ``has_embedding`` plus ``summary_embedding_hash`` answer. Measured on ~3k
        nodes: 2.7s with the vectors, 0.06s without.

        Because the entries carry no ``summary_embedding``, a consumer cannot copy the vector
        anywhere — which is exactly right for a same-project run, where the vector is already
        in place and re-writing it is pure cost (see :meth:`RAGIndexer.index_summaries`).
        Cross-project seeding must keep using :meth:`get_stored_summaries`.
        """
        records = await self._conn.execute_query(
            queries.GET_STORED_SUMMARIES_LIGHT, {"project": project}
        )
        return {
            r["id"]: {
                "summary": r["summary"],
                "summary_input_hash": r["summary_input_hash"],
                "summary_embedding_hash": r["summary_embedding_hash"],
                "has_embedding": bool(r["has_embedding"]),
            }
            for r in records
        }

    async def get_stored_call_facts(
        self, project: str, file_paths: Sequence[str]
    ) -> dict[str, dict[str, object]]:
        """Return ``{node_id: {file_path, external_calls, unresolved_calls, calls_provenance}}``
        for the nodes of *file_paths*.

        The cross-file half of a node row — what the whole-project call resolution decided
        about this node, as opposed to what its own file's bytes say. An incremental update
        uses it to prove that an unchanged file's stored row is already exactly what this run
        would write, and skip the MERGE (see ``GraphBuilder._narrow_writes``). A node id
        missing from the result has no stored row and must always be written.
        """
        file_paths = list(file_paths)
        if not file_paths:
            return {}
        records = await self._conn.execute_query(
            queries.GET_STORED_CALL_FACTS, {"project": project, "file_paths": file_paths}
        )
        return {
            str(r["id"]): {
                "file_path": str(r["file_path"] or ""),
                "external_calls": list(r["external_calls"] or []),
                "unresolved_calls": list(r["unresolved_calls"] or []),
                "calls_provenance": str(r["calls_provenance"] or ""),
            }
            for r in records
            if r.get("id")
        }

    # ------------------------------------------------------------------
    # Repair queries — incomplete state left by a killed index run
    # ------------------------------------------------------------------

    async def get_stale_summary_embedding_nodes(self, project: str) -> list[CodeNode]:
        """Return nodes whose summary embedding is missing OR stale for the current summary.

        Missing = never embedded; stale = the summary was regenerated but its vector still
        reflects the old text (audit finding 8a). The returned nodes carry the current
        ``summary_input_hash`` so re-embedding stamps a fresh coverage marker.
        """
        records = await self._conn.execute_query(queries.GET_STALE_SUMMARY_EMBEDDINGS, {"project": project})
        return [node for r in records if (node := _code_node_from_row(r)) is not None]

    async def get_nodes_missing_summaries(self, project: str) -> list[tuple[CodeNode, int]]:
        """Return (node, child_count) pairs for nodes with no summary at all."""
        records = await self._conn.execute_query(queries.GET_NODES_MISSING_SUMMARIES, {"project": project})
        return [
            (node, int(r["child_count"] or 0))
            for r in records
            if (node := _code_node_from_row(r)) is not None
        ]

    async def get_child_summary_rows(self, project: str, node_id: str) -> list[CodeNode]:
        """Return a container's direct children with their stored summaries, in source order."""
        records = await self._conn.execute_query(
            queries.GET_CHILD_SUMMARY_ROWS, {"project": project, "node_id": node_id}
        )
        return [node for r in records if (node := _code_node_from_row(r)) is not None]

    async def get_files_with_stale_chunks(self, project: str) -> list[str]:
        """Return file paths whose stored chunks are missing, partial, or stale content.

        A file is flagged when its module node's ``chunks_checksum`` coverage marker is
        absent or no longer equals the current file checksum — i.e. the chunks were never
        fully stored (missing / killed mid-window) or belong to old content (finding 8b).
        """
        records = await self._conn.execute_query(queries.GET_FILES_WITH_STALE_CHUNKS, {"project": project})
        return [r["file_path"] for r in records]

    async def get_nodes_missing_contains_edges(self, project: str) -> list[tuple[str, str]]:
        """Return ``(node_id, parent_id)`` for nodes whose parent exists but the CONTAINS edge is gone.

        The structural backbone is reconstructable from ``parent_id``; a run killed before
        re-MERGEing edges can leave the tree edgeless, which no other repair touches
        (audit finding 9).
        """
        records = await self._conn.execute_query(queries.GET_NODES_MISSING_CONTAINS, {"project": project})
        return [
            (str(r["id"]), str(r["parent_id"]))
            for r in records
            if r.get("id") and r.get("parent_id")
        ]

    async def repair_contains_edges(self, project: str, pairs: Sequence[tuple[str, str]]) -> int:
        """Re-MERGE missing CONTAINS edges from ``(child_id, parent_id)`` pairs. Returns count re-linked."""
        pairs = list(pairs)
        written = 0
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start : start + self._batch_size]
            rows = [{"id": cid, "parent_id": pid} for cid, pid in batch]
            result = await self._conn.execute_write_query(
                queries.REPAIR_CONTAINS_EDGES, {"project": project, "rows": rows}
            )
            written += _written(result)
        if pairs:
            logger.info("Repaired %d missing CONTAINS edge(s) for project=%s", written, project)
        return written

    async def get_nodes_for_files(self, project: str, file_paths: Sequence[str]) -> list[CodeNode]:
        """Return all stored nodes for the given file paths."""
        records = await self._conn.execute_query(
            queries.GET_NODES_FOR_FILES, {"project": project, "file_paths": list(file_paths)}
        )
        return [node for r in records if (node := _code_node_from_row(r)) is not None]

    async def store_summaries(self, project: str, items: Sequence[Mapping[str, object]]) -> int:
        """Set summary + summary_input_hash on Code nodes in UNWIND batches.

        Each item is ``{"node_id", "summary", "summary_input_hash"}``. Returns the number
        the DB actually wrote (matched rows), logging ERROR on a shortfall (INV-7, finding 14).
        """
        items = list(items)
        submitted = len(items)
        written = 0
        for start in range(0, submitted, self._batch_size):
            batch = items[start : start + self._batch_size]
            result = await self._conn.execute_write_query(
                queries.STORE_SUMMARIES_BATCH,
                {"project": project, "rows": [dict(item) for item in batch]},
            )
            written += _written(result)
        if written < submitted:
            logger.error(
                "Summary store wrote %d of %d rows for project=%s — %d node(s) missing at write time",
                written, submitted, project, submitted - written,
            )
        logger.debug("Stored %d summaries for project=%s", written, project)
        return written

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    async def get_stored_chunk_embeddings(self, project: str) -> dict[str, list[float]]:
        """Return {chunk_id: embedding} for reuse across rebuilds (null embeddings dropped)."""
        records = await self._conn.execute_query(queries.GET_STORED_CHUNK_EMBEDDINGS, {"project": project})
        return {r["chunk_id"]: r["embedding"] for r in records if r["embedding"] is not None}

    async def store_code_chunks(
        self,
        project: str,
        chunks: Sequence[dict[str, object]],
    ) -> int:
        """Store CodeChunk nodes with embeddings in UNWIND batches.

        Returns the number the DB actually wrote (matched rows) — a chunk whose owner
        Code node is missing is silently dropped by the MATCH, so the count is the truth,
        not the submission. A shortfall is logged as ERROR (INV-7, audit finding 14).
        """
        chunks = list(chunks)
        submitted = len(chunks)
        written = 0
        for start in range(0, submitted, self._batch_size):
            batch = chunks[start : start + self._batch_size]
            result = await self._conn.execute_write_query(
                queries.STORE_CODE_CHUNKS_BATCH,
                {"project": project, "rows": [dict(chunk) for chunk in batch]},
            )
            written += _written(result)
            logger.info("Stored %d/%d code chunks for project=%s", written, submitted, project)
        if written < submitted:
            logger.error(
                "Chunk store wrote %d of %d rows for project=%s — %d chunk(s) had a missing owner node",
                written, submitted, project, submitted - written,
            )
        return written

    async def store_summary_embeddings(
        self,
        project: str,
        items: Sequence[dict[str, object]],
    ) -> int:
        """Set summary_embedding (+ its freshness marker) on Code nodes in UNWIND batches.

        Each item is ``{"node_id", "embedding", "embedding_hash"}``. Returns the number of
        rows the DB actually wrote (matched rows via ``RETURN count(*)``), so a store
        racing a delete — or an unresolved id — reports the shortfall instead of claiming
        success. count(*) is used rather than ``properties_set`` because each row now sets
        two properties (embedding + hash), so a property count would double the truth.
        """
        items = list(items)
        submitted = len(items)
        written = 0
        for start in range(0, submitted, self._batch_size):
            batch = items[start : start + self._batch_size]
            result = await self._conn.execute_write_query(
                queries.STORE_SUMMARY_EMBEDDINGS_BATCH,
                {"project": project, "rows": [dict(item) for item in batch]},
            )
            written += _written(result)
            logger.info("Stored %d/%d summary embeddings for project=%s", written, submitted, project)
        if written < submitted:
            logger.error(
                "Summary-embedding store wrote %d of %d rows for project=%s — %d node(s) missing at write time",
                written, submitted, project, submitted - written,
            )
        return written

    async def search_code_vectors(
        self,
        project: str,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """Search code chunks by vector similarity.

        Retries Memgraph's post-deletion vector-index lag (see
        :func:`_vector_search_with_retry`); any other failure propagates untouched.
        """
        records = await _vector_search_with_retry(
            lambda: self._conn.execute_query(
                queries.VECTOR_SEARCH_CODE,
                {"embedding": embedding, "top_k": _vector_fetch_k(top_k), "project": project},
            )
        )
        return [
            {
                **_serialize_entity(r),
                "score": r["score"],
                "chunk_id": r["chunk_id"],
                "chunk_kind": r["chunk_kind"],
                "match_start_line": r["match_start_line"],
                "match_end_line": r["match_end_line"],
            }
            for r in records[:top_k]
        ]

    async def search_summary_vectors(
        self,
        project: str,
        embedding: list[float],
        top_k: int = 10,
        *,
        node_types: Sequence[str] | None = None,
    ) -> list[dict[str, object]]:
        """Search nodes by summary embedding similarity.

        When every requested ``node_types`` value has a label-scoped index of its own
        (``schema.LEVEL_SUMMARY_INDEXES`` — directories, modules), the search runs there, so
        those few nodes compete only with each other instead of with every function of
        every project in the shared index. Results are then already of those types. Any
        other ``node_types`` (or none) uses the shared index, and the caller filters.

        Retries Memgraph's post-deletion vector-index lag (see
        :func:`_vector_search_with_retry`); any other failure propagates untouched.
        """
        level_indexes = [LEVEL_SUMMARY_INDEXES.get(t) for t in node_types or ()]
        if level_indexes and all(level_indexes):
            rows = await self._search_level_summary_vectors(project, embedding, top_k, level_indexes)
            if rows is not None:
                return rows
        records = await _vector_search_with_retry(
            lambda: self._conn.execute_query(
                queries.VECTOR_SEARCH_SUMMARY,
                {"embedding": embedding, "top_k": _vector_fetch_k(top_k), "project": project},
            )
        )
        return [
            {**_serialize_entity(r), "score": r["score"]}
            for r in records[:top_k]
        ]

    async def _search_level_summary_vectors(
        self,
        project: str,
        embedding: list[float],
        top_k: int,
        indexes: Sequence[str],
    ) -> list[dict[str, object]] | None:
        """Search label-scoped summary indexes and merge them by score, best first.

        Returns ``None`` when an index is not there (a database set up before these
        indexes existed, until the next ``setup_schema``) so the caller falls back to the
        shared index rather than failing a search that used to work.
        """
        merged: list[dict[str, object]] = []
        for index in dict.fromkeys(indexes):
            try:
                records = await _vector_search_with_retry(
                    lambda index=index: self._conn.execute_query(
                        queries.VECTOR_SEARCH_SUMMARY_IN_INDEX,
                        {
                            "index": index,
                            "embedding": embedding,
                            "top_k": max(_vector_fetch_k(top_k), _LEVEL_MIN_FETCH),
                            "project": project,
                        },
                    )
                )
            except StaleVectorIndexError:
                raise
            except Exception:  # noqa: BLE001 — a missing optional index degrades, never fails
                logger.warning(
                    "Summary index %s unavailable — falling back to the shared summary index "
                    "(restart the server so setup_schema creates it)",
                    index,
                    exc_info=True,
                )
                return None
            merged.extend({**_serialize_entity(r), "score": r["score"]} for r in records)
        merged.sort(key=lambda row: float(row["score"] or 0.0), reverse=True)  # type: ignore[arg-type]
        return merged[:top_k]

    async def search_summary_text(
        self,
        project: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """BM25 keyword search over Code.summary via the summary_text text index."""
        records = await self._conn.execute_query(
            queries.TEXT_SEARCH_SUMMARY,
            {"query": query, "top_k": top_k, "project": project},
        )
        return [
            {**_serialize_entity(r), "score": r["score"]}
            for r in records
        ]

    async def search_symbol_text(
        self,
        project: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """BM25 keyword search over Code.name / Code.signature via code_symbols_text."""
        records = await self._conn.execute_query(
            queries.TEXT_SEARCH_SYMBOL,
            {"query": query, "top_k": top_k, "project": project},
        )
        return [
            {**_serialize_entity(r), "score": r["score"]}
            for r in records
        ]

    async def search_code_text(
        self,
        project: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        """BM25 keyword search over CodeChunk.content, joined to the owning Code entity."""
        records = await self._conn.execute_query(
            queries.TEXT_SEARCH_CODE,
            {"query": query, "top_k": top_k, "project": project},
        )
        return [
            {
                **_serialize_entity(r),
                "score": r["score"],
                "chunk_id": r["chunk_id"],
                "chunk_kind": r["chunk_kind"],
                "match_start_line": r["match_start_line"],
                "match_end_line": r["match_end_line"],
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    async def list_entities(
        self,
        project: str,
        node_types: list[str] | None = None,
        limit: int = 51,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        records = await self._conn.execute_query(
            queries.LIST_ENTITIES,
            {"project": project, "node_types": node_types, "limit": limit, "offset": offset},
        )
        return [_serialize_entity(r, "n") for r in records]

    async def get_structure_root(
        self,
        project: str,
        node_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, object] | None:
        if node_id:
            query = queries.GET_STRUCTURE_ROOT_BY_ID
            params = {"project": project, "node_id": node_id}
        elif file_path and (path := _relative_tree_path(file_path)) != ".":
            query = queries.GET_STRUCTURE_ROOT_BY_FILE
            params = {"project": project, "file_path": path}
        else:
            query = queries.GET_PROJECT_ROOT
            params = {"project": project}
        records = await self._conn.execute_query(query, params)
        return _serialize_entity(records[0], "root") if records else None

    async def get_entity_structure(
        self,
        project: str,
        root_id: str,
        depth: int,
        node_types: list[str] | None = None,
        limit: int = 51,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        bounded_depth = min(max(depth, 1), 3)
        records = await self._conn.execute_query(
            queries.GET_ENTITY_STRUCTURE.format(depth=bounded_depth),
            {"project": project, "root_id": root_id, "node_types": node_types, "limit": limit, "offset": offset},
        )
        return [_serialize_entity(r, "n") for r in records]

    async def get_entity_context(
        self,
        project: str,
        node_id: str,
        relation_types: list[str] | None = None,
        limit: int = 26,
        offset: int = 0,
    ) -> dict[str, object] | None:
        records = await self._conn.execute_query(
            queries.GET_ENTITY_CONTEXT,
            {
                "project": project,
                "node_id": node_id,
                "relation_types": relation_types,
                "limit": limit,
                "offset": offset,
            },
        )
        if not records:
            return None
        record = records[0]
        return {
            "entity": _serialize_entity(record["entity"]),
            "parent": _serialize_entity(record["parent"]),
            "children": [_serialize_entity(child) for child in record["children"] if child],
            "incoming": [
                {"relation": row["relation"], "entity": _serialize_entity(row["entity"])}
                for row in record["incoming"]
                if row.get("entity")
            ],
            "outgoing": [
                {"relation": row["relation"], "entity": _serialize_entity(row["entity"])}
                for row in record["outgoing"]
                if row.get("entity")
            ],
        }

    async def get_related_entities(
        self,
        project: str,
        node_id: str,
        direction: str = "both",
        relation_types: list[str] | None = None,
        limit: int = 51,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        records = await self._conn.execute_query(
            queries.GET_RELATED_ENTITIES,
            {
                "project": project,
                "node_id": node_id,
                "direction": direction,
                "relation_types": relation_types,
                "limit": limit,
                "offset": offset,
            },
        )
        return [
            {
                "relation": r["relation"],
                "direction": r["direction"],
                "entity": _serialize_entity(r, "related"),
            }
            for r in records
        ]

    async def search_entities(
        self,
        project: str,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 21,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        records = await self._conn.execute_query(
            queries.SEARCH_ENTITIES,
            {"project": project, "query": query, "node_types": node_types, "limit": limit, "offset": offset},
        )
        return [_serialize_entity(r, "n") for r in records]

