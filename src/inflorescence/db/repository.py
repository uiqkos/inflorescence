"""High-level async repository wrapping Cypher queries for Memgraph."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence

from inflorescence.code_indexer.models import CodeNode, Edge, ExternalRef, NodeType
from inflorescence.db import queries
from inflorescence.db.connection import MemgraphConnection

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
_UPSERT_EDGES_BATCH_TEMPLATE = (
    "UNWIND $rows AS row "
    "MATCH (s:Code {{id: row.source, project: $project}}) "
    "MATCH (t {{id: row.target, project: $project}}) "
    "WHERE t:Code OR t:External "
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
}
_NODE_TYPE_VALUE_BY_LABEL = {label: value for value, label in _NODE_TYPE_LABEL_BY_VALUE.items()}

# Memgraph's vector_search.search has no filtered variant: it returns the GLOBAL
# top-k across every project sharing the index, and the project filter runs after.
# With several indexed projects a small top_k can be fully consumed by another
# project's nodes, silently starving the query. Over-fetch and truncate post-filter.
_VECTOR_OVERFETCH_FACTOR = 20
_VECTOR_MIN_FETCH = 256


def _vector_fetch_k(top_k: int) -> int:
    return max(top_k * _VECTOR_OVERFETCH_FACTOR, _VECTOR_MIN_FETCH)


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
        """Delete all graph and vector data for a project."""
        await self._conn.execute_write(queries.DELETE_PROJECT, {"project": project})
        logger.debug("Deleted all nodes for project=%s", project)

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
        """Search code chunks by vector similarity."""
        records = await self._conn.execute_query(
            queries.VECTOR_SEARCH_CODE,
            {"embedding": embedding, "top_k": _vector_fetch_k(top_k), "project": project},
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
    ) -> list[dict[str, object]]:
        """Search nodes by summary embedding similarity."""
        records = await self._conn.execute_query(
            queries.VECTOR_SEARCH_SUMMARY,
            {"embedding": embedding, "top_k": _vector_fetch_k(top_k), "project": project},
        )
        return [
            {**_serialize_entity(r), "score": r["score"]}
            for r in records[:top_k]
        ]

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
        elif file_path:
            query = queries.GET_STRUCTURE_ROOT_BY_FILE
            params = {"project": project, "file_path": file_path}
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

