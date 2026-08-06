"""Data access for the dashboard: read-mostly queries against Memgraph."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import neo4j.graph

from inflorescence.config import Settings
from inflorescence.db import queries
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.repository import GraphRepository
from inflorescence.tools.cypher import sanitize_readonly_query, validate_readonly_query

logger = logging.getLogger(__name__)

# Embedding vectors are large and useless in API payloads.
_EXCLUDED_PROPS = frozenset({"summary_embedding", "embedding"})

_TYPE_LABELS = ("Directory", "Module", "Class", "Function", "Method", "Interface", "Enum", "Struct", "Trait")

_RELATION_TYPES = ("CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "CONTAINS")

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".php": "php",
    ".sh": "bash",
}

_TREE_SUMMARY_CHARS = 240
_GRAPH_SUMMARY_CHARS = 200
_MAX_GRAPH_NODES = 300
_MAX_QUERY_LIMIT = 500
# Per-hop edge cap. A hop that returns this many rows hit the ceiling, so the
# neighborhood is incomplete and must report truncated=True (audit finding 15) —
# otherwise a capped hop is indistinguishable from a fully-explored one.
_MAX_HOP_EDGES = 3000
# The exhaustive view ("every entity in the project") is bounded too, but far more
# generously than a neighborhood: the browser has to lay out and paint whatever we
# send. Past these the payload reports truncated=True and drops the least connected
# entities first, so what survives is the part of the codebase that talks the most.
_MAX_FULL_NODES = 4000
_MAX_FULL_EDGES = 20000


def language_for_path(file_path: str | None) -> str | None:
    if not file_path:
        return None
    return _LANGUAGE_BY_EXT.get(Path(file_path).suffix.lower())


def verify_root_path(project: str, path: str) -> str | None:
    """Check that *path* is the directory a project was indexed from.

    Project identifiers are ``{basename}_{md5(resolved_path)[:8]}``, so a
    candidate path can be verified without any extra state. Returns the
    resolved path on success, None otherwise.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        return None
    digest = hashlib.md5(str(resolved).encode()).hexdigest()[:8]  # noqa: S324 — identifier, not security
    return str(resolved) if f"{resolved.name}_{digest}" == project else None


def stitch_chunks(chunks: list[dict[str, Any]]) -> tuple[str, int] | None:
    """Reassemble entity source from overlapping CodeChunks.

    Chunks carry (content, start_line, end_line); consecutive chunks overlap,
    so drop the overlapping leading lines of each subsequent chunk. Returns
    (content, first_line) or None when there is nothing to stitch.
    """
    usable = [c for c in chunks if c.get("content")]
    if not usable:
        return None
    usable.sort(key=lambda c: int(c.get("start_line") or 0))
    lines: list[str] = []
    first_line = int(usable[0].get("start_line") or 1)
    covered_until = first_line - 1  # last line number already emitted
    for chunk in usable:
        start = int(chunk.get("start_line") or (covered_until + 1))
        chunk_lines = str(chunk["content"]).splitlines()
        if start > covered_until + 1 and lines:
            lines.append("")  # gap between non-adjacent chunks
        skip = max(0, covered_until + 1 - start)
        if skip >= len(chunk_lines):
            continue
        lines.extend(chunk_lines[skip:])
        covered_until = max(covered_until, start + len(chunk_lines) - 1)
    return "\n".join(lines), first_line


def _score(value: Any) -> float:
    try:
        return round(float(value or 0), 3)
    except (TypeError, ValueError):
        return 0.0


def _entity_type(labels: Any) -> str:
    if isinstance(labels, (list, tuple, set, frozenset)):
        for label in _TYPE_LABELS:
            if label in labels:
                return label.lower()
    return "code"


def _trim(text: Any, limit: int) -> str | None:
    if not text:
        return None
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _entity_ref(record: Any, prefix: str = "") -> dict[str, Any]:
    p = f"{prefix}." if prefix else ""
    return {
        "id": record[f"{p}id"],
        "name": record[f"{p}name"],
        "type": _entity_type(record[f"{p}labels"]),
        "file_path": record[f"{p}file_path"],
        "line_start": record[f"{p}line_start"],
    }


_ENTITY_PROPS = (
    "{p}.id AS `{a}.id`, {p}.name AS `{a}.name`, labels({p}) AS `{a}.labels`, "
    "{p}.file_path AS `{a}.file_path`, {p}.line_start AS `{a}.line_start`"
)


def _props(var: str, alias: str | None = None) -> str:
    return _ENTITY_PROPS.format(p=var, a=alias or var)


class DashboardService:
    """All dashboard reads (and the one root-path write) against Memgraph."""

    def __init__(self, conn: MemgraphConnection, settings: Settings) -> None:
        self._conn = conn
        self._settings = settings
        self._repo = GraphRepository(conn)

    # ------------------------------------------------------------------
    # Projects & stats
    # ------------------------------------------------------------------

    async def list_projects(self) -> list[dict[str, Any]]:
        records = await self._conn.execute_query(
            "MATCH (n:Code) RETURN DISTINCT n.project AS project ORDER BY project"
        )
        projects = []
        for record in records:
            project = record["project"]
            counts = await self._conn.execute_query(
                """
                MATCH (n:Code {project: $project})
                RETURN count(n) AS entities,
                       sum(CASE WHEN n:Module THEN 1 ELSE 0 END) AS files,
                       sum(CASE WHEN n.summary IS NOT NULL AND n.summary <> '' THEN 1 ELSE 0 END) AS summaries
                """,
                {"project": project},
            )
            root = await self._get_root(project)
            row = counts[0]
            projects.append(
                {
                    "project": project,
                    "display_name": project.rsplit("_", 1)[0],
                    "entities": int(row["entities"] or 0),
                    "files": int(row["files"] or 0),
                    "summaries": int(row["summaries"] or 0),
                    "root_path": root.get("root_path") if root else None,
                }
            )
        projects.sort(key=lambda p: -p["entities"])
        return projects

    async def _get_root(self, project: str) -> dict[str, Any] | None:
        records = await self._conn.execute_query(
            "MATCH (n:Code:Directory {project: $project, id: 'dir:.'}) RETURN n",
            {"project": project},
        )
        return _serialize_node(records[0]["n"]) if records else None

    async def resolved_root_path(self, project: str) -> Path | None:
        """Root directory for a project, if recorded and still present on disk."""
        root = await self._get_root(project)
        raw = (root or {}).get("root_path")
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_dir() else None

    async def set_root_path(self, project: str, path: str) -> dict[str, Any]:
        verified = verify_root_path(project, path)
        if verified is None:
            return {
                "ok": False,
                "error": "Path does not match this project — the project id encodes "
                "an md5 of the indexed directory's absolute path.",
            }
        await self._repo.set_project_root_path(project, verified)
        return {"ok": True, "root_path": verified}

    async def project_status(self, project: str) -> dict[str, Any]:
        """Cheap live-index probe polled by the frontend: lease state + coarse counts.

        A cleanly released lease keeps its :IndexLock node with holder = NULL /
        expires_at = 0 (db/lease.py), so bare node existence is NOT "indexing" —
        the holder must be set and the lease unexpired, on the same millisecond
        clock the lease writes with. A crashed holder reads as indexing for at
        most LEASE_TTL_S until its lease expires.
        """
        lock_rows = await self._conn.execute_query(
            "MATCH (l:IndexLock {project: $project}) "
            "RETURN l.holder AS holder, l.expires_at AS expires_at",
            {"project": project},
        )
        now_ms = int(time.time() * 1000)
        indexing = False
        if lock_rows:
            row = lock_rows[0]
            indexing = bool(row.get("holder")) and int(row.get("expires_at") or 0) >= now_ms

        counts = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project})
            RETURN count(n) AS nodes,
                   sum(CASE WHEN NOT n:Directory AND n.summary IS NOT NULL AND n.summary <> ''
                       THEN 1 ELSE 0 END) AS summarized,
                   sum(CASE WHEN NOT n:Directory THEN 1 ELSE 0 END) AS summarizable
            """,
            {"project": project},
        )
        edge_rows = await self._conn.execute_query(
            "MATCH (:Code {project: $project})-[r]->(:Code {project: $project}) "
            "RETURN count(r) AS edges",
            {"project": project},
        )
        c = counts[0] if counts else {}
        return {
            "project": project,
            "indexing": indexing,
            "nodes": int(c.get("nodes") or 0),
            "edges": int((edge_rows[0].get("edges") if edge_rows else 0) or 0),
            "summarized": int(c.get("summarized") or 0),
            "summarizable": int(c.get("summarizable") or 0),
        }

    async def project_stats(self, project: str) -> dict[str, Any]:
        by_type_records = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project})
            UNWIND labels(n) AS label
            WITH label, count(*) AS count
            WHERE label <> 'Code'
            RETURN label, count ORDER BY count DESC
            """,
            {"project": project},
        )
        edge_records = await self._conn.execute_query(
            """
            MATCH (s:Code {project: $project})-[r]->(t:Code {project: $project})
            RETURN type(r) AS relation, count(r) AS count ORDER BY count DESC
            """,
            {"project": project},
        )
        totals = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project})
            RETURN count(n) AS entities,
                   sum(CASE WHEN n:Module THEN 1 ELSE 0 END) AS files,
                   sum(CASE WHEN NOT n:Directory AND n.summary IS NOT NULL AND n.summary <> '' THEN 1 ELSE 0 END)
                     AS summarized,
                   sum(CASE WHEN NOT n:Directory THEN 1 ELSE 0 END) AS summarizable,
                   sum(CASE WHEN n.summary_embedding IS NOT NULL THEN 1 ELSE 0 END) AS summary_embeddings
            """,
            {"project": project},
        )
        chunk_records = await self._conn.execute_query(
            "MATCH (c:CodeChunk {project: $project}) RETURN count(c) AS chunks",
            {"project": project},
        )
        hubs = await self._conn.execute_query(
            f"""
            MATCH (caller:Code {{project: $project}})-[:CALLS]->(n:Code {{project: $project}})
            WITH n, count(caller) AS callers
            RETURN {_props('n')}, callers
            ORDER BY callers DESC LIMIT 8
            """,
            {"project": project},
        )
        imported = await self._conn.execute_query(
            f"""
            MATCH (importer:Code {{project: $project}})-[:IMPORTS]->(n:Code {{project: $project}})
            WITH n, count(importer) AS importers
            RETURN {_props('n')}, importers
            ORDER BY importers DESC LIMIT 8
            """,
            {"project": project},
        )
        largest = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project})
            WHERE NOT n:Directory AND NOT n:Module
            WITH n.file_path AS file_path, count(n) AS entities
            OPTIONAL MATCH (m:Code:Module {project: $project})
            WHERE m.file_path = file_path
            RETURN file_path, entities, m.id AS module_id
            ORDER BY entities DESC LIMIT 8
            """,
            {"project": project},
        )
        module_paths = await self._conn.execute_query(
            "MATCH (m:Code:Module {project: $project}) RETURN m.file_path AS file_path",
            {"project": project},
        )
        languages = Counter(
            language_for_path(record["file_path"]) or "other" for record in module_paths
        )

        root = await self._get_root(project)
        root_path = (root or {}).get("root_path")
        totals_row = totals[0]
        return {
            "project": project,
            "display_name": project.rsplit("_", 1)[0],
            "root_path": root_path,
            "root_path_exists": bool(root_path and Path(str(root_path)).is_dir()),
            "entities": int(totals_row["entities"] or 0),
            "files": int(totals_row["files"] or 0),
            "chunks": int(chunk_records[0]["chunks"] or 0),
            "summarized": int(totals_row["summarized"] or 0),
            "summarizable": int(totals_row["summarizable"] or 0),
            "summary_embeddings": int(totals_row["summary_embeddings"] or 0),
            "edges": sum(int(r["count"]) for r in edge_records if r["relation"] != "HAS_CHUNK"),
            "by_type": [{"type": r["label"].lower(), "count": int(r["count"])} for r in by_type_records],
            "by_relation": [
                {"relation": r["relation"], "count": int(r["count"])}
                for r in edge_records
                if r["relation"] != "HAS_CHUNK"
            ],
            "most_called": [{**_entity_ref(r, "n"), "callers": int(r["callers"])} for r in hubs],
            "most_imported": [{**_entity_ref(r, "n"), "importers": int(r["importers"])} for r in imported],
            "largest_files": [
                {"file_path": r["file_path"], "entities": int(r["entities"]), "module_id": r["module_id"]}
                for r in largest
            ],
            "languages": [
                {"language": lang, "files": count}
                for lang, count in sorted(languages.items(), key=lambda kv: -kv[1])
            ],
        }

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    async def tree_children(self, project: str, parent_id: str | None) -> dict[str, Any]:
        if parent_id is None:
            root = await self._get_root(project)
            if root is None:
                return {"root": None, "children": []}
            parent_id = "dir:."
        else:
            root = None

        records = await self._conn.execute_query(
            f"""
            MATCH (p:Code {{project: $project, id: $parent_id}})-[:CONTAINS]->(c:Code {{project: $project}})
            OPTIONAL MATCH (c)-[:CONTAINS]->(gc:Code {{project: $project}})
            WITH c, count(gc) AS child_count
            RETURN {_props('c')}, c.summary AS `c.summary`, c.signature AS `c.signature`,
                   c.line_end AS `c.line_end`, child_count
            """,
            {"project": project, "parent_id": parent_id},
        )
        children = []
        for record in records:
            ref = _entity_ref(record, "c")
            children.append(
                {
                    **ref,
                    "line_end": record["c.line_end"],
                    "summary": _trim(record["c.summary"], _TREE_SUMMARY_CHARS),
                    "signature": record["c.signature"] or None,
                    "child_count": int(record["child_count"] or 0),
                }
            )
        order = {"directory": 0, "module": 1}
        children.sort(
            key=lambda c: (
                order.get(c["type"], 2),
                c["file_path"] or "",
                c["line_start"] if c["line_start"] is not None else 0,
                c["name"] or "",
            )
        )
        payload: dict[str, Any] = {"children": children}
        if root is not None:
            payload["root"] = {
                "id": root["id"],
                "name": root.get("name") or ".",
                "type": "directory",
                "summary": _trim(root.get("summary"), _TREE_SUMMARY_CHARS),
                "child_count": len(children),
            }
        return payload

    # ------------------------------------------------------------------
    # Entity detail
    # ------------------------------------------------------------------

    async def entity(self, project: str, node_id: str) -> dict[str, Any] | None:
        records = await self._conn.execute_query(
            "MATCH (n:Code {project: $project, id: $node_id}) RETURN n",
            {"project": project, "node_id": node_id},
        )
        if not records:
            return None
        node = _serialize_node(records[0]["n"])

        # NB: property access inside a list comprehension over nodes(path) misbehaves
        # on Memgraph (all elements get the first node's properties), so return the
        # nodes themselves and unpack driver-side.
        crumb_records = await self._conn.execute_query(
            """
            MATCH path = (root:Code {project: $project, id: 'dir:.'})-[:CONTAINS*0..25]->(n:Code {project: $project, id: $node_id})
            RETURN nodes(path) AS crumbs
            LIMIT 1
            """,
            {"project": project, "node_id": node_id},
        )
        breadcrumbs = []
        if crumb_records:
            for crumb in crumb_records[0]["crumbs"]:
                serialized = _serialize_node(crumb)
                breadcrumbs.append(
                    {"id": serialized.get("id"), "name": serialized.get("name"), "type": serialized["type"]}
                )

        outgoing = await self._relations(project, node_id, "out")
        incoming = await self._relations(project, node_id, "in")
        children = (await self.tree_children(project, node_id))["children"]
        chunk_records = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project, id: $node_id})-[:HAS_CHUNK]->(c:CodeChunk {project: $project})
            RETURN count(c) AS chunks
            """,
            {"project": project, "node_id": node_id},
        )
        return {
            "entity": node,
            "breadcrumbs": breadcrumbs,
            "children": children,
            "outgoing": outgoing,
            "incoming": incoming,
            "chunks": int(chunk_records[0]["chunks"] or 0),
        }

    async def _relations(self, project: str, node_id: str, direction: str) -> list[dict[str, Any]]:
        pattern = (
            "(n:Code {project: $project, id: $node_id})-[r]->(other:Code {project: $project})"
            if direction == "out"
            else "(other:Code {project: $project})-[r]->(n:Code {project: $project, id: $node_id})"
        )
        records = await self._conn.execute_query(
            f"""
            MATCH {pattern}
            WHERE type(r) <> 'CONTAINS'
            RETURN type(r) AS relation, {_props('other')}, other.summary AS `other.summary`
            ORDER BY relation, `other.file_path`, `other.line_start`
            LIMIT 200
            """,
            {"project": project, "node_id": node_id},
        )
        return [
            {
                "relation": record["relation"],
                **_entity_ref(record, "other"),
                "summary": _trim(record["other.summary"], _GRAPH_SUMMARY_CHARS),
            }
            for record in records
        ]

    # ------------------------------------------------------------------
    # Source code
    # ------------------------------------------------------------------

    async def entity_code(self, project: str, node_id: str, whole_file: bool = False) -> dict[str, Any]:
        records = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project, id: $node_id})
            RETURN labels(n) AS labels, n.file_path AS file_path,
                   n.line_start AS line_start, n.line_end AS line_end
            """,
            {"project": project, "node_id": node_id},
        )
        if not records:
            return {"source": None, "reason": "entity_not_found"}
        record = records[0]
        entity_type = _entity_type(record["labels"])
        if entity_type == "directory":
            return {"source": None, "reason": "directory"}
        file_path = record["file_path"]
        line_start = int(record["line_start"] or 1)
        line_end = int(record["line_end"] or line_start)

        root = await self.resolved_root_path(project)
        if root is not None and file_path:
            target = (root / file_path).resolve()
            if target.is_relative_to(root) and target.is_file():
                try:
                    text = target.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    text = None
                if text is not None:
                    lines = text.splitlines()
                    if whole_file or entity_type == "module":
                        content, first = "\n".join(lines), 1
                    else:
                        content = "\n".join(lines[line_start - 1 : line_end])
                        first = line_start
                    return {
                        "source": "file",
                        "language": language_for_path(file_path),
                        "file_path": file_path,
                        "first_line": first,
                        "total_lines": len(lines),
                        "content": content,
                    }

        chunk_records = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project, id: $node_id})-[:HAS_CHUNK]->(c:CodeChunk {project: $project})
            RETURN c.content AS content, c.start_line AS start_line,
                   c.end_line AS end_line, c.language AS language
            ORDER BY c.start_line
            """,
            {"project": project, "node_id": node_id},
        )
        stitched = stitch_chunks([dict(r) for r in chunk_records])
        if stitched is None:
            return {
                "source": None,
                "reason": "no_root_path" if root is None else "no_code",
            }
        content, first = stitched
        return {
            "source": "chunks",
            "language": (chunk_records[0]["language"] if chunk_records else None) or language_for_path(file_path),
            "file_path": file_path,
            "first_line": first,
            "total_lines": None,
            "content": content,
        }

    # ------------------------------------------------------------------
    # Graph extraction
    # ------------------------------------------------------------------

    async def graph_overview(self, project: str) -> dict[str, Any]:
        """Directory/module level graph with aggregated cross-file relations."""
        node_records = await self._conn.execute_query(
            f"""
            MATCH (n:Code {{project: $project}})
            WHERE n:Module OR n:Directory
            RETURN {_props('n')}, n.summary AS `n.summary`
            """,
            {"project": project},
        )
        contains_records = await self._conn.execute_query(
            """
            MATCH (a:Code {project: $project})-[:CONTAINS]->(b:Code {project: $project})
            WHERE (a:Module OR a:Directory) AND (b:Module OR b:Directory)
            RETURN a.id AS source, b.id AS target
            """,
            {"project": project},
        )
        aggregated_records = await self._conn.execute_query(
            """
            MATCH (s:Code {project: $project})-[r]->(t:Code {project: $project})
            WHERE NOT type(r) IN ['CONTAINS', 'HAS_CHUNK'] AND s.file_path <> t.file_path
            MATCH (sm:Code:Module {project: $project, file_path: s.file_path})
            MATCH (tm:Code:Module {project: $project, file_path: t.file_path})
            RETURN sm.id AS source, tm.id AS target, type(r) AS relation, count(r) AS weight
            """,
            {"project": project},
        )
        size_records = await self._conn.execute_query(
            """
            MATCH (m:Code {project: $project})-[:CONTAINS]->(x:Code {project: $project})
            WHERE m:Module OR m:Directory
            RETURN m.id AS id, count(x) AS size
            """,
            {"project": project},
        )
        sizes = {r["id"]: int(r["size"]) for r in size_records}
        nodes = [
            {
                **_entity_ref(r, "n"),
                "summary": _trim(r["n.summary"], _GRAPH_SUMMARY_CHARS),
                "size": sizes.get(r["n.id"], 0),
            }
            for r in node_records
        ]
        edges = [
            {"source": r["source"], "target": r["target"], "relation": "CONTAINS", "weight": 1}
            for r in contains_records
        ] + [
            {
                "source": r["source"],
                "target": r["target"],
                "relation": r["relation"],
                "weight": int(r["weight"]),
            }
            for r in aggregated_records
        ]
        return {"mode": "overview", "nodes": nodes, "edges": edges, "truncated": False}

    async def graph_full(self, project: str, cap: int = _MAX_FULL_NODES) -> dict[str, Any]:
        """Every entity in the project, not just the module skeleton.

        The module overview answers "how do the files relate"; this one answers
        "show me everything", down to functions and methods. Both are unrooted, so
        the client can swap between them without losing its place.
        """
        cap = min(max(cap, 50), _MAX_FULL_NODES)
        node_records = await self._conn.execute_query(
            f"""
            MATCH (n:Code {{project: $project}})
            RETURN {_props('n')}, n.summary AS `n.summary`
            """,
            {"project": project},
        )
        edge_records = await self._conn.execute_query(
            f"""
            MATCH (a:Code {{project: $project}})-[r]->(b:Code {{project: $project}})
            WHERE NOT type(r) IN ['HAS_CHUNK']
            RETURN a.id AS source, b.id AS target, type(r) AS relation
            LIMIT {_MAX_FULL_EDGES}
            """,
            {"project": project},
        )
        truncated = len(edge_records) >= _MAX_FULL_EDGES

        degree: dict[str, int] = {}
        for record in edge_records:
            degree[record["source"]] = degree.get(record["source"], 0) + 1
            degree[record["target"]] = degree.get(record["target"], 0) + 1

        # Over the cap, keep the connected core: directories and modules first (they
        # are the map you navigate by), then whatever has the most relations.
        rank = {"Directory": 0, "Module": 1}
        def sort_key(record: Any) -> tuple[int, int]:
            labels = record["n.labels"] or []
            structural = min((rank[label] for label in labels if label in rank), default=2)
            return (structural, -degree.get(record["n.id"], 0))

        if len(node_records) > cap:
            node_records = sorted(node_records, key=sort_key)[:cap]
            truncated = True

        nodes = [
            {
                **_entity_ref(record, "n"),
                "summary": _trim(record["n.summary"], _GRAPH_SUMMARY_CHARS),
                "degree": degree.get(record["n.id"], 0),
            }
            for record in node_records
        ]
        kept = {node["id"] for node in nodes}
        edges = [
            {
                "source": record["source"],
                "target": record["target"],
                "relation": record["relation"],
                "weight": 1,
            }
            for record in edge_records
            if record["source"] in kept and record["target"] in kept
        ]
        return {"mode": "full", "nodes": nodes, "edges": edges, "truncated": truncated}

    async def graph_neighborhood(
        self,
        project: str,
        root_id: str,
        depth: int = 2,
        relations: list[str] | None = None,
        cap: int = _MAX_GRAPH_NODES,
    ) -> dict[str, Any]:
        """BFS around a node, one hop per query, bounded by *cap* nodes."""
        rels = [r for r in (relations or list(_RELATION_TYPES)) if r in _RELATION_TYPES]
        if not rels:
            rels = list(_RELATION_TYPES)
        # CONTAINS is the structural spine: a module reaches its functions and classes
        # only through it, and those nested entities are where CALLS/INHERITS/IMPLEMENTS
        # live. Always traverse it so that deselecting a relation filters the *view*
        # (client-side) without severing reachability — otherwise a module-rooted
        # neighborhood collapses to just its module-level imports.
        traversal_rels = rels if "CONTAINS" in rels else [*rels, "CONTAINS"]
        depth = min(max(depth, 1), 4)
        cap = min(max(cap, 10), _MAX_GRAPH_NODES)

        root_records = await self._conn.execute_query(
            f"MATCH (n:Code {{project: $project, id: $node_id}}) "
            f"RETURN {_props('n')}, n.summary AS `n.summary`",
            {"project": project, "node_id": root_id},
        )
        if not root_records:
            return {"mode": "neighborhood", "nodes": [], "edges": [], "truncated": False}

        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        truncated = False

        def add_node(record: Any, prefix: str) -> None:
            ref = _entity_ref(record, prefix)
            ref["summary"] = _trim(record[f"{prefix}.summary"], _GRAPH_SUMMARY_CHARS)
            nodes.setdefault(ref["id"], ref)

        add_node(root_records[0], "n")
        frontier = [root_id]

        for _ in range(depth):
            if not frontier or truncated:
                break
            records = await self._conn.execute_query(
                f"""
                MATCH (a:Code {{project: $project}})-[r]->(b:Code {{project: $project}})
                WHERE (a.id IN $frontier OR b.id IN $frontier) AND type(r) IN $rels
                RETURN {_props('a')}, a.summary AS `a.summary`,
                       {_props('b')}, b.summary AS `b.summary`,
                       type(r) AS relation
                LIMIT {_MAX_HOP_EDGES}
                """,
                {"project": project, "frontier": frontier, "rels": traversal_rels},
            )
            if len(records) >= _MAX_HOP_EDGES:
                truncated = True
            next_frontier: list[str] = []
            for record in records:
                for prefix in ("a", "b"):
                    node_id = record[f"{prefix}.id"]
                    if node_id not in nodes:
                        if len(nodes) >= cap:
                            truncated = True
                            continue
                        add_node(record, prefix)
                        next_frontier.append(node_id)
                if record["a.id"] in nodes and record["b.id"] in nodes:
                    key = (record["a.id"], record["b.id"], record["relation"])
                    edges.setdefault(
                        key,
                        {
                            "source": record["a.id"],
                            "target": record["b.id"],
                            "relation": record["relation"],
                            "weight": 1,
                        },
                    )
            frontier = next_frontier

        degree_records = await self._conn.execute_query(
            """
            MATCH (n:Code {project: $project})-[r]-(m:Code {project: $project})
            WHERE n.id IN $ids AND type(r) IN $rels
            RETURN n.id AS id, count(r) AS degree
            """,
            {"project": project, "ids": list(nodes.keys()), "rels": traversal_rels},
        )
        degrees = {r["id"]: int(r["degree"]) for r in degree_records}
        for node_id, node in nodes.items():
            node["degree"] = degrees.get(node_id, 0)
        return {
            "mode": "neighborhood",
            "root": root_id,
            "nodes": list(nodes.values()),
            "edges": list(edges.values()),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, project: str, query: str, mode: str = "quick") -> dict[str, Any]:
        result: dict[str, Any] = {"name": [], "text": [], "semantic": [], "notes": []}
        if not query.strip():
            return result

        name_records = await self._conn.execute_query(
            queries.SEARCH_ENTITIES,
            {"project": project, "query": query, "node_types": None, "limit": 15, "offset": 0},
        )
        result["name"] = [
            {**_entity_ref(r, "n"), "summary": _trim(r["n.summary"], _GRAPH_SUMMARY_CHARS)}
            for r in name_records
        ]

        try:
            from inflorescence.tools.search import _sanitize_text_query

            sanitized = _sanitize_text_query(query)
            if sanitized:
                symbol_rows = await self._repo.search_symbol_text(project, sanitized, top_k=10)
                summary_rows = await self._repo.search_summary_text(project, sanitized, top_k=10)
                seen = {str(hit["id"]) for hit in result["name"]}
                for row in symbol_rows + summary_rows:
                    row_id = str(row["id"])
                    if row_id in seen:
                        continue
                    seen.add(row_id)
                    result["text"].append(
                        {
                            "id": row_id,
                            "name": row["name"],
                            "type": row["node_type"] or "code",
                            "file_path": row["file_path"],
                            "line_start": row["line_start"],
                            "summary": _trim(row["summary"], _GRAPH_SUMMARY_CHARS),
                            "score": _score(row.get("score")),
                        }
                    )
        except Exception:
            result["notes"].append("Full-text search unavailable (requires Memgraph MAGE text indexes).")

        if mode == "semantic":
            semantic = await self._semantic_search(project, query)
            result["semantic"] = semantic.get("hits", [])
            if semantic.get("note"):
                result["notes"].append(semantic["note"])
        return result

    async def _semantic_search(self, project: str, query: str) -> dict[str, Any]:
        if not self._settings.llm_api_key.strip():
            return {"hits": [], "note": "Semantic search needs OPENROUTER_API_KEY."}
        try:
            from inflorescence.rag.embeddings import EmbeddingClient

            embedder = EmbeddingClient(self._settings, model=self._settings.summary_embedding_model)
            embeddings = await asyncio.to_thread(embedder.embed, [query])
        except Exception as exc:
            logger.exception("Semantic search embedding failed")
            return {"hits": [], "note": f"Embedding failed: {exc}"}
        if not embeddings or embeddings[0] is None:
            return {"hits": [], "note": "Embedding returned no vector."}
        embedding = embeddings[0]
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        backends = (self._repo.search_summary_vectors, self._repo.search_code_vectors)
        succeeded = 0
        for search in backends:
            try:
                rows = await search(project, embedding, top_k=10)
            except Exception:
                # INV-7: a swallowed backend failure must not read as "no matches".
                logger.exception("Semantic search backend %s failed", getattr(search, "__name__", search))
                continue
            succeeded += 1
            for row in rows:
                row_id = str(row["id"])
                if row_id in seen:
                    continue
                seen.add(row_id)
                hits.append(
                    {
                        "id": row_id,
                        "name": row["name"],
                        "type": row["node_type"] or "code",
                        "file_path": row["file_path"],
                        "line_start": row["line_start"],
                        "summary": _trim(row["summary"], _GRAPH_SUMMARY_CHARS),
                        "score": _score(row.get("score")),
                    }
                )
        if succeeded == 0:
            return {
                "hits": [],
                "note": "Semantic search unavailable — vector search failed "
                "(needs the Memgraph MAGE image with vector indexes).",
            }
        hits.sort(key=lambda h: -h["score"])
        result: dict[str, Any] = {"hits": hits[:15]}
        if succeeded < len(backends):
            result["note"] = "Semantic search partially unavailable — one vector backend failed; results may be incomplete."
        return result

    # ------------------------------------------------------------------
    # Ad-hoc Cypher
    # ------------------------------------------------------------------

    async def run_query(
        self, project: str, query: str, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        validation_error = validate_readonly_query(query)
        if validation_error is not None:
            return {"error": validation_error}
        limit = min(max(limit, 1), _MAX_QUERY_LIMIT)
        offset = max(offset, 0)
        # Execute the sanitized text, never the original — see sanitize_readonly_query.
        paginated = f"{sanitize_readonly_query(query)} SKIP $offset LIMIT $limit"
        try:
            records = await self._conn.execute_query(
                paginated,
                {"project": project, "limit": limit + 1, "offset": offset},
                max_records=limit + 1,
            )
        except Exception as exc:
            return {"error": str(exc)}

        graph_nodes: dict[str, dict[str, Any]] = {}
        element_to_id: dict[str, str] = {}
        graph_edges: dict[tuple[str, str, str], dict[str, Any]] = {}

        def collect(value: Any) -> None:
            if isinstance(value, neo4j.graph.Node):
                serialized = _serialize_node(value)
                if serialized.get("id"):
                    graph_nodes.setdefault(str(serialized["id"]), serialized)
                    element_to_id[value.element_id] = str(serialized["id"])
            elif isinstance(value, neo4j.graph.Relationship):
                pass  # resolved after all nodes are collected
            elif isinstance(value, neo4j.graph.Path):
                for node in value.nodes:
                    collect(node)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        def collect_edges(value: Any) -> None:
            if isinstance(value, neo4j.graph.Relationship):
                if value.start_node is None or value.end_node is None:
                    return
                source = element_to_id.get(value.start_node.element_id)
                target = element_to_id.get(value.end_node.element_id)
                if source and target:
                    key = (source, target, value.type)
                    graph_edges.setdefault(
                        key, {"source": source, "target": target, "relation": value.type, "weight": 1}
                    )
            elif isinstance(value, neo4j.graph.Path):
                for rel in value.relationships:
                    collect_edges(rel)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_edges(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect_edges(item)

        rows = []
        for record in records[:limit]:
            for value in record.values():
                collect(value)
            rows.append({key: _serialize_value(value) for key, value in record.items()})
        for record in records[:limit]:
            for value in record.values():
                collect_edges(value)

        return {
            "columns": list(records[0].keys()) if records else [],
            "rows": rows,
            "has_more": len(records) > limit,
            "offset": offset,
            "limit": limit,
            "graph": {
                "nodes": list(graph_nodes.values()),
                "edges": list(graph_edges.values()),
            },
        }


# ---------------------------------------------------------------------------
# neo4j value serialization
# ---------------------------------------------------------------------------


def _serialize_node(node: neo4j.graph.Node) -> dict[str, Any]:
    props = {k: v for k, v in dict(node).items() if k not in _EXCLUDED_PROPS}
    labels = sorted(node.labels)
    props["labels"] = labels
    props["type"] = _entity_type(labels)
    props["has_summary_embedding"] = "summary_embedding" in dict(node)
    return props


def _serialize_value(value: Any) -> Any:
    if isinstance(value, neo4j.graph.Node):
        return {"__kind": "node", **_serialize_node(value)}
    if isinstance(value, neo4j.graph.Relationship):
        return {"__kind": "relationship", "relation": value.type}
    if isinstance(value, neo4j.graph.Path):
        return {
            "__kind": "path",
            "nodes": [_serialize_value(n) for n in value.nodes],
            "relations": [r.type for r in value.relationships],
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value
