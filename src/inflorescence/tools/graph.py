"""Graph navigation MCP tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from mcp.server.fastmcp import FastMCP

from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.tools.path_filters import PathFilterConfig, filter_records_by_path
from inflorescence.tools.responses import (
    MAX_GRAPH_LIMIT,
    NAV_SUMMARY_MAX_CHARS,
    clamp_limit,
    clamp_offset,
    entity_ref,
    error_response,
    paginated_response,
)

FetchRows = Callable[[int, int], Awaitable[list[dict[str, object]]]]

# Upper bound on raw rows scanned while applying path filters before pagination.
# A sparse include-filter over a large project would otherwise walk the whole
# entity set (unbounded round-trips + memory) just to fill one page; cap the scan.
MAX_FILTER_SCAN_ROWS = 5000


async def _fetch_filtered_page(
    fetch_rows: FetchRows,
    filters: PathFilterConfig,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, object]], bool]:
    """Fetch until path filters have been applied before pagination.

    Returns ``(rows, truncated)``. Bounded by MAX_FILTER_SCAN_ROWS: when a filter
    excludes most entities, the scan stops rather than walking the entire project.
    ``truncated`` is True when the scan hit that ceiling with more raw rows still
    unscanned and the page not yet filled — so a short/empty page is reported as
    "the scan stopped early", not silently as "no more matches" (audit finding 15).
    """
    if not filters.include_all and not filters.include:
        return [], False

    batch_size = MAX_GRAPH_LIMIT
    raw_offset = 0
    filtered: list[dict[str, object]] = []
    needed = offset + limit + 1

    while len(filtered) < needed and raw_offset < MAX_FILTER_SCAN_ROWS:
        rows = await fetch_rows(batch_size + 1, raw_offset)
        filtered.extend(filter_records_by_path(rows[:batch_size], filters))
        if len(rows) <= batch_size:
            break
        raw_offset += batch_size

    # We stopped at the cap (not by exhausting raw rows) while still short of a full
    # page: unscanned rows remain that could match, so the result is truncated.
    truncated = raw_offset >= MAX_FILTER_SCAN_ROWS and len(filtered) < needed
    return filtered[offset:needed], truncated


def register_graph_tools(mcp: FastMCP, manager: ProjectManager, repo: GraphRepository) -> None:
    """Register graph navigation tools with the MCP server."""

    @mcp.tool()
    async def list_entities(
        directory: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List indexed code entities with pagination and glob path filters.

        Summaries are truncated to a short preview; use ``get_entity_context``
        for the full summary of a specific entity.
        """
        project = manager.get_project(directory)
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.list_entities(project, node_types, page_limit, page_offset),
            filters,
            bounded_limit,
            bounded_offset,
        )
        items = [ref for row in rows if (ref := entity_ref(row, summary_max_chars=NAV_SUMMARY_MAX_CHARS))]
        return paginated_response(
            items,
            bounded_limit,
            bounded_offset,
            {"project": project, "path_filters": filters.meta(), "node_types": node_types or []},
            truncated=truncated,
        )

    @mcp.tool()
    async def get_entity_structure(
        directory: str,
        node_id: str | None = None,
        file_path: str | None = None,
        depth: int = 1,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return the CONTAINS hierarchy under a node, file module, or project root.

        Item summaries are truncated to a short preview (the root keeps its full
        summary); use ``get_entity_context`` for the full summary of an item.
        """
        project = manager.get_project(directory)
        root = await repo.get_structure_root(project, node_id=node_id, file_path=file_path)
        if root is None:
            return error_response("not_found", "Structure root not found")
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.get_entity_structure(
                project,
                str(root["id"]),
                depth,
                node_types,
                page_limit,
                page_offset,
            ),
            filters,
            bounded_limit,
            bounded_offset,
        )
        items = [ref for row in rows if (ref := entity_ref(row, summary_max_chars=NAV_SUMMARY_MAX_CHARS))]
        payload = paginated_response(
            items,
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "depth": min(max(depth, 1), 3),
                "node_id": node_id,
                "file_path": file_path,
                "node_types": node_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        )
        payload["root"] = entity_ref(root)
        return payload

    @mcp.tool()
    async def get_entity_context(
        directory: str,
        node_id: str,
        relation_types: list[str] | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        """Return parent, children, incoming, and outgoing context for an entity.

        ``relation_types`` filters the ``incoming`` and ``outgoing`` sections to those
        relationship types (e.g. ``["CALLS"]``); ``parent`` and the CONTAINS ``children``
        are the structural skeleton and are always returned. ``offset`` paginates each
        section, so walking ``next_offset`` terminates instead of looping.
        """
        project = manager.get_project(directory)
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        result = await repo.get_entity_context(project, node_id, relation_types, bounded_limit + 1, bounded_offset)
        if result is None:
            return error_response("not_found", f"Entity not found: {node_id}")

        children = [ref for row in result["children"] if (ref := entity_ref(row))]
        incoming = [
            {"relation": row["relation"], "entity": ref}
            for row in result["incoming"]
            if (ref := entity_ref(row["entity"]))
        ]
        outgoing = [
            {"relation": row["relation"], "entity": ref}
            for row in result["outgoing"]
            if (ref := entity_ref(row["entity"]))
        ]
        return {
            "entity": entity_ref(result["entity"]),
            "parent": entity_ref(result["parent"]),
            "sections": {
                "children": paginated_response(children, bounded_limit, bounded_offset, {}),
                "incoming": paginated_response(incoming, bounded_limit, bounded_offset, {}),
                "outgoing": paginated_response(outgoing, bounded_limit, bounded_offset, {}),
            },
            "meta": {"project": project, "node_id": node_id, "relation_types": relation_types or []},
        }

    @mcp.tool()
    async def get_related_entities(
        directory: str,
        node_id: str,
        direction: str = "both",
        relation_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return entities related to a node by graph relationship.

        Summaries are truncated to a short preview; use ``get_entity_context``
        for the full summary of a specific entity.
        """
        project = manager.get_project(directory)
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.get_related_entities(
                project,
                node_id,
                direction,
                relation_types,
                page_limit,
                page_offset,
            ),
            filters,
            bounded_limit,
            bounded_offset,
        )
        filtered_rows = []
        for row in rows:
            entity = row["entity"]
            filtered_rows.append({
                "relation": row["relation"],
                "direction": row["direction"],
                "entity": entity_ref(entity, summary_max_chars=NAV_SUMMARY_MAX_CHARS),
            })
        return paginated_response(
            filtered_rows,
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "node_id": node_id,
                "direction": direction,
                "relation_types": relation_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        )

    @mcp.tool()
    async def search_entities(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search code entities by name substring."""
        project = manager.get_project(directory)
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.search_entities(project, query, node_types, page_limit, page_offset),
            filters,
            bounded_limit,
            bounded_offset,
        )
        items = [ref for row in rows if (ref := entity_ref(row))]
        return paginated_response(
            items,
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "query": query,
                "node_types": node_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        )
