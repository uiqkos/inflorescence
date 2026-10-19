"""Graph navigation MCP tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.tools import descriptions as doc
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import build_checker
from inflorescence.tools.gates import ensure_indexed, normalize_node_types
from inflorescence.tools.path_filters import (
    PathFilterConfig,
    filter_records_by_path,
    is_test_path,
    record_path,
)
from inflorescence.tools.responses import (
    FULL_SUMMARY_MAX_LIMIT,
    MAX_GRAPH_LIMIT,
    NAV_SUMMARY_MAX_CHARS,
    clamp_limit,
    clamp_offset,
    entity_ref,
    error_response,
    paginated_response,
)

FetchRows = Callable[[int, int], Awaitable[list[dict[str, object]]]]
Keep = Callable[[dict[str, object]], bool]

# Upper bound on raw rows scanned while applying path filters before pagination.
# A sparse include-filter over a large project would otherwise walk the whole
# entity set (unbounded round-trips + memory) just to fill one page; cap the scan.
MAX_FILTER_SCAN_ROWS = 5000

# Aliases accepted for get_related_entities' ``direction`` argument, normalized to the
# canonical values the Cypher in db/queries.py matches ('both' | 'outgoing' | 'incoming').
# Matching is case-insensitive; anything not in this map is rejected before the query
# runs rather than silently matching zero rows (INV-7: empty and invalid must differ).
_DIRECTION_ALIASES = {
    "both": "both",
    "outgoing": "outgoing",
    "out": "outgoing",
    "incoming": "incoming",
    "in": "incoming",
}


def _normalize_direction(direction: str) -> str | dict:
    """Normalize ``direction`` to 'both' | 'outgoing' | 'incoming', case-insensitively.

    Returns the canonical string on success, or an ``error_response`` payload
    (dict) for any value not in ``_DIRECTION_ALIASES`` — the caller must check
    ``isinstance(result, dict)`` and return it as-is before touching the DB.
    """
    normalized = _DIRECTION_ALIASES.get(direction.strip().lower())
    if normalized is None:
        return error_response(
            "invalid_direction",
            f"Invalid direction {direction!r}. Valid values: 'both', 'outgoing' (alias 'out'), "
            "'incoming' (alias 'in'). Matching is case-insensitive.",
        )
    return normalized


# Candidates listed when a name is ambiguous. Enough to recognise the one you meant,
# short enough that an error stays an error rather than a search result; anything past it
# is reported as a count so the caller knows the list was cut.
MAX_NAME_CANDIDATES = 10
# How many matches are fetched to produce that list. Beyond this the surplus is reported
# as "N+ more" — the exact total is not worth a full scan of a common substring.
_NAME_CANDIDATE_SCAN = 50


def _describe_candidate(row: dict[str, object]) -> str:
    """One candidate as ``id (node_type, file_path)`` — everything a precise re-call needs."""
    return f"{row.get('id')} ({row.get('node_type')}, {row.get('file_path')})"


def _relative_inside(path: Path, root: Path) -> str | None:
    """*path* relative to *root* as typed, else with symlinks resolved; None when outside."""
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return None


def _tree_path(directory: str, file_path: str) -> str | None:
    """*file_path* in the form the graph keys files and directories by, or None if outside.

    The graph stores project-relative paths (``src/pkg``, a file's ``src/pkg/a.py``) and the
    root as ``.``; callers type ``src/pkg/``, ``./src/pkg``, ``./`` or an absolute path inside
    the project just as often, and each used to miss. An absolute path outside the project is
    not a place in it, so it is refused rather than looked up as a relative name.
    """
    path = file_path.strip().replace("\\", "/")
    if Path(path).is_absolute():
        inside = _relative_inside(Path(path), Path(directory))
        if inside is None:
            return None
        path = inside
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/") or "."


def _structure_root_missing(directory: str, node_id: str | None, file_path: str | None) -> str:
    """Explain a missing structure root, checking the filesystem before blaming the caller.

    ``file_path`` is the one argument this server can verify on the spot, and the two
    readings of "not found" are worlds apart: a typo'd path, or a real file that was never
    indexed. Answering both with the same three words let an agent conclude a 174-line
    module did not exist (field report, 10.09.2026) — one ``exists()`` call tells them apart.
    A directory is looked up by its path like a file, so a directory that gets here holds no
    indexed file at all.
    """
    if not file_path:
        return (
            f"No indexed entity with node_id={node_id!r}. Ids come from a search or listing "
            "in this project; pass file_path= to address a file or a directory instead."
        )
    try:
        target = Path(directory) / file_path
        is_dir = target.is_dir()
        on_disk = is_dir or target.exists()
    except OSError:  # an unreadable path is not evidence either way
        is_dir = on_disk = False
    if is_dir:
        return (
            f"{file_path!r} is a directory on disk, but no file under it is in this index — "
            "every file there is excluded (.gitignore, .inflorescenceignore, an exclude "
            "pattern), is of a type this index does not admit, or was created after the last "
            f"index run. Read it from the filesystem; index_directory({directory!r}) picks up "
            "files that are merely new."
        )
    if on_disk:
        return (
            f"{file_path!r} exists on disk but is NOT in this index, so nothing about it is "
            "searchable here — it was created after the last index run, or an exclude pattern "
            "or .gitignore keeps it out. Read it from the filesystem, and run "
            f"index_directory({directory!r}) to index it. This is not 'the file has no structure'."
        )
    return (
        f"No indexed entity for file_path={file_path!r}, and no such file or directory on disk "
        f"under {directory!r} either. Paths are relative to the project root; list_entities("
        f"node_types=[\"module\", \"document\"]) shows which files this index holds."
    )


def _pick_by_name(rows: Sequence[dict[str, object]], name: str) -> tuple[dict[str, object] | None, str, list]:
    """Apply the name rules to candidate rows: ``(resolved row or None, rule, candidates)``.

    The rules, in order: rows named exactly *name*; rows named *name* ignoring case; every
    row. The first rule that finds any row decides — one row resolves, several are the
    candidates (a narrower list than every substring hit).
    """
    exact = [row for row in rows if row.get("name") == name]
    if exact:
        return (exact[0] if len(exact) == 1 else None), "exact", exact
    folded = name.casefold()
    ignoring_case = [row for row in rows if str(row.get("name") or "").casefold() == folded]
    if ignoring_case:
        return (ignoring_case[0] if len(ignoring_case) == 1 else None), "exact_ignoring_case", ignoring_case
    return (rows[0] if len(rows) == 1 else None), "substring", list(rows)


def _narrows(filters: PathFilterConfig) -> bool:
    return not filters.include_all or bool(filters.exclude)


async def _resolve_name(
    repo: GraphRepository, project: str, name: str, filters: PathFilterConfig
) -> tuple[str, str] | dict:
    """Resolve *name* to exactly one node id, or to the error explaining why not.

    Returns ``(node_id, rule)`` on success — the rule that decided, for ``meta.name_match`` —
    and an ``error_response`` payload otherwise, so the caller checks ``isinstance(result,
    dict)``. Ambiguity is never resolved by picking the first match: "who calls X" answered
    about the wrong X is a wrong answer that looks right, and the caller cannot tell. An exact
    spelling is not a pick, it is what was asked for: ``TASKS`` is the constant, not
    ``readTasks`` (field report, 15.09.2026, where the constant was not even among the ten
    candidates shown). The search query lists exact names first, so they are never cut off by
    the candidate scan. Path filters narrow only what the rules leave ambiguous.
    """
    rows = await repo.search_entities(project, name, None, _NAME_CANDIDATE_SCAN + 1, 0)
    if not rows:
        return error_response(
            "not_found",
            f"No indexed entity's name contains {name!r}. Names are matched case-insensitively "
            "as a substring; for a term that appears in code or text rather than in a name, "
            "use search_text. Note this says nothing about the filesystem: entities exist only "
            "for files that were indexed, so a file written since the last index run has none — "
            "search_entities reports what this index actually covers in meta.index_coverage.",
        )
    picked, rule, candidates = _pick_by_name(rows, name)
    if picked is None and _narrows(filters):
        scoped, _ = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.search_entities(project, name, None, page_limit, page_offset),
            filters,
            _NAME_CANDIDATE_SCAN + 1,
            0,
        )
        if scoped:
            picked, rule, candidates = _pick_by_name(scoped, name)
            rule += "_within_path_filters"
    if picked is not None:
        return str(picked["id"]), rule
    shown = [_describe_candidate(row) for row in candidates[:MAX_NAME_CANDIDATES]]
    surplus = len(candidates) - len(shown)
    more = ""
    if surplus:
        more = f", and {surplus}+ more" if len(candidates) > _NAME_CANDIDATE_SCAN else f", and {surplus} more"
    kind = {
        "exact": "is the exact name of",
        "exact_ignoring_case": "is the name (ignoring case) of",
    }.get(rule.removesuffix("_within_path_filters"), "matches")
    return error_response(
        "ambiguous_name",
        f"{name!r} {kind} more than one entity — re-call with node_id set to one of: "
        f"{'; '.join(shown)}{more}. include_paths/exclude_paths narrow the candidates too.",
    )


async def _fetch_filtered_page(
    fetch_rows: FetchRows,
    filters: PathFilterConfig,
    limit: int,
    offset: int,
    keep: Keep | None = None,
) -> tuple[list[dict[str, object]], bool]:
    """Fetch until path filters (and *keep*) have been applied before pagination.

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
        allowed = filter_records_by_path(rows[:batch_size], filters)
        filtered.extend(row for row in allowed if keep is None or keep(row))
        if len(rows) <= batch_size:
            break
        raw_offset += batch_size

    # We stopped at the cap (not by exhausting raw rows) while still short of a full
    # page: unscanned rows remain that could match, so the result is truncated.
    truncated = raw_offset >= MAX_FILTER_SCAN_ROWS and len(filtered) < needed
    return filtered[offset:needed], truncated


def _not_test(row: dict[str, object]) -> bool:
    return not is_test_path(record_path(row))


def _flag_items(items: Sequence[MutableMapping[str, Any]]) -> None:
    """Mark test items in place, the way search hits are marked; order is kept."""
    for item in items:
        if is_test_path(record_path(item)):
            item["is_test"] = True


def _page_bounds(limit: int, full_summaries: bool) -> tuple[int, int | None]:
    """``(limit, summary_max_chars)`` for a navigation page: previews, or whole summaries on a smaller page."""
    if full_summaries:
        return clamp_limit(limit, FULL_SUMMARY_MAX_LIMIT), None
    return clamp_limit(limit), NAV_SUMMARY_MAX_CHARS


_NAVIGATION_COMMON = (doc.PATH_FILTERS, doc.WORKTREE_FIRST_QUERY)

LIST_ENTITIES_DESCRIPTION = doc.join(
    """List indexed code entities with pagination and glob path filters.

For inventory and orientation — "what is in this package", "how many classes are under src/" —
not for finding code by meaning (``search_semantic``) or by name (``search_entities``); this
returns entities in path order, unranked. It is flat and exhaustive — every module of a big
package on one page, summaries included, which runs to tens of thousands of tokens — so to *find
where something lives*, walk the tree with ``get_entity_structure`` instead (one directory level
per call, each with its subtree summary) and come here for a complete inventory of a subtree you
have already chosen. Combine ``include_paths``/``exclude_paths`` globs with ``node_types`` to
scope the listing.""",
    doc.NODE_TYPES,
    doc.LISTING_FLAGS,
    doc.FULL_SUMMARIES,
    *_NAVIGATION_COMMON,
    doc.PROVENANCE,
    doc.EMPTY_PAGE,
)

GET_ENTITY_STRUCTURE_DESCRIPTION = doc.join(
    """Return the CONTAINS hierarchy under a directory, a file, a node, or the project root.

The top-down navigator, and the cheapest way to find where something lives. Every directory
carries a summary of its whole subtree (purpose, capabilities, entry points), so walk down one
level per call, reading summaries to pick the next branch:

1. ``get_entity_structure(directory)`` — no ``file_path``/``node_id``: the project root's full
   summary in ``root``, its top-level files and directories in ``items``.
2. ``get_entity_structure(directory, file_path="src/pkg")`` — a directory: its summary and
   children.
3. ``get_entity_structure(directory, file_path="src/pkg/client.py")`` — a file: its classes,
   functions and methods with line ranges; ``depth=2`` adds methods of classes. Then read just
   those lines.

``file_path`` is relative to the project root; ``"./src/pkg/"``, ``"."`` (the root) and an
absolute path inside ``directory`` work too — an absolute path outside it is an
``invalid_argument`` error. ``node_id`` (an id from any tool, e.g. ``"dir:src/lib"``) addresses
the root directly. ``depth`` goes up to 3 when you want a bigger slice at once. For the edges
*between* entities — callers, callees, imports — use ``get_related_entities`` or
``get_entity_context`` instead. ``node_types`` and the path filters apply to the items only,
never the root.""",
    doc.NODE_TYPES,
    doc.LISTING_FLAGS,
    doc.FULL_SUMMARIES,
    *_NAVIGATION_COMMON,
    doc.PROVENANCE,
    """A ``not_found`` error distinguishes the cases that matter: a ``file_path`` that does not
exist under this directory at all; a file on disk that was never indexed (created after the
last index run, or excluded) — a coverage gap, so read it directly and ``index_directory`` to
make it queryable; and a directory on disk none of whose files is indexed.""",
)

GET_ENTITY_CONTEXT_DESCRIPTION = doc.join(
    """Return parent, children, incoming, and outgoing context for an entity.

The one-call deep dive on a known ``node_id``: full (untruncated) summary plus its neighbourhood
in every direction. Use it on a hit from any search tool; for a whole subtree use
``get_entity_structure``, and for one edge kind over many hops use ``get_related_entities``.
``relation_types`` filters the ``incoming`` and ``outgoing`` sections to those relationship types
(e.g. ``["CALLS"]``); ``parent`` and the CONTAINS ``children`` are the structural skeleton and are
always returned. ``offset`` paginates each section, so walking ``next_offset`` terminates instead
of looping. A top-level declaration (``variable``, ``type_alias``) has no summary; its
``signature`` is its first line, and its lines are ``line_start``–``line_end``.""",
    doc.IMPORT_FACTS,
    doc.WORKTREE_FIRST_QUERY,
    doc.PROVENANCE,
)

GET_RELATED_ENTITIES_DESCRIPTION = doc.join(
    """Return entities related to a node by graph relationship.

Name the entity with **exactly one** of ``node_id`` (an id from any search tool) or ``name`` (an
entity's name, resolved here). Passing both, or neither, returns ``invalid_argument``. ``name``
is the one-call form of "who calls X": ``get_related_entities(directory, name="load_board",
direction="incoming", relation_types=["CALLS"])``; ``meta.resolved_node_id`` reports what it
resolved to.""",
    doc.NAME_RESOLUTION,
    """``direction`` selects which edges to follow relative to that entity: ``"both"`` (default),
``"outgoing"`` (alias ``"out"``), or ``"incoming"`` (alias ``"in"``). Matching is
case-insensitive. Any other value returns an ``invalid_direction`` error instead of an empty
result. ``include_paths``/``exclude_paths`` and ``include_tests`` filter the related entities
(and narrow an ambiguous ``name``).""",
    doc.IMPORT_FACTS,
    doc.LISTING_FLAGS,
    doc.FULL_SUMMARIES,
    *_NAVIGATION_COMMON,
    doc.PROVENANCE,
    doc.EMPTY_PAGE,
)

SEARCH_ENTITIES_DESCRIPTION = doc.join(
    """Search code entities by case-insensitive substring of their name.

Pure name matching — ``"handl"`` finds ``handle_request`` and ``ErrorHandler`` — so use it when
you already know (part of) an identifier, and when a name-based answer must be complete the
moment a first-time index finishes parsing, before any summary exists. Entities named exactly
the query are listed first (exact spelling, then ignoring case), then every other match in path
order. It never looks inside bodies, signatures, or summaries: for a string literal, an error
message, or any text in the code use ``search_text``; to find code by what it does use
``search_semantic``.""",
    doc.NODE_TYPES,
    doc.LISTING_FLAGS,
    *_NAVIGATION_COMMON,
    doc.PROVENANCE,
    doc.EMPTY_PAGE,
)


def register_graph_tools(mcp: FastMCP, manager: ProjectManager, repo: GraphRepository) -> None:
    """Register graph navigation tools with the MCP server."""

    @mcp.tool(description=LIST_ENTITIES_DESCRIPTION)
    async def list_entities(
        directory: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        full_summaries: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Path-ordered inventory (published description: LIST_ENTITIES_DESCRIPTION)."""
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit, summary_chars = _page_bounds(limit, full_summaries)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.list_entities(project, node_types, page_limit, page_offset),
            filters,
            bounded_limit,
            bounded_offset,
            keep=None if include_tests else _not_test,
        )
        items = [ref for row in rows if (ref := entity_ref(row, summary_max_chars=summary_chars))]
        payload = paginated_response(
            items,
            bounded_limit,
            bounded_offset,
            _listing_meta(project, filters, node_types, include_tests, full_summaries),
            truncated=truncated,
        )
        _flag_items(payload["items"])
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)

    @mcp.tool(description=GET_ENTITY_STRUCTURE_DESCRIPTION)
    async def get_entity_structure(
        directory: str,
        node_id: str | None = None,
        file_path: str | None = None,
        depth: int = 1,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        full_summaries: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """CONTAINS outline under a root (published description: GET_ENTITY_STRUCTURE_DESCRIPTION)."""
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        tree_path = _tree_path(directory, file_path) if file_path and not node_id else None
        if file_path and not node_id and tree_path is None:
            return error_response(
                "invalid_argument",
                f"file_path {file_path!r} is an absolute path outside {directory!r}; pass a path "
                "relative to the project root (or an absolute path inside it).",
            )
        # A directory is found by its path like a file (field report, 14.09.2026 — the error
        # sent the caller to re-index a directory that was indexed).
        root = await repo.get_structure_root(project, node_id=node_id, file_path=tree_path)
        if root is None:
            return error_response("not_found", _structure_root_missing(directory, node_id, file_path))
        bounded_limit, summary_chars = _page_bounds(limit, full_summaries)
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
            keep=None if include_tests else _not_test,
        )
        items = [ref for row in rows if (ref := entity_ref(row, summary_max_chars=summary_chars))]
        meta = _listing_meta(project, filters, node_types, include_tests, full_summaries)
        meta.update({"depth": min(max(depth, 1), 3), "node_id": node_id, "file_path": file_path})
        payload = paginated_response(items, bounded_limit, bounded_offset, meta, truncated=truncated)
        _flag_items(payload["items"])
        checker = await build_checker(repo, project, directory)
        # The root is annotated before ``stamp``, so its file counts in index_freshness.
        payload["root"] = checker.annotate(entity_ref(root))
        return checker.stamp(payload)

    @mcp.tool(description=GET_ENTITY_CONTEXT_DESCRIPTION)
    async def get_entity_context(
        directory: str,
        node_id: str,
        relation_types: list[str] | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        """One entity and its neighbourhood (published description: GET_ENTITY_CONTEXT_DESCRIPTION)."""
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        result = await repo.get_entity_context(project, node_id, relation_types, bounded_limit + 1, bounded_offset)
        if result is None:
            return error_response("not_found", f"Entity not found: {node_id}")

        children = [ref for row in result["children"] if (ref := entity_ref(row))]
        incoming = [_relation_item(row, ref) for row in result["incoming"] if (ref := entity_ref(row["entity"]))]
        outgoing = [_relation_item(row, ref) for row in result["outgoing"] if (ref := entity_ref(row["entity"]))]
        checker = await build_checker(repo, project, directory)
        payload = {
            "entity": checker.annotate(entity_ref(result["entity"])),
            "parent": checker.annotate(entity_ref(result["parent"])),
            "sections": {
                "children": paginated_response(children, bounded_limit, bounded_offset, {}),
                "incoming": paginated_response(incoming, bounded_limit, bounded_offset, {}),
                "outgoing": paginated_response(outgoing, bounded_limit, bounded_offset, {}),
            },
            "meta": {"project": project, "node_id": node_id, "relation_types": relation_types or []},
        }
        # Sections are paginated envelopes of their own; annotate the items the caller
        # actually gets (paginated_response already trimmed the limit-plus-one row), then
        # stamp the single meta block covering entity, parent and every section.
        for section in payload["sections"].values():
            checker.annotate_items(section["items"])
        return checker.stamp(payload)

    @mcp.tool(description=GET_RELATED_ENTITIES_DESCRIPTION)
    async def get_related_entities(
        directory: str,
        node_id: str | None = None,
        name: str | None = None,
        direction: str = "both",
        relation_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        full_summaries: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Edges of one entity (published description: GET_RELATED_ENTITIES_DESCRIPTION)."""
        # Argument validation before anything that touches the DB or auto-indexes.
        if bool(node_id) == bool(name):
            return error_response(
                "invalid_argument",
                "Pass exactly one of node_id (an id from a search result) or name (an entity's "
                "name, resolved here), not both and not neither.",
            )
        normalized_direction = _normalize_direction(direction)
        if isinstance(normalized_direction, dict):
            return normalized_direction
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        resolved: tuple[str, str] | None = None
        if name:
            outcome = await _resolve_name(repo, project, name, filters)
            if isinstance(outcome, dict):
                return outcome
            resolved = outcome
        target_id = resolved[0] if resolved else str(node_id)
        bounded_limit, summary_chars = _page_bounds(limit, full_summaries)
        bounded_offset = clamp_offset(offset)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.get_related_entities(
                project,
                target_id,
                normalized_direction,
                relation_types,
                page_limit,
                page_offset,
            ),
            filters,
            bounded_limit,
            bounded_offset,
            keep=None if include_tests else _not_test,
        )
        filtered_rows = [
            {"direction": row["direction"], **_relation_item(row, entity_ref(row["entity"], summary_max_chars=summary_chars))}
            for row in rows
        ]
        meta: dict[str, object] = {
            "project": project,
            "node_id": target_id,
            "direction": normalized_direction,
            "relation_types": relation_types or [],
            "path_filters": filters.meta(),
            "include_tests": include_tests,
        }
        if full_summaries:
            meta["full_summaries"] = True
        if resolved is not None:
            # Present only on the name path: their presence *is* the statement "this id was
            # resolved from a name, not given" (INV-7's flag convention).
            meta["resolved_node_id"], meta["name_match"] = resolved
        payload = paginated_response(
            filtered_rows,
            bounded_limit,
            bounded_offset,
            meta,
            truncated=truncated,
        )
        _flag_items(payload["items"])
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)

    @mcp.tool(description=SEARCH_ENTITIES_DESCRIPTION)
    async def search_entities(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Name substring search (published description: SEARCH_ENTITIES_DESCRIPTION)."""
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        rows, truncated = await _fetch_filtered_page(
            lambda page_limit, page_offset: repo.search_entities(project, query, node_types, page_limit, page_offset),
            filters,
            bounded_limit,
            bounded_offset,
            keep=None if include_tests else _not_test,
        )
        items = [ref for row in rows if (ref := entity_ref(row))]
        meta = _listing_meta(project, filters, node_types, include_tests, False)
        meta["query"] = query
        payload = paginated_response(items, bounded_limit, bounded_offset, meta, truncated=truncated)
        _flag_items(payload["items"])
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)


def _relation_item(row: dict[str, object], ref: dict[str, object] | None) -> dict[str, object]:
    """A relation row as published: ``{relation, entity}``, plus ``import`` on IMPORTS edges that carry it."""
    item: dict[str, object] = {"relation": row["relation"], "entity": ref}
    if row.get("import"):
        item["import"] = row["import"]
    return item


def _listing_meta(
    project: str,
    filters: PathFilterConfig,
    node_types: list[str] | None,
    include_tests: bool,
    full_summaries: bool,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "project": project,
        "path_filters": filters.meta(),
        "node_types": node_types or [],
        "include_tests": include_tests,
    }
    if full_summaries:
        meta["full_summaries"] = True
    return meta
