"""Graph navigation MCP tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import build_checker
from inflorescence.tools.gates import ensure_indexed, normalize_node_types
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


def _structure_root_missing(directory: str, node_id: str | None, file_path: str | None) -> str:
    """Explain a missing structure root, checking the filesystem before blaming the caller.

    ``file_path`` is the one argument this server can verify on the spot, and the two
    readings of "not found" are worlds apart: a typo'd path, or a real file that was never
    indexed. Answering both with the same three words let an agent conclude a 174-line
    module did not exist (field report, 10.09.2026) — one ``exists()`` call tells them apart.
    """
    if not file_path:
        return (
            f"No indexed entity with node_id={node_id!r}. Ids come from a search or listing "
            "in this project; pass file_path= to address a file instead."
        )
    try:
        on_disk = (Path(directory) / file_path).exists()
    except OSError:  # an unreadable path is not evidence either way
        on_disk = False
    if on_disk:
        return (
            f"{file_path!r} exists on disk but is NOT in this index, so nothing about it is "
            "searchable here — it was created after the last index run, or an exclude pattern "
            "or .gitignore keeps it out. Read it from the filesystem, and run "
            f"index_directory({directory!r}) to index it. This is not 'the file has no structure'."
        )
    return (
        f"No indexed entity for file_path={file_path!r}, and no such file on disk under "
        f"{directory!r} either. Paths are relative to the project root; list_entities("
        f"node_types=[\"module\", \"document\"]) shows which files this index holds."
    )


async def _resolve_name(repo: GraphRepository, project: str, name: str) -> str | dict:
    """Resolve a name substring to exactly one node id, or to the error explaining why not.

    Returns the id on success and an ``error_response`` payload otherwise, so the caller
    checks ``isinstance(result, dict)``. Ambiguity is never resolved by picking the first
    match: "who calls X" answered about the wrong X is a wrong answer that looks right, and
    the caller cannot tell. The error carries the candidates instead, so the next call is
    exact rather than another guess.
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
    if len(rows) == 1:
        return str(rows[0]["id"])
    shown = [_describe_candidate(row) for row in rows[:MAX_NAME_CANDIDATES]]
    surplus = len(rows) - len(shown)
    more = ""
    if surplus:
        more = f", and {surplus}+ more" if len(rows) > _NAME_CANDIDATE_SCAN else f", and {surplus} more"
    return error_response(
        "ambiguous_name",
        f"{name!r} matches more than one entity — re-call with node_id set to one of: "
        f"{'; '.join(shown)}{more}.",
    )


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

        For inventory and orientation — "what is in this package", "how many classes are
        under src/" — not for finding code by meaning (``search_semantic``) or by name
        (``search_entities``); this returns entities in path order, unranked. Combine
        ``include_paths``/``exclude_paths`` globs with ``node_types`` to scope the listing.
        ``node_types`` values (case-insensitive): module, class, function, method,
        directory, interface, enum, struct, trait, document. Summaries are truncated to a
        short preview; use ``get_entity_context`` for the full summary of one entity.
        A path pattern with no glob character (``"docs/"``, ``"docs"``) covers both that
        exact path and everything under it; ``meta.path_filters`` echoes the expanded form.

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected items carry ``"stale": true`` / ``"file_missing": true`` —
        so an unflagged item was checked and still matches the index. This is file-level:
        "stale" means the file changed after indexing (lines may have shifted, the summary
        may describe older code), not that this particular entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
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
            lambda page_limit, page_offset: repo.list_entities(project, node_types, page_limit, page_offset),
            filters,
            bounded_limit,
            bounded_offset,
        )
        items = [ref for row in rows if (ref := entity_ref(row, summary_max_chars=NAV_SUMMARY_MAX_CHARS))]
        payload = paginated_response(
            items,
            bounded_limit,
            bounded_offset,
            {"project": project, "path_filters": filters.meta(), "node_types": node_types or []},
            truncated=truncated,
        )
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)

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

        The outline view: what a file or class contains, one to three levels deep. Use it
        after locating something ("show me this module's classes and methods"); for the
        edges *between* entities — callers, callees, imports — use ``get_related_entities``
        or ``get_entity_context`` instead. ``node_types`` filters the items only, never the
        root; valid values (case-insensitive): module, class, function, method, directory,
        interface, enum, struct, trait, document. Item summaries are truncated to a short
        preview (the root keeps its full summary).

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page *including the root*, and affected items carry ``"stale": true`` /
        ``"file_missing": true`` — so an unflagged item was checked and still matches the
        index. This is file-level: "stale" means the file changed after indexing (lines may
        have shifted, the summary may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        ``meta.nested_checkouts`` appears when the files on this page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) — that
        checkout's HEAD, not ``answered_from``, is the code you are reading.

        A ``not_found`` error here distinguishes the two cases that matter: a ``file_path``
        that does not exist under this directory at all, and one that exists on disk but was
        never indexed (created after the last index run, or excluded) — the second is a
        coverage gap, so read the file directly and ``index_directory`` to make it queryable.
        """
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        root = await repo.get_structure_root(project, node_id=node_id, file_path=file_path)
        if root is None:
            return error_response("not_found", _structure_root_missing(directory, node_id, file_path))
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
        checker = await build_checker(repo, project, directory)
        # The root is annotated before ``stamp``, so its file counts in index_freshness.
        payload["root"] = checker.annotate(entity_ref(root))
        return checker.stamp(payload)

    @mcp.tool()
    async def get_entity_context(
        directory: str,
        node_id: str,
        relation_types: list[str] | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict:
        """Return parent, children, incoming, and outgoing context for an entity.

        The one-call deep dive on a known ``node_id``: full (untruncated) summary plus its
        neighbourhood in every direction. Use it on a hit from any search tool; for a whole
        subtree use ``get_entity_structure``, and for one edge kind over many hops use
        ``get_related_entities``. ``relation_types`` filters the ``incoming`` and
        ``outgoing`` sections to those relationship types (e.g. ``["CALLS"]``); ``parent``
        and the CONTAINS ``children`` are the structural skeleton and are always returned.
        ``offset`` paginates each section, so walking ``next_offset`` terminates instead of
        looping.

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers every file
        named here — entity, parent and all sections — and affected refs carry
        ``"stale": true`` / ``"file_missing": true``, so an unflagged one was checked and
        still matches the index. This is file-level: "stale" means the file changed after
        indexing (lines may have shifted, the summary may describe older code), not that
        this entity changed. ``index_freshness.live_worktrees``, when present, is a
        sentence naming how many git worktrees of this checkout are NOT in this index —
        their branch state is elsewhere; ask with ``directory`` set to the worktree itself,
        which indexes it almost for free (no LLM spend).
        """
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
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

    @mcp.tool()
    async def get_related_entities(
        directory: str,
        node_id: str | None = None,
        name: str | None = None,
        direction: str = "both",
        relation_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Return entities related to a node by graph relationship.

        Name the entity with **exactly one** of ``node_id`` (an id from any search tool) or
        ``name`` (a case-insensitive substring of an entity's name, resolved here). Passing
        both, or neither, returns ``invalid_argument``. ``name`` is the one-call form of
        "who calls X": ``get_related_entities(directory, name="load_board",
        direction="incoming", relation_types=["CALLS"])``. It is resolved the way
        ``search_entities`` matches, and ``meta.resolved_node_id`` reports what it resolved
        to. A substring matching several entities is never resolved by picking one: the call
        fails with ``ambiguous_name`` and lists the candidates (id, node_type, file_path) so
        the next call can pass an exact ``node_id``; a substring matching nothing fails with
        ``not_found``.

        ``direction`` selects which edges to follow relative to that entity:
        ``"both"`` (default), ``"outgoing"`` (alias ``"out"``), or ``"incoming"``
        (alias ``"in"``). Matching is case-insensitive. Any other value returns
        an ``invalid_direction`` error instead of an empty result.

        Summaries are truncated to a short preview; use ``get_entity_context``
        for the full summary of a specific entity. A path pattern with no glob character
        (``"docs/"``, ``"docs"``) covers both that exact path and everything under it;
        ``meta.path_filters`` echoes the expanded form.

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected entities carry ``"stale": true`` / ``"file_missing": true``
        — so an unflagged one was checked and still matches the index. This is file-level:
        "stale" means the file changed after indexing (lines may have shifted, the summary
        may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
        # Argument validation before anything that touches the DB or auto-indexes.
        if bool(node_id) == bool(name):
            return error_response(
                "invalid_argument",
                "Pass exactly one of node_id (an id from a search result) or name (a "
                "case-insensitive substring of an entity's name), not both and not neither.",
            )
        normalized_direction = _normalize_direction(direction)
        if isinstance(normalized_direction, dict):
            return normalized_direction
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        resolved_node_id: str | None = None
        if name:
            resolved = await _resolve_name(repo, project, name)
            if isinstance(resolved, dict):
                return resolved
            resolved_node_id = resolved
        target_id = resolved_node_id or str(node_id)
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
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
        )
        filtered_rows = []
        for row in rows:
            entity = row["entity"]
            filtered_rows.append({
                "relation": row["relation"],
                "direction": row["direction"],
                "entity": entity_ref(entity, summary_max_chars=NAV_SUMMARY_MAX_CHARS),
            })
        meta: dict[str, object] = {
            "project": project,
            "node_id": target_id,
            "direction": normalized_direction,
            "relation_types": relation_types or [],
            "path_filters": filters.meta(),
        }
        if resolved_node_id is not None:
            # Present only on the name path: its presence *is* the statement "this id was
            # resolved from a name, not given" (INV-7's flag convention).
            meta["resolved_node_id"] = resolved_node_id
        payload = paginated_response(
            filtered_rows,
            bounded_limit,
            bounded_offset,
            meta,
            truncated=truncated,
        )
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)

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
        """Search code entities by case-insensitive substring of their name.

        Pure name matching — ``"handl"`` finds ``handle_request`` and ``ErrorHandler`` — so
        use it when you already know (part of) an identifier, and when a name-based answer
        must be complete the moment a first-time index finishes parsing, before any summary
        exists. It never looks inside bodies, signatures, or summaries: for a string
        literal, an error message, or any text in the code use ``search_text``; to find code
        by what it does use ``search_semantic``. ``node_types`` values (case-insensitive):
        module, class, function, method, directory, interface, enum, struct, trait,
        document. A path pattern with no glob character (``"docs/"``, ``"docs"``) covers
        both that exact path and everything under it; ``meta.path_filters`` echoes the
        expanded form.

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected items carry ``"stale": true`` / ``"file_missing": true`` —
        so an unflagged item was checked and still matches the index. This is file-level:
        "stale" means the file changed after indexing (lines may have shifted, the summary
        may describe older code), not that this particular entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
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
        )
        items = [ref for row in rows if (ref := entity_ref(row))]
        payload = paginated_response(
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
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)
