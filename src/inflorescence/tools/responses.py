"""Shared public response helpers for MCP tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_GRAPH_LIMIT = 500
MAX_SEARCH_LIMIT = 100

# Summary preview bound for navigation tools (list/structure/related): a page of
# hundreds of entities each carrying a multi-paragraph summary overflows tool
# token limits, so those tools ship a one-line preview and point at
# get_entity_context for the full text.
NAV_SUMMARY_MAX_CHARS = 200

_ENTITY_FIELDS = ("id", "name", "node_type", "file_path", "line_start", "line_end", "summary")


def clamp_limit(limit: int, max_limit: int = MAX_GRAPH_LIMIT) -> int:
    """Clamp a user limit into a positive bounded value."""
    return min(max(int(limit or 1), 1), max_limit)


def clamp_offset(offset: int) -> int:
    """Clamp a user offset into a non-negative value."""
    return max(int(offset or 0), 0)


def truncate_summary(text: str, max_chars: int) -> str:
    """Truncate a summary to at most max_chars, appending an ellipsis."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def entity_ref(raw: Mapping[str, Any] | None, summary_max_chars: int | None = None) -> dict[str, Any] | None:
    """Return the compact public representation of a Code node.

    summary_max_chars bounds the summary to a preview (navigation tools);
    None keeps the full summary (search relevance, single-entity focus).
    """
    if raw is None:
        return None
    ref: dict[str, Any] = {}
    for field in _ENTITY_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if field == "summary" and summary_max_chars is not None:
            value = truncate_summary(str(value), summary_max_chars)
        ref[field] = value
    return ref


def page_info(item_count: int, limit: int, offset: int, truncated: bool = False) -> dict[str, Any]:
    """Build page metadata from a limit-plus-one item count.

    ``truncated=True`` marks a page whose underlying scan hit an internal ceiling
    (a filtered-vector candidate cap, or MAX_FILTER_SCAN_ROWS) before the full
    candidate set was examined. It tells the caller that an empty or short page is
    *not* proof the project has no more matches — the distinction INV-7 demands
    between "no matches" and "the scan stopped early". Only present when True.
    """
    bounded_limit = clamp_limit(limit)
    bounded_offset = clamp_offset(offset)
    has_more = item_count > bounded_limit
    info = {
        "limit": bounded_limit,
        "offset": bounded_offset,
        "next_offset": bounded_offset + bounded_limit if has_more else None,
        "has_more": has_more,
    }
    if truncated:
        info["truncated"] = True
    return info


def paginated_response(
    items: Sequence[Mapping[str, Any]],
    limit: int,
    offset: int,
    meta: Mapping[str, Any] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    """Return a shared paginated envelope, trimming a limit-plus-one item list.

    ``truncated`` flows into ``page`` to signal an early-stopped scan (see
    :func:`page_info`); it defaults to False so existing responses are unchanged.
    """
    bounded_limit = clamp_limit(limit)
    bounded_offset = clamp_offset(offset)
    public_items = list(items[:bounded_limit])
    return {
        "items": public_items,
        "page": page_info(len(items), bounded_limit, bounded_offset, truncated),
        "meta": dict(meta or {}),
    }


def error_response(code: str, message: str) -> dict[str, dict[str, str]]:
    """Return a structured MCP error payload."""
    return {"error": {"code": code, "message": message}}
