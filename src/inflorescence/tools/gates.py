"""Preconditions every read tool checks before it touches the graph.

Both guards here answer the same INV-7 question — "would an empty page be an honest
answer?" — and both must answer it identically for every tool, so they live in one
module instead of being re-implemented per tool file:

* :func:`ensure_indexed` — the project actually holds an index (or the manager just
  built one, for free, from a donor checkout).
* :func:`normalize_node_types` — the caller's ``node_types`` name real node types.

A guard that only some tools apply is worse than none: it teaches the caller that an
empty page means "no matches" and then hands them one that means "you typo'd".
"""

from __future__ import annotations

from inflorescence.code_indexer.models import NodeType
from inflorescence.project_manager import ProjectManager
from inflorescence.tools.responses import error_response

# The canonical node types, in NodeType declaration order — the exact strings stored on
# nodes (and lower-cased labels in Cypher). Derived from the enum so a new node type
# cannot be added without the tools' error messages learning about it.
VALID_NODE_TYPES: tuple[str, ...] = tuple(node_type.value for node_type in NodeType)

_NODE_TYPE_BY_LOWER = {value.lower(): value for value in VALID_NODE_TYPES}

# Names that stand for a *group* of node types. One exists because the question behind it is
# one of the most common an agent asks — "is there an X in this codebase", as opposed to "is X
# mentioned in the design notes" — and spelling the group out type by type was a hint nobody
# took (field reports, 15-17.09.2026: a keyword popular in a repository's own docs returned a
# page of markdown and no code, and the agent could not tell whether code existed). Derived
# from the enum, so a node type added later lands in the group without anyone remembering to.
NODE_TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "code": tuple(
        value for value in VALID_NODE_TYPES
        if value not in (NodeType.DOCUMENT.value, NodeType.DIRECTORY.value)
    ),
}


async def ensure_indexed(manager: ProjectManager, directory: str) -> tuple[str, dict | None]:
    """Resolve *directory* to a queryable project, or to the error explaining why it is not.

    ``get_project`` synthesises a valid id from any path, so a typo or an adjacent directory
    otherwise yields an empty result indistinguishable from a genuine "no matches" (audit
    finding T5 / INV-7). Surfacing it as an error tells the caller what to do instead of
    letting them trust a silent empty page.

    Whether the directory can instead be indexed on the spot — a fresh git worktree of an
    indexed checkout, which seeds from that checkout's caches and costs no LLM spend — is the
    manager's call, not a tool's; see :meth:`ProjectManager.ensure_indexed_for_query`. The
    tool only forwards the reason it gets back, including the "auto-index attempted and
    failed: …" one, which is why a failed attempt still reaches the caller as a structured
    error rather than an exception.
    """
    project, reason = await manager.ensure_indexed_for_query(directory)
    if reason is None:
        return project, None
    return project, error_response(
        "project_not_indexed",
        f"No index found for {directory!r} (project '{project}') — {reason}. This is different "
        "from an indexed project that simply has no matches.",
    )


def normalize_node_types(node_types: list[str] | None) -> list[str] | dict | None:
    """Normalize ``node_types`` to canonical NodeType values, case-insensitively.

    Returns the normalized list (``None`` stays ``None``: no filter at all), or an
    ``error_response`` payload (dict) when any value is not a node type. A group name
    (:data:`NODE_TYPE_GROUPS`, e.g. ``"code"``) expands to its members, and the expanded list
    is what the tools echo in ``meta.node_types`` — the filter as actually applied. Callers must
    check ``isinstance(result, dict)`` and return it as-is **before** querying.

    Same reasoning as ``direction`` in graph.py: unvalidated, ``node_types=["func"]``
    matches no label and the tool answers with a confident empty page, which reads as
    "this project has no such code" rather than "that is not a node type" (INV-7).
    """
    if node_types is None:
        return None
    normalized: list[str] = []
    for value in node_types:
        key = str(value).strip().lower()
        group = NODE_TYPE_GROUPS.get(key)
        if group is not None:
            normalized.extend(member for member in group if member not in normalized)
            continue
        canonical = _NODE_TYPE_BY_LOWER.get(key)
        if canonical is None:
            groups = ", ".join(f"{name!r} ({', '.join(members)})" for name, members in NODE_TYPE_GROUPS.items())
            return error_response(
                "invalid_node_type",
                f"{value!r} is not a valid node type; valid: {', '.join(VALID_NODE_TYPES)}, or the "
                f"group {groups}. Matching is case-insensitive.",
            )
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized
