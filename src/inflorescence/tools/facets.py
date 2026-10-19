"""``meta.facets``: where an answer's candidates come from, counted — facts, not advice.

A page can be technically correct and still be the wrong page: field reports kept describing
tests out-ranking the implementation they test, a word popular in the repository's own notes
burying the code, four copies of an old prototype taking 9 of 12 places. Each time the fix was
one more call with a narrower scope — ``exclude_paths`` for the tree that is not the answer,
``include_paths`` for the one that is. What the caller lacked was not a rule telling it which
parameter to pass but the shape of the candidate set, which a page of ten hides.

So every search and ``find_text`` reports, over everything the page was drawn from (not just
the page), how many candidates sit under each directory, of which node types, and how many are
test code. The caller decides what that means for its question; nothing here filters, reorders
or recommends. Directories are counted one level below the deepest directory all candidates
share, so the breakdown stays informative at any scope: a whole repository splits into its top
level, ``include_paths=["frontend/src/"]`` splits ``frontend/src`` into its parts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Directories listed by name; the rest are summed into ``other_directories``.
FACET_DIRECTORIES = 8


def _entity(item: Mapping[str, Any]) -> Mapping[str, Any]:
    entity = item.get("entity")
    return entity if isinstance(entity, Mapping) else item


def _parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").strip("/").split("/") if part and part != "."]


def _common_directory(paths: Sequence[list[str]]) -> list[str]:
    """The deepest directory every path is under (a file's own name never counts)."""
    if not paths:
        return []
    common = paths[0][:-1]
    for parts in paths[1:]:
        directory = parts[:-1]
        depth = 0
        while depth < min(len(common), len(directory)) and common[depth] == directory[depth]:
            depth += 1
        common = common[:depth]
        if not common:
            break
    return common


def directory_facets(path_counts: Mapping[str, int]) -> dict[str, Any]:
    """Counts per directory one level below the candidates' common directory.

    ``path_counts`` maps a file path to how many candidates it holds. Files directly in the
    common directory are counted under it with ``"direct": true``.
    """
    split = {path: _parts(path) for path in path_counts}
    base = _common_directory(list(split.values()))
    counts: Counter[str] = Counter()
    direct = 0
    for path, parts in split.items():
        below = parts[len(base):]
        if len(below) > 1:
            counts["/".join([*base, below[0]]) + "/"] += path_counts[path]
        else:
            direct += path_counts[path]
    ranked = counts.most_common()
    listed = [{"path": path, "count": count} for path, count in ranked[:FACET_DIRECTORIES]]
    if direct:
        listed.append({"path": ("/".join(base) + "/") if base else "./", "count": direct, "direct": True})
        listed.sort(key=lambda entry: -entry["count"])
    result: dict[str, Any] = {"common_path": ("/".join(base) + "/") if base else "./", "by_directory": listed}
    rest = sum(count for _, count in ranked[FACET_DIRECTORIES:])
    if rest:
        result["other_directories"] = {"directories": len(ranked) - FACET_DIRECTORIES, "count": rest}
    return result


def hit_facets(items: Iterable[Mapping[str, Any]], *, count_tests: bool) -> dict[str, Any]:
    """Facets over ranked candidates (search hits or listed entities): one per entity.

    ``count_tests`` is false when the caller already dropped test paths — a count that can only
    be zero says nothing.
    """
    candidates = list(items)
    if not candidates:
        return {}
    paths: Counter[str] = Counter()
    node_types: Counter[str] = Counter()
    tests = 0
    for item in candidates:
        entity = _entity(item)
        paths[str(entity.get("file_path") or item.get("file_path") or "")] += 1
        node_type = entity.get("node_type") or item.get("node_type")
        if node_type:
            node_types[str(node_type)] += 1
        if item.get("is_test"):
            tests += 1
    facets: dict[str, Any] = {"counted": len(candidates), **directory_facets(paths)}
    if node_types:
        facets["by_node_type"] = dict(node_types.most_common())
    if count_tests:
        facets["tests"] = tests
    return facets
