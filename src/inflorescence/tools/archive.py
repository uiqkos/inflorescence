"""Paths a repository declares to be history: indexed, searchable, and ranked below live code.

Most repositories keep some code that is no longer the code — archived prototypes, a copy of
the previous design kept for reference, a legacy tree nobody may develop. Its summaries use
exactly the vocabulary of the live feature, so a semantic search about that feature is
dominated by it: a field report (17.09.2026) measured 9 of 12 hits from four near-identical
copies of an old prototype, with the two real answers ranked 7th and 8th. The repository's own
notes said in bold that the tree was history. The index had no way to know.

Ignoring the tree (``.inflorescenceignore``) is the wrong answer: history is still worth
finding when it is what you ask for — a twin of a bug, the rationale of an old decision. So
the repository says it once, in a committed file, and every search does two things with it:
a hit from such a path carries ``"archived": true``, and archived hits are listed after every
live hit on the same query instead of interleaved with them. Nothing is removed; excluding the
tree for one query is still ``exclude_paths``.

The file is ``.inflorescencearchive`` at the repository root, in gitignore syntax — the same
convention as ``.inflorescenceignore``, so the rule travels with the repository instead of
living in one machine's database. It is read at query time (and cached by modification time),
so an edit takes effect on the next search without re-indexing anything.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pathspec

logger = logging.getLogger(__name__)

ARCHIVE_FILE = ".inflorescencearchive"

# root -> (mtime_ns of the file, compiled spec or None). A search reads the spec on every
# call; the stat that validates the cache entry is the whole cost when nothing changed.
_cache: dict[Path, tuple[int, pathspec.PathSpec | None]] = {}


@dataclass(frozen=True)
class ArchiveSpec:
    """The compiled archive patterns of one repository root."""

    spec: pathspec.PathSpec

    def covers(self, file_path: str) -> bool:
        normalized = str(file_path).replace("\\", "/").lstrip("/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return bool(normalized) and self.spec.match_file(normalized)


def load_archive(root: Path) -> ArchiveSpec | None:
    """The archive declared at *root*, or ``None`` when there is no such file (or it is empty).

    Never raises: an unreadable file is logged and treated as absent, because a search must
    not fail over a ranking preference.
    """
    path = root / ARCHIVE_FILE
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        _cache.pop(root, None)
        return None
    cached = _cache.get(root)
    if cached is not None and cached[0] == mtime:
        return ArchiveSpec(cached[1]) if cached[1] is not None else None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
    except (OSError, ValueError):
        logger.warning("Could not read %s — ranking without it", path, exc_info=True)
        return None
    compiled = spec if len(spec) else None
    _cache[root] = (mtime, compiled)
    return ArchiveSpec(compiled) if compiled is not None else None


def _hit_path(item: MutableMapping[str, Any]) -> str:
    entity = item.get("entity") if isinstance(item.get("entity"), MutableMapping) else item
    return str((entity or {}).get("file_path") or "")


def demote_archived(
    hits: Sequence[MutableMapping[str, Any]], archive: ArchiveSpec | None
) -> tuple[list[MutableMapping[str, Any]], int]:
    """Flag archived hits and move them after the live ones, keeping each group's order.

    Returns the reordered list and how many hits were archived. Applied to the whole ranked
    candidate list *before* it is paged, so page 1 holds the best live hits and pagination
    stays stable across pages.
    """
    if archive is None:
        return list(hits), 0
    live: list[MutableMapping[str, Any]] = []
    archived: list[MutableMapping[str, Any]] = []
    for hit in hits:
        if archive.covers(_hit_path(hit)):
            hit["archived"] = True
            archived.append(hit)
        else:
            live.append(hit)
    return live + archived, len(archived)


def flag_archived(items: Sequence[MutableMapping[str, Any]], archive: ArchiveSpec | None) -> int:
    """Flag archived items in place without reordering (for unranked, path-ordered listings)."""
    if archive is None:
        return 0
    flagged = 0
    for item in items:
        if archive.covers(_hit_path(item)):
            # On the row itself, like ``demote_archived`` and ``is_test`` — a relation row's
            # flag sits beside its ``entity``, not inside it.
            item["archived"] = True
            flagged += 1
    return flagged
