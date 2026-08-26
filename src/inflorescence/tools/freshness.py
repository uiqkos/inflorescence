"""Whether the files behind a page of entities still match what was indexed.

Every module/document node carries the MD5 of its file as of indexing
(``repo.get_checksums``). A tool answering from the graph is therefore answering from a
snapshot: the file may have been edited, or deleted, since. Left unsaid, that snapshot
reads as the present — the caller trusts ``line_start``/``line_end`` and a summary that
may describe code which no longer exists.

So every tool that returns entity refs re-hashes the files on the page at return time and
reports what it found:

* ``fresh`` — the file on disk still hashes to the stored checksum;
* ``stale`` — it changed after indexing;
* ``missing`` — it is gone from disk (a ghost node the reconcile has not swept yet);
* ``unknown`` — it is unreadable, or nothing was stored for it (never a module/document
  node, so no checksum exists).

The report is deliberately two-part. ``meta.index_freshness`` is present on *every* such
response, empty lists and all, and only that constancy makes the per-item flags readable
as "everything unflagged is fresh" rather than "this tool may or may not have checked".
``unknown`` stays in meta alone: an item flag there would be noise on a page where most
entities have no stored checksum by construction.

This is a **file-level** check, not a node-level one. "stale" means the file changed —
line numbers may have shifted and the summary may describe older code — not that this
particular function or class is what changed.

Cost: one MD5 per *distinct* path on the page (memoized here), so a 50-item page of one
module reads one file. Nothing outside the page is scanned.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from inflorescence.code_indexer.models import FileChecksum

FRESH = "fresh"
STALE = "stale"
MISSING = "missing"
UNKNOWN = "unknown"

# The per-item flag each status raises, if any. ``unknown`` is absent on purpose (see
# the module docstring): it is a meta-only signal.
_ITEM_FLAG = {STALE: "stale", MISSING: "file_missing"}


class FreshnessChecker:
    """Re-hashes the files behind one page of entity refs and annotates them.

    Built once per tool call — ``stored`` costs a single ``get_checksums`` round trip for
    the whole project — then asked about whatever paths that call happens to return.
    """

    def __init__(self, root: Path, stored: Mapping[str, str]) -> None:
        self._root = root
        self._stored = dict(stored)
        # path -> status; doubles as the "already read this file" memo and as the record
        # of which paths this page actually touched (`checked` counts its keys).
        self._statuses: dict[str, str] = {}

    def status(self, file_path: str) -> str:
        """Return the freshness status of one root-relative path, hashing it at most once."""
        cached = self._statuses.get(file_path)
        if cached is not None:
            return cached
        status = self._compute(file_path)
        self._statuses[file_path] = status
        return status

    def _compute(self, file_path: str) -> str:
        stored = self._stored.get(file_path)
        if stored is None:
            # Nothing to compare against — reading the file would tell us nothing.
            return UNKNOWN
        path = self._root / file_path
        try:
            checksum = FileChecksum.try_compute(path, self._root)
        except ValueError:
            # file_path escaped the root (absolute, or '..'-prefixed): not something we
            # can relate to this project's tree, so we decline to judge it.
            return UNKNOWN
        if checksum is not None:
            return FRESH if checksum.md5 == stored else STALE
        # try_compute swallowed an OSError: gone, or merely unreadable (permissions, a
        # racing checkout). Only the first is evidence of a ghost node.
        try:
            exists = path.exists()
        except OSError:
            exists = True
        return UNKNOWN if exists else MISSING

    def annotate(self, ref: MutableMapping[str, Any] | None) -> MutableMapping[str, Any] | None:
        """Flag one entity ref in place when its file is stale or missing; returns it.

        Accepts ``None`` (``entity_ref`` returns it for absent nodes) so call sites stay
        one line. Refs without a ``file_path`` are left alone and never counted.
        """
        if not ref:
            return ref
        file_path = ref.get("file_path")
        if not file_path:
            return ref
        flag = _ITEM_FLAG.get(self.status(str(file_path)))
        if flag:
            ref[flag] = True
        return ref

    def annotate_items(self, items: Iterable[Any]) -> None:
        """Annotate a page of items, whether they are entity refs or hits wrapping one.

        Search hits and ``get_related_entities`` rows carry the ref under ``"entity"``;
        list/structure items *are* the ref. No entity ref has an ``"entity"`` field
        (see ``responses._ENTITY_FIELDS``), so the shape is unambiguous.
        """
        for item in items:
            if not isinstance(item, MutableMapping):
                continue
            if "entity" in item:
                self.annotate(item.get("entity"))
            else:
                self.annotate(item)

    def meta(self) -> dict[str, Any]:
        """Return the ``index_freshness`` block for everything checked so far.

        ``checked`` counts *distinct files*, not entities: the page's unique paths.
        """
        buckets: dict[str, list[str]] = {STALE: [], MISSING: [], UNKNOWN: []}
        for file_path, status in self._statuses.items():
            if status in buckets:
                buckets[status].append(file_path)
        return {
            "checked": len(self._statuses),
            "stale": sorted(buckets[STALE]),
            "missing": sorted(buckets[MISSING]),
            "unknown": sorted(buckets[UNKNOWN]),
        }

    def stamp(self, payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Annotate ``payload["items"]`` (if any) and write ``meta.index_freshness``.

        Always call this *last*: the meta block reports only the paths already checked,
        so anything annotated outside ``items`` (a structure root, a context's parent)
        must be annotated first.
        """
        self.annotate_items(payload.get("items") or [])
        meta = payload.setdefault("meta", {})
        meta["index_freshness"] = self.meta()
        return payload


def unchecked_empty_page() -> dict[str, Any]:
    """The ``index_freshness`` block for a response that returns no entities at all.

    A degenerate page still carries the block — the "unflagged means fresh" convention
    only reads if it is unconditional — and says plainly that it checked nothing. Built
    from an empty checker so its shape cannot drift from the real one.
    """
    return FreshnessChecker(Path(), {}).meta()


async def build_checker(repo: Any, project: str, directory: str) -> FreshnessChecker:
    """Load a project's stored checksums once and wrap them in a checker.

    One ``get_checksums`` per tool call, by construction: tools build the checker once
    and reuse it for every ref they return.
    """
    stored = await repo.get_checksums(project)
    return FreshnessChecker(Path(directory).resolve(), stored)
