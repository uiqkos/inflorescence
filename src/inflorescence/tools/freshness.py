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

The block carries one more honesty signal, ``live_worktrees``. Every file of the main
checkout can hash fresh while a branch checked out in one of its git worktrees holds a
completely different version of that code: the worktree is a *separate project*, indexed
separately, and nothing in this index knows about it. A clean freshness block would then
read as "you are seeing everything" to a reviewer asking the main checkout about work that
lives on a branch. So a query against a main checkout that has live worktrees says how
many — the answer is elsewhere, and one query with ``directory`` set to the worktree gets
it (indexing a worktree is near-free, seeded from this very index). Present only when the
queried directory is *not* itself a worktree and the count is non-zero, and phrased as a
sentence rather than a number: a bare count beside ``checked``/``stale`` reads as a defect
count for *this* answer, which is the opposite of what it means.

Cost: one MD5 per *distinct* path on the page (memoized here), so a 50-item page of one
module reads one file. Nothing outside the page is scanned. The worktree count adds one
directory listing of ``.git/worktrees`` per tool call.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from inflorescence.code_indexer.models import FileChecksum
from inflorescence.worktree import registered_worktree_paths, worktree_main_root

logger = logging.getLogger(__name__)

FRESH = "fresh"
STALE = "stale"
MISSING = "missing"
UNKNOWN = "unknown"

# The per-item flag each status raises, if any. ``unknown`` is absent on purpose (see
# the module docstring): it is a meta-only signal.
_ITEM_FLAG = {STALE: "stale", MISSING: "file_missing"}


def _live_worktrees_notice(count: int) -> str:
    """Phrase the worktree count as the warning it is, not as a bare number.

    A field reading ``"live_worktrees": 55`` sitting next to ``checked``/``stale`` counts was
    read in the field as a *property of this answer* — "55 of something here is off" — and
    agents stopped to re-check which tree they were querying. The count alone cannot say
    which; the sentence can, and it carries the next action inside it, so the meaning does
    not depend on the reader having the tool description in context.
    """
    noun = "worktree is" if count == 1 else "worktrees are"
    return f"{count} {noun} not in this index; pass a worktree directory to query its branch state"


def count_live_worktrees(root: Path) -> int:
    """How many linked worktrees of *root*'s checkout currently exist on disk.

    Zero when *root* is itself a worktree (its own answer *is* the branch state, so there
    is nothing to warn about) and zero outside git entirely. Registration outlives the
    directory — git drops an entry only on ``git worktree prune`` — so deleted checkouts
    are counted out: a path nobody can query is not a gap in the answer.
    """
    try:
        if worktree_main_root(root) is not None:
            return 0
        return sum(1 for path in registered_worktree_paths(root) if path.exists())
    except OSError:  # an unreadable .git layout is not worth failing a search over
        logger.debug("Could not count worktrees under %s", root, exc_info=True)
        return 0


class FreshnessChecker:
    """Re-hashes the files behind one page of entity refs and annotates them.

    Built once per tool call — ``stored`` costs a single ``get_checksums`` round trip for
    the whole project — then asked about whatever paths that call happens to return.
    """

    def __init__(self, root: Path, stored: Mapping[str, str], live_worktrees: int = 0) -> None:
        self._root = root
        self._stored = dict(stored)
        self._live_worktrees = live_worktrees
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
        ``live_worktrees`` appears only when there are any — an absent field costs no
        tokens and says the same thing a zero would — and its value is a self-describing
        sentence rather than a count (see :func:`_live_worktrees_notice`).
        """
        buckets: dict[str, list[str]] = {STALE: [], MISSING: [], UNKNOWN: []}
        for file_path, status in self._statuses.items():
            if status in buckets:
                buckets[status].append(file_path)
        block: dict[str, Any] = {
            "checked": len(self._statuses),
            "stale": sorted(buckets[STALE]),
            "missing": sorted(buckets[MISSING]),
            "unknown": sorted(buckets[UNKNOWN]),
        }
        if self._live_worktrees:
            block["live_worktrees"] = _live_worktrees_notice(self._live_worktrees)
        return block

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


def unchecked_empty_page(directory: str | None = None) -> dict[str, Any]:
    """The ``index_freshness`` block for a response that returns no entities at all.

    A degenerate page still carries the block — the "unflagged means fresh" convention
    only reads if it is unconditional — and says plainly that it checked nothing. Built
    from an empty checker so its shape cannot drift from the real one. Pass *directory*
    to keep the ``live_worktrees`` warning on this page too: an empty answer is exactly
    where "the branch state is elsewhere" is most worth saying.
    """
    if directory is None:
        return FreshnessChecker(Path(), {}).meta()
    root = Path(directory).resolve()
    return FreshnessChecker(root, {}, live_worktrees=count_live_worktrees(root)).meta()


async def build_checker(repo: Any, project: str, directory: str) -> FreshnessChecker:
    """Load a project's stored checksums once and wrap them in a checker.

    One ``get_checksums`` per tool call, by construction: tools build the checker once
    and reuse it for every ref they return.
    """
    stored = await repo.get_checksums(project)
    root = Path(directory).resolve()
    return FreshnessChecker(root, stored, live_worktrees=count_live_worktrees(root))
