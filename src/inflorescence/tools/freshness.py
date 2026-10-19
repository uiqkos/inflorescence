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

Directory nodes are exempt: their ``file_path`` names a folder, which has no content to
hash, and judging one only ever produced a meta-level ``unknown`` that read as doubt about
the index.

The report is deliberately two-part. ``meta.index_freshness`` is present on *every* such
response, empty lists and all, and only that constancy makes the per-item flags readable
as "everything unflagged is fresh" rather than "this tool may or may not have checked".
``unknown`` stays in meta alone: an item flag there would be noise on a page where most
entities have no stored checksum by construction.

This is a **file-level** check, not a node-level one. "stale" means the file changed —
line numbers may have shifted and the summary may describe older code — not that this
particular function or class is what changed.

The block also answers the question a clean page provokes rather than settles: *will this
stay true?* ``watcher`` says whether a watch daemon is following this tree right now, is
registered but not running (the state after a reboot), or was never registered — with the
command that changes it. A page whose files all hash fresh and whose watcher is ``none`` is
an index that was correct at some past moment and is drifting from that moment onwards; the
freshness flags alone cannot say that, because they only ever look at the handful of files
this answer happened to return.

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

Alongside the freshness block, every such response carries ``meta.answered_from`` — the
tree the query was answered from, named by path, branch and short commit (see
:func:`describe_origin`). Freshness says whether the answer is current; that line says
*current with what*, which is the question a repository with several worktrees on disk
actually raises. When the files on the page belong to an independent checkout *inside* that
tree — a submodule, a vendored repository — ``meta.nested_checkouts`` names it and its own
HEAD (:func:`describe_nested_checkouts`), because the outer branch says nothing about code
that is on a different one.

One more line covers what the page-sized check cannot: ``drift``. An index records the commit
it last matched the disk at (``indexed_head``). When the checkout's HEAD is a different commit
— a merge, a rebase, a checkout the index did not follow — every indexed file is re-hashed once
(bounded, cached per commit pair) and the block says how many differ, because a merge rewrites
files an answer never returns, and "this page is fresh" then says nothing about the rest.

What this module cannot see is a file that was never indexed: with no node there is no
checksum, so ``checked``/``stale``/``missing`` are all silent about it and a clean block over
a missing file reads as a clean bill of health. That gap is answered separately, on empty
pages, by :mod:`inflorescence.tools.coverage`.

Cost: one MD5 per *distinct* path on the page (memoized here), so a 50-item page of one
module reads one file. Nothing outside the page is scanned. The worktree count adds one
directory listing of ``.git/worktrees`` per tool call, and the origin line a handful of
small reads under ``.git``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from time import monotonic
from typing import Any

from inflorescence.code_indexer.models import FileChecksum, NodeType
from inflorescence.worktree import head_description, head_sha, registered_worktree_paths, worktree_main_root

logger = logging.getLogger(__name__)

FRESH = "fresh"
STALE = "stale"
MISSING = "missing"
UNKNOWN = "unknown"

# The per-item flag each status raises, if any. ``unknown`` is absent on purpose (see
# the module docstring): it is a meta-only signal.
_ITEM_FLAG = {STALE: "stale", MISSING: "file_missing"}

# Directory nodes carry a folder as their ``file_path``; nothing about them is hashable.
_DIRECTORY_NODE_TYPE = NodeType.DIRECTORY.value


def _watcher_notice(state: Mapping[str, Any]) -> str:
    """Say whether anything is keeping this index current, and what to do when nothing is.

    Freshness answers "do the files on THIS page still match" — a page-sized question. The
    reader's next one is bigger: can the rest of the answer be trusted, and will it still be
    true in an hour. A clean page says nothing about that, and a reader with no other signal
    reads a clean page as a current index, which is exactly how a graph built last week gets
    quoted in the present tense.

    Three states, each with its action inside the sentence, because the reader may not have
    the tool description in context: followed right now, registered but with nothing running
    (the state after a reboot — the fix is to start the daemon, not to re-index), and never
    registered (an index that only ever moves when someone re-indexes it by hand).
    """
    if state.get("live"):
        # The clause after the semicolon is not decoration. Read without it, on an empty
        # page, this sentence says "everything on disk is in the index" — and an agent acted
        # on exactly that reading over a file the daemon had never seen (field report,
        # 10.09.2026). A watcher promises to follow saves; it cannot promise coverage, and
        # meta.index_coverage is what actually checks.
        return (
            "live — a watch daemon is following this tree and re-indexes what you save; a file "
            "it has not picked up yet, or one created while it was not running, is simply "
            "absent from the index"
        )
    if state.get("watched") and state.get("daemon_live"):
        # Registered seconds ago and not yet picked up, or picked up by nothing because the
        # daemon refused it. "Start the daemon" is wrong advice in both — one is running.
        return (
            "registered but not yet followed by the running daemon — everything below is as "
            "fresh as the last index until it is; `inflorescence watch status` says why"
        )
    if state.get("watched"):
        return (
            "registered but no daemon is running it — everything below is as fresh as the "
            "last index and no fresher; start it with `inflorescence watch run`"
        )
    return (
        "none — this index changes only when something re-indexes it; follow this tree with "
        "`inflorescence watch add <path>`"
    )


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


def describe_origin(root: Path) -> str:
    """Name the tree this answer came from: path, branch, commit, and worktree lineage.

    Reported as ``meta.answered_from`` on every response that carries entities, because
    "which tree am I even looking at" turned out to be a live question in the field: with
    several worktrees of one repository on disk, two agents an hour apart disagreed about
    whether their branch work was in the graph, and nothing in the response could settle
    it. The path alone answers most of it; the branch and short commit say *when*, so a
    freshness block that reads clean can be read against a specific state of the code
    rather than against "the repository".

    This is the tree the query was answered *from*, not the commit the index was built at
    — those differ exactly when files changed since indexing, which is what
    ``index_freshness`` is for. Outside git, or with metadata this process cannot read,
    the line degrades to the path.
    """
    line = str(root)
    head = head_description(root)
    if head:
        line = f"{line} ({head})"
    try:
        main_root = worktree_main_root(root)
    except OSError:
        main_root = None
    if main_root is not None:
        line = f"{line}, a git worktree of {main_root}"
    return line


# How many distinct directories a page's paths may be traced up through when looking for a
# nested checkout. A page is at most 100 items and its files cluster in a handful of folders,
# so this is generous; it exists so a pathological page cannot turn one answer into hundreds
# of `stat` calls.
_MAX_SUBPROJECT_DIRS = 24


def describe_nested_checkouts(root: Path, rel_paths: Iterable[str]) -> list[str]:
    """Name any independent git checkout the answer's files actually live in.

    ``answered_from`` names the tree the query was addressed to. In a monorepo that vendors
    its subprojects as submodules — or simply holds nested repositories — that is the *outer*
    checkout, and its branch says nothing about the code being returned: a field report
    (10.09.2026) has ``answered_from`` reading ``main @ 1a2b3c4`` while every file on the
    page came from a subproject sitting on ``feature/nightly-batch`` at a different commit. The
    line was not wrong so much as answering a different question than the one it appeared to.

    So the response says both: the tree it was answered from, and — when the files are in one
    — the nested checkout they belong to, with its own branch and commit.

    Walks each path upward to its nearest enclosing ``.git`` (memoized per directory, capped
    by :data:`_MAX_SUBPROJECT_DIRS`), and reports nothing at all in the ordinary case where
    that is the project root itself. Every failure is silence: a provenance line that might
    be wrong is worse than none (the same rule :func:`describe_origin` follows).
    """
    nested: dict[str, str | None] = {}
    seen_dirs: set[str] = set()
    for rel_path in rel_paths:
        directory = str(rel_path).replace("\\", "/").rsplit("/", 1)[0] if "/" in str(rel_path) else ""
        if not directory or directory in seen_dirs:
            continue
        if len(seen_dirs) >= _MAX_SUBPROJECT_DIRS:
            break
        seen_dirs.add(directory)
        segments = directory.split("/")
        # Deepest first: a submodule inside a submodule belongs to the inner one.
        for depth in range(len(segments), 0, -1):
            candidate = "/".join(segments[:depth])
            if candidate in nested:
                if nested[candidate] is not None:
                    break
                continue
            try:
                is_checkout = (root / candidate / ".git").exists()
            except OSError:
                is_checkout = False
            if not is_checkout:
                nested[candidate] = None
                continue
            head = head_description(root / candidate)
            nested[candidate] = head or "an independent checkout"
            break
    return [
        f"{directory} ({head})"
        for directory, head in sorted(nested.items())
        if head is not None
    ]


def _nested_checkouts_notice(lines: list[str]) -> str:
    """Phrase the nested-checkout list as the correction to ``answered_from`` that it is."""
    joined = "; ".join(lines)
    return (
        f"files on this page live in a checkout of their own, whose HEAD — not the one in "
        f"answered_from — is the code you are reading: {joined}"
    )


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


# Drift is measured once per (tree, HEAD, index state): re-hashing every indexed file is
# seconds on a large repository, and nothing about the answer changes until one of the three
# does. Keyed by the stored checksums' identity too, so a re-index that did not move HEAD
# (a watcher catching up on uncommitted work) is seen on the next call.
_drift_cache: dict[tuple[str, str, str, int], tuple[int, list[str], bool]] = {}
_DRIFT_CACHE_MAX = 64
# The re-hash runs on a read tool's path, so it is bounded; a partial count says so.
DRIFT_TIME_BUDGET_S = 4.0
_DRIFT_EXAMPLES = 5


def measure_drift(
    root: Path, stored: Mapping[str, str], indexed_head: str | None
) -> tuple[int, list[str], bool] | None:
    """How far the index is behind a checkout whose HEAD moved since it was last brought up to date.

    Returns ``(differing_files, examples, partial)``, or ``None`` when there is nothing to say:
    no recorded commit (an index from before this was stamped), no readable HEAD, the same
    commit — or a moved HEAD over which every indexed file still hashes equal (a commit of
    work the watcher had already indexed).

    Why it exists: the per-page freshness flags only ever look at the handful of files an
    answer returns, and a merge rewrites files the answer never touched. Field reports
    (15-16.09.2026): a worktree absorbed four merges touching ~40 files, and every answer
    afterwards said ``stale: []`` about the four files it happened to return while the index
    described the pre-merge tree; another answered a review question from files the branch
    had rewritten, with ``answered_from`` naming the branch. HEAD moving is the cheap trigger;
    the re-hash is what makes the claim exact rather than a guess about which files a merge
    touched. Blocking I/O — call it off the event loop.
    """
    if not indexed_head:
        return None
    current = head_sha(root)
    if not current or current == indexed_head:
        return None
    key = (str(root), current, indexed_head, len(stored))
    cached = _drift_cache.get(key)
    if cached is not None:
        return cached if cached[0] else None
    deadline = monotonic() + DRIFT_TIME_BUDGET_S
    differing: list[str] = []
    partial = False
    for rel_path, md5 in sorted(stored.items()):
        if monotonic() > deadline:
            partial = True
            break
        checksum = FileChecksum.try_compute(root / rel_path, root)
        if checksum is None or checksum.md5 != md5:
            differing.append(rel_path)
    result = (len(differing), differing[:_DRIFT_EXAMPLES], partial)
    if len(_drift_cache) >= _DRIFT_CACHE_MAX:
        _drift_cache.clear()
    _drift_cache[key] = result
    return result if differing else None


def _drift_notice(root: Path, indexed_head: str, drift: tuple[int, list[str], bool]) -> str:
    """Phrase a drift measurement as the warning it is, with the command that clears it."""
    count, examples, partial = drift
    now = head_description(root) or "a different commit"
    amount = f"at least {count}" if partial else str(count)
    noun = "file differs" if count == 1 else "files differ"
    shown = ", ".join(examples)
    more = "…" if count > len(examples) else ""
    return (
        f"the checkout moved on since this index was last brought up to date at "
        f"{indexed_head[:7]}; HEAD is now {now}, and {amount} indexed {noun} from what the "
        f"index holds ({shown}{more}). Entities, line numbers and summaries for those files "
        f"describe the older version, whatever this page's own flags say — read them from disk, "
        f"or run index_directory({str(root)!r}) to catch up (preview=true prices it first)"
    )


class FreshnessChecker:
    """Re-hashes the files behind one page of entity refs and annotates them.

    Built once per tool call — ``stored`` costs a single ``get_checksums`` round trip for
    the whole project — then asked about whatever paths that call happens to return.
    """

    def __init__(
        self,
        root: Path,
        stored: Mapping[str, str],
        live_worktrees: int = 0,
        origin: str | None = None,
        watch_state: Mapping[str, Any] | None = None,
        drift: str | None = None,
    ) -> None:
        self._root = root
        self._stored = dict(stored)
        self._live_worktrees = live_worktrees
        self._origin = origin
        self._watch_state = watch_state
        self._drift = drift
        # path -> status; doubles as the "already read this file" memo and as the record
        # of which paths this page actually touched (`checked` counts its keys).
        self._statuses: dict[str, str] = {}

    @property
    def indexed_files(self) -> set[str]:
        """Every path this project stored a checksum for — the index's file inventory.

        Already loaded for the freshness check, so the coverage check on an empty page
        (:mod:`inflorescence.tools.coverage`) costs no extra round trip.
        """
        return set(self._stored)

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
        if ref.get("node_type") == _DIRECTORY_NODE_TYPE:
            # A directory node's file_path names a folder, and a folder has no content to
            # hash. Judging it produced a page-wide `unknown: ["src/components"]` that read
            # as "this index is unsure about your code" when it only meant "that is not a
            # file" (field report, 27.08.2026).
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
        if self._watch_state is not None:
            block["watcher"] = _watcher_notice(self._watch_state)
        if self._drift:
            block["drift"] = self._drift
        if self._live_worktrees:
            block["live_worktrees"] = _live_worktrees_notice(self._live_worktrees)
        return block

    def stamp(self, payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        """Annotate ``payload["items"]`` (if any) and write the meta honesty block.

        Writes ``meta.answered_from`` (which tree, which commit) and
        ``meta.index_freshness`` (whether that tree still matches the index).

        Always call this *last*: the meta block reports only the paths already checked,
        so anything annotated outside ``items`` (a structure root, a context's parent)
        must be annotated first.
        """
        self.annotate_items(payload.get("items") or [])
        meta = payload.setdefault("meta", {})
        if self._origin:
            meta["answered_from"] = self._origin
            nested = describe_nested_checkouts(self._root, self._statuses)
            if nested:
                meta["nested_checkouts"] = _nested_checkouts_notice(nested)
        meta["index_freshness"] = self.meta()
        return payload


def unchecked_empty_page(
    directory: str | None = None, watch_state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The ``index_freshness`` block for a response that returns no entities at all.

    A degenerate page still carries the block — the "unflagged means fresh" convention
    only reads if it is unconditional — and says plainly that it checked nothing. Built
    from an empty checker so its shape cannot drift from the real one. Pass *directory*
    to keep the ``live_worktrees`` warning on this page too, and *watch_state* to keep the
    ``watcher`` line: an empty answer is exactly where "the branch state is elsewhere" and
    "nothing has been keeping this current" are most worth saying, because an empty page
    reads as "there is nothing here" and both are reasons it might not be.
    """
    if directory is None:
        return FreshnessChecker(Path(), {}, watch_state=watch_state).meta()
    root = Path(directory).resolve()
    return FreshnessChecker(
        root, {}, live_worktrees=count_live_worktrees(root), watch_state=watch_state
    ).meta()


async def build_checker(repo: Any, project: str, directory: str) -> FreshnessChecker:
    """Load a project's stored checksums once and wrap them in a checker.

    One ``get_checksums`` per tool call, by construction: tools build the checker once
    and reuse it for every ref they return.
    """
    stored = await repo.get_checksums(project)
    root = Path(directory).resolve()
    drift_notice: str | None = None
    indexed_head = await repo.get_indexed_head(project)
    if indexed_head:
        try:
            drift = await asyncio.to_thread(measure_drift, root, stored, indexed_head)
        except Exception:  # noqa: BLE001 — a provenance line must never break the answer
            logger.debug("Drift measurement failed for %s", root, exc_info=True)
            drift = None
        if drift is not None:
            drift_notice = _drift_notice(root, indexed_head, drift)
    return FreshnessChecker(
        root,
        stored,
        live_worktrees=count_live_worktrees(root),
        origin=describe_origin(root),
        watch_state=await repo.get_watch_state(project),
        drift=drift_notice,
    )
