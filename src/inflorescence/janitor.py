"""Reap the graphs of git worktrees that no longer exist.

A linked worktree is a disposable checkout: created for a branch, deleted a day later. It is
indexed as its own project (its tree differs from the main checkout's), so without a janitor
every dead worktree leaves a graph behind that answers questions about files nobody can open
any more. Deleting one is cheap to undo — re-creating the worktree re-indexes almost free by
donating the main checkout's cached summaries and chunk vectors — which is the whole reason
this module is allowed to delete anything at all.

Deleting the WRONG project is not cheap to undo: a project's index is paid for in LLM
summaries and embeddings, dollars that a rebuild spends again. So the janitor is written as a
sequence of independent gates, each of which alone is sufficient to stop a deletion. Every
layer is tested on its own in ``tests/test_janitor.py`` — a regression in one must fail a test
rather than be masked by the other four.

The five layers, in order:

1. **Marker.** Only a project carrying the worktree marker (``worktree_of`` +
   ``main_root_path``, written at index time by ``set_project_worktree_meta``) is ever a
   candidate. An ordinary project is never touched, whatever happened to its ``root_path``.
   This is what makes the classic disaster harmless: an unmounted volume makes every
   project's root path vanish at once, and the janitor still deletes nothing, sweep after
   sweep. A half-written marker counts as no marker.
2. **The worktree's own directory is gone.** ``root_path`` must not resolve to a directory.
3. **The donor is still there and still looks like a repository.** ``main_root_path`` must
   exist and contain ``.git``. If the donor vanished too, we cannot distinguish "the worktree
   was deleted" from "the disk holding both is not mounted" — so we skip and log a warning.
   Conservatism beats tidiness: a stale graph costs disk, a wrong delete costs money.
4. **Git's registry has the last word.** If the donor still registers this path *and* the
   registered directory exists, the worktree is alive and the layer-2 check was lying (a
   removed intermediate component, a permission blip, a mount in flux) — skip. Registration
   alone does not save it: ``rm -rf`` without ``git worktree prune`` leaves the entry in place
   while the content is dead, and that is precisely the case worth reaping.
5. **The guarded deleter.** The janitor calls only
   :meth:`GraphRepository.delete_worktree_project`, never the unguarded ``delete_project``.
   That wrapper re-reads the marker from the database and re-asserts it inside the delete
   statement itself, so a bug in everything above still ends in a refusal.

Two further rules, orthogonal to the layers:

* the sweep takes the project's advisory lease before deleting, so a graph is never pulled
  out from under a running index; a held lease is a skip, never a wait;
* an exception while considering one project is contained to that project — the sweep goes
  on, because a single unreadable row must not stop the others from being reaped.

Every deletion is logged at WARNING with the project, its path and its donor: this is an
expensive, irreversible operation and it has to be visible in the log afterwards. Every skip
is logged at INFO with a reason.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from inflorescence.config import Settings
from inflorescence.db.lease import ProjectLease, ProjectLockTimeoutError
from inflorescence.worktree import registered_worktree_paths

logger = logging.getLogger(__name__)

# Skip reasons. Constants rather than literals so a test asserts on the same symbol the
# implementation reports, and a renamed reason cannot silently pass a stale assertion.
REASON_ON_DISK = "worktree_still_on_disk"
REASON_DONOR_MISSING = "donor_missing"
REASON_STILL_REGISTERED = "still_registered"
REASON_LEASE_HELD = "lease_held"
REASON_REFUSED = "delete_refused"
REASON_ERROR = "error"


class JanitorRepository(Protocol):
    """The repository surface the janitor uses — deliberately three methods, one of them read-only."""

    async def list_projects(self) -> list[dict[str, Any]]: ...

    async def get_project_worktree_meta(self, project: str) -> dict[str, str] | None: ...

    async def delete_worktree_project(self, project: str) -> bool: ...


def _no_wait_lease(conn: Any, project: str) -> Any:
    """The project's advisory lease, acquired without waiting, or a no-op without a connection.

    ``timeout_s=0`` turns :class:`ProjectLease` into a single compare-and-set attempt: held by
    somebody else means :class:`ProjectLockTimeoutError` immediately. The janitor must not
    queue behind a running index — that index may run for an hour, and by the time it
    finished the answer to "is this worktree gone?" would be stale anyway. Waiting is also how
    a background chore turns into a stuck server task.

    ``conn=None`` (unit tests with a fake repository) degrades to a no-op context manager, the
    same convention :func:`inflorescence.db.lease.project_lease` uses.
    """
    if conn is None:
        return contextlib.nullcontext()
    return ProjectLease(conn, project, timeout_s=0.0)


def _empty_report() -> dict[str, Any]:
    return {"scanned": 0, "candidates": 0, "deleted": [], "skipped": []}


def _marker_of(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """Layer 1: ``(worktree_of, main_root_path)`` when the row carries a complete marker.

    A half-written marker (donor id without its path, or the reverse) is reported as absent —
    it tells us the project may be a worktree but not whose, which is not enough to verify
    anything against, and "not enough information" must always resolve to "keep".
    """
    worktree_of = str(row.get("worktree_of") or "").strip()
    main_root_path = str(row.get("main_root_path") or "").strip()
    if not worktree_of or not main_root_path:
        return None
    return worktree_of, main_root_path


def _is_present(path: Path) -> bool:
    """``True`` when *path* is a directory we can see. Any OS error counts as "present".

    An error is not evidence of absence: a permission failure or an I/O error on a flaky
    mount would otherwise read as "the worktree is gone" and hand the deleter a false
    positive. ``Path.is_dir`` already swallows most errors into ``False``, which is why layer
    4 re-checks against git's registry.
    """
    try:
        return path.is_dir()
    except OSError:
        return True


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, ValueError, RuntimeError):
        return path


async def sweep(
    repo: JanitorRepository,
    settings: Settings,
    *,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Delete the graphs of worktrees that are gone. Returns a report of what happened.

    The report is ``{"scanned": int, "candidates": int, "deleted": [project_id, ...],
    "skipped": [{"project": id, "reason": str}, ...]}``. ``candidates`` counts projects that
    passed layer 1; ``skipped`` holds only candidates that were spared, since a project
    without the marker is not a thing the janitor considers at all.

    Never raises: a database that cannot be listed, or a single project that cannot be
    reasoned about, is reported, not propagated. This runs as a background chore on a server
    whose real job is answering queries.
    """
    report = _empty_report()

    # Second belt on the off switch. The caller (server startup) already refuses to schedule
    # a disabled janitor; a disabled sweep that is called anyway must still do nothing at
    # all — not even list the projects.
    if not settings.worktree_janitor_enabled:
        logger.info("Worktree janitor disabled (worktree_janitor_enabled=False) — no sweep")
        return report

    try:
        rows = await repo.list_projects()
    except Exception:  # noqa: BLE001 — a background chore must not take the server with it
        logger.warning("Worktree janitor could not list projects — skipping this sweep", exc_info=True)
        return report

    report["scanned"] = len(rows)
    for row in rows:
        project = str(row.get("project") or "").strip()
        root_path = str(row.get("root_path") or "").strip()
        if not project or not root_path:
            # Nothing to verify against; such a row can never become a candidate.
            logger.debug("Worktree janitor: ignoring row without project id or root_path: %r", row)
            continue

        marker = _marker_of(row)  # Layer 1
        if marker is None:
            continue
        report["candidates"] += 1

        try:
            outcome = await _consider(repo, project, root_path, marker, conn=conn)
        except Exception:  # noqa: BLE001 — one bad project must not end the sweep
            logger.exception("Worktree janitor failed on project=%s — skipping it", project)
            report["skipped"].append({"project": project, "reason": REASON_ERROR})
            continue

        if outcome is None:
            report["deleted"].append(project)
        else:
            logger.info("Worktree janitor kept project=%s (%s)", project, outcome)
            report["skipped"].append({"project": project, "reason": outcome})

    logger.info(
        "Worktree janitor swept %d project(s): %d candidate(s), %d deleted, %d skipped",
        report["scanned"], report["candidates"], len(report["deleted"]), len(report["skipped"]),
    )
    return report


async def _consider(
    repo: JanitorRepository,
    project: str,
    root_path: str,
    marker: tuple[str, str],
    *,
    conn: Any | None,
) -> str | None:
    """Run layers 2-5 for one candidate. Returns a skip reason, or ``None`` when deleted."""
    worktree_of, main_root_path = marker
    root = Path(root_path)
    donor = Path(main_root_path)

    # Layer 2 — the worktree's own directory must be gone.
    if _is_present(root):
        return REASON_ON_DISK

    # Layer 3 — the donor must still be there and still be a repository. Both paths missing
    # is the signature of an unmounted volume just as much as of a deleted worktree, and the
    # two are not distinguishable from here.
    if not _is_present(donor) or not _donor_looks_like_a_repo(donor):
        logger.warning(
            "Worktree janitor: project=%s looks deleted (%s) but its donor %s is missing or is "
            "not a git checkout — keeping the index, since a missing donor is also what an "
            "unmounted volume looks like",
            project, root_path, main_root_path,
        )
        return REASON_DONOR_MISSING

    # Layer 4 — git's registry. Registered *and* present means the worktree is alive and
    # layer 2's answer was wrong; registered but absent is the `rm -rf` case we exist for.
    registered = registered_worktree_paths(donor)
    resolved_root = _resolved(root)
    if resolved_root in registered and _is_present(resolved_root):
        return REASON_STILL_REGISTERED

    # Layer 5 — take the lease so an index in flight is never deleted from under, then call
    # the guarded deleter, which verifies the marker again in the database.
    try:
        async with _no_wait_lease(conn, project):
            deleted = await repo.delete_worktree_project(project)
    except ProjectLockTimeoutError:
        return REASON_LEASE_HELD

    if not deleted:
        # The wrapper refused: the marker is gone from the database. Whatever the reason,
        # the project is no longer ours to delete.
        logger.warning(
            "Worktree janitor: guarded delete refused project=%s — the worktree marker is no "
            "longer in the database",
            project,
        )
        return REASON_REFUSED

    logger.warning(
        "Worktree janitor DELETED the index of project=%s (root_path=%s), a worktree of "
        "project=%s at %s. The directory is gone and git no longer keeps a live checkout "
        "there; re-creating the worktree re-indexes it from the donor's cache.",
        project, root_path, worktree_of, main_root_path,
    )
    return None


def _donor_looks_like_a_repo(donor: Path) -> bool:
    """``True`` when the donor directory holds a ``.git`` entry (directory or file)."""
    try:
        return (donor / ".git").exists()
    except OSError:
        # Unreadable is not "not a repo" — fail towards keeping the index.
        return True
