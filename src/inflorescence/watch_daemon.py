"""A watcher that outlives the session that indexed the project.

``index_directory`` starts a :class:`~inflorescence.watcher.FileWatcher` inside whichever
process served the call — for the MCP server, a stdio child of one Claude session. That
watcher dies with the session, so in practice an index is current only until the window
that built it is closed, and the next session opens onto a graph describing yesterday's
code while every tool answers in the confident present tense. The field report that
prompted this module was blunter: five orphaned server processes, and not one
``Started file watcher`` line in a week of logs.

This module is the process that is *supposed* to be running: it holds no MCP connection, is
started by launchd (or by hand) rather than by a tool call, and follows a registry that
lives in the database instead of in one process's memory.

Three properties are worth stating, because each is a decision rather than an accident:

**A project is followed only when it was registered** (``inflorescence watch add``). Not
every indexed tree — a machine accumulates worktrees for branches abandoned weeks ago, and
following those means paying to summarize commits nobody will read. The registry is a flag
on the project root node, so it is visible to `watch status`, to the daemon, and to any MCP
server, without a shared file or a second source of truth.

**One daemon per machine, and one per project underneath that.** Starting a second daemon
refuses at the door (:func:`claim_singleton`), because the common accident is not exotic —
``watch install`` puts one under launchd and then ``watch run`` gets typed in a terminal — and
a daemon that boots only to follow nothing still pays for a pre-flight, holds a connection and
sweeps worktrees on its own schedule. Underneath that, each adopted project is claimed
separately (:func:`~inflorescence.db.lease.watch_claim_key`) and held for as long as it is
watched: the singleton is keyed by hostname, so a shared Memgraph still lets a second machine
run its own daemon, and the per-project claims are what keep those two from colliding. The claim is deliberately *not* the project's
index lock — holding that for hours would block every index and every flush — but it is the
same compare-and-set node with the same TTL, so a daemon killed with SIGKILL releases its
projects a couple of minutes later instead of wedging them forever.

**Adoption reconciles before it watches.** A watcher only ever hears about edits made while
it is running; everything that changed while nothing was watching is invisible to it. So
adopting a project runs one ordinary incremental re-index first — the same operation, with
the same ``MAX_INDEX_COST_USD`` gate, that a user would run by hand. Without it the daemon
would faithfully track a graph that was already wrong, which is the exact failure it exists
to end.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inflorescence.config import Settings
from inflorescence.cost import IndexCostExceededError
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.lease import (
    ProjectLease,
    ProjectLockTimeoutError,
    daemon_claim_key,
    read_lease,
    watch_claim_key,
)
from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import IndexPreflightError, ProjectManager
from inflorescence.watcher import FileWatcher

logger = logging.getLogger(__name__)


async def claim_singleton(
    conn: MemgraphConnection | None,
) -> tuple[ProjectLease | None, str | None]:
    """Take this machine's daemon claim, or explain who already holds it.

    Returns ``(claim, None)`` when this process may run, and ``(None, message)`` when another
    daemon on this machine owns the job. The message names the holder — a refusal that cannot
    say *what* is already running is just another opaque failure, and the most likely cause is
    a launchd job the person forgot they installed.

    Without a connection (in-process tests) there is nothing to claim and nothing to refuse.
    """
    if conn is None:
        return None, None
    key = daemon_claim_key()
    claim = ProjectLease(conn, key, timeout_s=0.0)
    try:
        await claim.__aenter__()
    except ProjectLockTimeoutError:
        held = await read_lease(conn, key)
        holder = (held or {}).get("holder") or "unknown process"
        return None, (
            f"A watch daemon is already running on this machine ({holder}); this one would "
            "follow nothing. Stop that one first — `inflorescence watch uninstall` for the "
            "launchd job, or kill the process id in that holder string."
        )
    return claim, None


@dataclass
class Adopted:
    """One project this daemon currently follows."""

    project: str
    root: Path
    # None only where there is no connection to claim against (in-process tests); every real
    # daemon holds a claim for as long as it follows the project.
    claim: ProjectLease | None
    catch_up: str = "not attempted"


@dataclass
class RefreshReport:
    """What one registry pass did — returned so tests and `watch run` can assert on it."""

    adopted: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    watching: list[str] = field(default_factory=list)


class WatchDaemon:
    """Follows every registered project, reconciling each one before it starts watching."""

    def __init__(
        self,
        *,
        manager: ProjectManager,
        repo: GraphRepository,
        conn: MemgraphConnection | None,
        settings: Settings,
        watcher: FileWatcher,
        catch_up: bool | None = None,
    ) -> None:
        self._manager = manager
        self._repo = repo
        self._conn = conn
        self._settings = settings
        self._watcher = watcher
        self._catch_up = settings.watch_catch_up if catch_up is None else catch_up
        self._held: dict[str, Adopted] = {}
        # Why each project is not being followed, remembered so the reason is logged when it
        # first appears (and when it changes) rather than every refresh pass. A daemon whose
        # only output is a log file must explain an empty screen — "started, watching nothing"
        # with no reason is the same invisible failure this module exists to end — without
        # reprinting that explanation twice a minute forever.
        self._skips: dict[str, str] = {}
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @property
    def watching(self) -> dict[str, Adopted]:
        """The projects currently followed, by project id."""
        return dict(self._held)

    async def run(self) -> None:
        """Adopt the registry, then re-read it every ``watch_refresh_seconds`` until stopped."""
        interval = max(1.0, float(self._settings.watch_refresh_seconds))
        try:
            while not self._stop.is_set():
                try:
                    await self.refresh()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad pass must not end the daemon
                    logger.exception("Watch registry refresh failed — retrying next pass")
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
        finally:
            await self.shutdown()

    def request_stop(self) -> None:
        """Ask :meth:`run` to finish its current pass and shut down (signal-handler safe)."""
        self._stop.set()

    async def shutdown(self) -> None:
        """Stop every watcher and release every claim, so a restart re-adopts immediately."""
        for project in list(self._held):
            await self._drop(project, "shutdown")
        logger.info("Watch daemon stopped")

    # ------------------------------------------------------------------
    # the registry pass
    # ------------------------------------------------------------------

    async def refresh(self) -> RefreshReport:
        """Reconcile what is followed with what the registry says should be."""
        report = RefreshReport()
        desired: dict[str, Path] = {}
        for row in await self._repo.list_projects():
            if not row.get("watched"):
                continue
            project = str(row.get("project") or "")
            root_path = str(row.get("root_path") or "")
            if not project or not root_path:
                continue
            root = Path(root_path)
            if not root.is_dir():
                # Registration outlives the directory (an unmounted volume, a deleted
                # checkout). Dropping the watcher is free and reversible — the next pass
                # re-adopts it — so this never touches the index itself.
                report.skipped[project] = f"root path is not a directory: {root}"
                continue
            desired[project] = root

        for project in list(self._held):
            if project not in desired:
                await self._drop(project, "no longer registered")
                report.dropped.append(project)

        for project, root in desired.items():
            if project in self._held:
                continue
            reason = await self._adopt(project, root)
            if reason is None:
                report.adopted.append(project)
            else:
                report.skipped[project] = reason

        for project, reason in report.skipped.items():
            if self._skips.get(project) != reason:
                logger.warning("Not watching project=%s: %s", project, reason)
        self._skips = dict(report.skipped)

        report.watching = sorted(self._held)
        return report

    async def _adopt(self, project: str, root: Path) -> str | None:
        """Claim, watch and reconcile one project. Returns ``None`` on success, else why not."""
        claim = ProjectLease(self._conn, watch_claim_key(project), timeout_s=0.0) if self._conn else None
        if claim is not None:
            try:
                await claim.__aenter__()
            except ProjectLockTimeoutError:
                return "another watch daemon already follows it"

        derived = self._manager.get_project(str(root))
        if derived != project:
            # The registered path no longer derives this project id — the tree was moved or
            # renamed, and the graph under `project` describes a directory that is not this
            # one. Watching it would apply this tree's edits to that other project's graph.
            await self._release(claim)
            return f"root path {root} now derives project {derived}, not {project}"

        # Watch first, reconcile second: a catch-up can take minutes, and edits made during
        # it must not fall into a gap. A flush that overlaps the catch-up queues behind the
        # project lease and then finds nothing left to do, which costs a scan, not a summary.
        self._watcher.start(project, str(root))
        adopted = Adopted(project=project, root=root, claim=claim)
        self._held[project] = adopted
        adopted.catch_up = await self._reconcile(project, root)
        logger.info(
            "Watching project=%s at %s (catch-up: %s)", project, root, adopted.catch_up
        )
        return None

    async def _drop(self, project: str, reason: str) -> None:
        self._skips.pop(project, None)
        adopted = self._held.pop(project, None)
        if adopted is None:
            return
        self._watcher.stop(project)
        await self._release(adopted.claim)
        logger.info("Stopped watching project=%s (%s)", project, reason)

    @staticmethod
    async def _release(claim: ProjectLease | None) -> None:
        if claim is None:
            return
        with contextlib.suppress(Exception):
            await claim.__aexit__(None, None, None)

    # ------------------------------------------------------------------
    # catch-up
    # ------------------------------------------------------------------

    async def _reconcile(self, project: str, root: Path) -> str:
        """Bring a freshly adopted project up to date with disk. Never raises.

        An ordinary incremental re-index: unchanged files are reused by content hash at no
        cost, and the run is priced and gated over only what moved
        (:meth:`ProjectManager.index_directory`). A cap refusal is a *result*, not a crash —
        the daemon keeps watching, so the next edits still land, and ``watch status`` reports
        the refusal so the number is not buried in a log nobody opens.
        """
        if not self._catch_up:
            return "disabled"
        include, exclude = await self._repo.get_project_scope(project)
        try:
            result = await self._manager.index_directory(
                str(root),
                include_patterns=include or None,
                exclude_patterns=exclude or None,
            )
        except IndexCostExceededError as exc:
            logger.error("Catch-up for project=%s refused by the cost cap: %s", project, exc)
            return f"blocked by cost cap: {exc}"
        except IndexPreflightError as exc:
            logger.error("Catch-up for project=%s cannot run: %s", project, exc)
            return f"dependencies unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001 — one project's failure is not the daemon's
            logger.exception("Catch-up failed for project=%s", project)
            return f"failed: {exc}"
        return _describe_catch_up(result)


def _describe_catch_up(result: dict[str, Any]) -> str:
    """One line about what the reconcile actually did, for the log and for `watch status`."""
    if result.get("reused"):
        return "already current"
    nodes = int(result.get("nodes") or 0)
    summaries = int(result.get("summaries") or 0)
    return f"reindexed {nodes} node(s), {summaries} summary write(s)"
