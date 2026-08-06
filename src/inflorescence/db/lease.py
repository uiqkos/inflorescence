"""Cross-process advisory lock on a single project, backed by a lease node in Memgraph.

INV-4 ("one mutation per project at a time") must hold **by construction**, and
across processes: a CLI ``inflorescence index`` overlapping a live server's watcher flush
addresses the *same* project in the *same* Memgraph, yet an in-process ``asyncio.Lock``
cannot see the other process. The one substrate both processes share is Memgraph itself, so
the lock lives there — a single ``:IndexLock {project}`` node acquired by compare-and-set.

Atomicity under concurrency rests on two Memgraph mechanisms:

* a **uniqueness constraint** on ``IndexLock.project`` (created in ``setup_schema``) makes
  concurrent ``MERGE``-creates conflict instead of producing duplicate lock nodes;
* Memgraph aborts one side of any write conflict with a ``TransientError``, which
  :meth:`MemgraphConnection.execute_write_query` retries.

So exactly one caller ever holds the lease; the rest wait.

A holder killed mid-run would otherwise wedge the project forever (violating INV-8,
self-healing). The lease therefore carries an ``expires_at`` the live holder refreshes with a
background heartbeat; if the heartbeat stops, the lease expires and the next waiter takes it
over. Timeouts are module constants — local and reversible.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
import uuid

import neo4j.exceptions

from inflorescence.db.connection import MemgraphConnection

logger = logging.getLogger(__name__)

# How long a lease stays valid without a heartbeat. A holder killed mid-run releases the
# project this many seconds later (self-healing); a live holder never lets it lapse.
LEASE_TTL_S = 120.0
# Heartbeat cadence — comfortably below the TTL so a single missed beat never expires a
# live lease.
HEARTBEAT_INTERVAL_S = 30.0
# How often a waiter re-checks a held lease.
ACQUIRE_POLL_S = 0.5
# Safety cap on how long a waiter blocks. A cross-process re-index legitimately waits for
# the holder to finish (that is the serialization); beyond this we surface a clear error
# rather than hang forever.
ACQUIRE_TIMEOUT_S = 1800.0

# Compare-and-set acquire. `free` is true when the lease is unheld, expired, or already
# ours (re-entrant refresh). SET only advances holder/expiry when free, so a live holder's
# lease is never stolen. Returns whether we now hold it.
_ACQUIRE = """
MERGE (l:IndexLock {project: $project})
ON CREATE SET l.holder = $token, l.expires_at = $expires, l.acquired_at = $now
ON MATCH SET
  l.holder = CASE WHEN l.holder IS NULL OR l.expires_at < $now OR l.holder = $token
                  THEN $token ELSE l.holder END,
  l.expires_at = CASE WHEN l.holder IS NULL OR l.expires_at < $now OR l.holder = $token
                      THEN $expires ELSE l.expires_at END
RETURN l.holder = $token AS acquired, l.holder AS holder
"""

# Refresh our own lease only — the guard prevents renewing a lease already taken over.
_RENEW = """
MATCH (l:IndexLock {project: $project})
WHERE l.holder = $token
SET l.expires_at = $expires
RETURN count(l) AS renewed
"""

# Release only if we still hold it; a lease taken over after expiry is left untouched.
_RELEASE = """
MATCH (l:IndexLock {project: $project})
WHERE l.holder = $token
SET l.holder = NULL, l.expires_at = 0
RETURN count(l) AS released
"""


class ProjectLockTimeoutError(RuntimeError):
    """Raised when a project lease cannot be acquired within the timeout."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_token() -> str:
    """A globally-unique holder token. Host/pid aid debugging; the uuid guarantees uniqueness."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


class ProjectLease:
    """Async context manager: hold the cross-process lock on ``project`` for its lifetime."""

    def __init__(
        self,
        conn: MemgraphConnection,
        project: str,
        *,
        ttl_s: float = LEASE_TTL_S,
        heartbeat_s: float = HEARTBEAT_INTERVAL_S,
        poll_s: float = ACQUIRE_POLL_S,
        timeout_s: float = ACQUIRE_TIMEOUT_S,
    ) -> None:
        self._conn = conn
        self._project = project
        self._ttl_ms = int(ttl_s * 1000)
        self._heartbeat_s = heartbeat_s
        self._poll_s = poll_s
        self._timeout_s = timeout_s
        self._token = _new_token()
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> ProjectLease:
        deadline = time.monotonic() + self._timeout_s
        waited = False
        while not await self._try_acquire():
            waited = True
            if time.monotonic() >= deadline:
                raise ProjectLockTimeoutError(
                    f"Could not acquire index lock for project={self._project} "
                    f"within {self._timeout_s:.0f}s — another indexer is holding it."
                )
            await asyncio.sleep(self._poll_s)
        if waited:
            logger.info("Acquired index lock for project=%s after waiting", self._project)
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
        await self._release()

    async def _try_acquire(self) -> bool:
        now = _now_ms()
        try:
            rows = await self._conn.execute_write_query(
                _ACQUIRE,
                {"project": self._project, "token": self._token, "now": now, "expires": now + self._ttl_ms},
            )
        except neo4j.exceptions.ClientError as exc:
            # A rare first-ever create race can trip the uniqueness constraint; the winner's
            # node now exists, so treat this as "held by someone else" and poll again.
            logger.debug("Lock acquire contended for project=%s: %s", self._project, exc)
            return False
        return bool(rows and rows[0]["acquired"])

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            try:
                await self._conn.execute_write_query(
                    _RENEW,
                    {"project": self._project, "token": self._token, "expires": _now_ms() + self._ttl_ms},
                )
            except Exception:  # noqa: BLE001 — heartbeat is best-effort; a lapse only risks takeover
                logger.warning("Failed to renew index lock for project=%s", self._project, exc_info=True)

    async def _release(self) -> None:
        try:
            await self._conn.execute_write_query(
                _RELEASE, {"project": self._project, "token": self._token}
            )
        except Exception:  # noqa: BLE001 — release failure only delays takeover until TTL expiry
            logger.warning("Failed to release index lock for project=%s", self._project, exc_info=True)


def project_lease(conn: MemgraphConnection | None, project: str):  # noqa: ANN201 — async CM union
    """Return the cross-process lease for *project*, or a no-op when *conn* is absent.

    Production always passes a connection (server and CLI both build one before any mutation).
    ``conn=None`` — pure in-process unit tests with a fake repository — degrades to a no-op
    context manager; in that case the caller's per-project ``asyncio.Lock`` still serializes
    within the process, and the cross-process guarantee is exercised by the live integration test.
    """
    if conn is None:
        return contextlib.nullcontext()
    return ProjectLease(conn, project)
