"""Async Memgraph connection wrapping the neo4j Bolt driver."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Self

import neo4j
from neo4j import AsyncGraphDatabase

from inflorescence.config import Settings

logger = logging.getLogger(__name__)

# Memgraph aborts one side of a write conflict with a TransientError
# ("Cannot resolve conflicting transactions"). Concurrent updates to the same
# project — e.g. two watcher flushes racing, or a manual re-index overlapping a
# live rebuild — can trigger it. The writes here are idempotent (MERGE/UNWIND
# upserts, project-scoped deletes), so retrying the aborted statement is safe.
_WRITE_MAX_RETRIES = 4
_WRITE_RETRY_BASE_DELAY_S = 0.1


class MemgraphConnection:
    """Thin async wrapper around neo4j driver for Memgraph."""

    def __init__(self, settings: Settings) -> None:
        auth: tuple[str, str] | None = None
        if settings.memgraph_user is not None and settings.memgraph_password is not None:
            auth = (settings.memgraph_user, settings.memgraph_password)

        self._driver: neo4j.AsyncDriver = AsyncGraphDatabase.driver(
            settings.memgraph_url,
            auth=auth,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    async def execute_query(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
        max_records: int | None = None,
    ) -> list[neo4j.Record]:
        """Run a read query and return its result records.

        ``max_records`` stops reading the cursor after that many rows. Callers that append
        their own ``LIMIT`` should still pass it: it bounds memory independently of the query
        text, so a row cap expressed only in Cypher cannot be the single point of failure.
        """
        logger.debug("execute_query: %s  params=%s", query, params)
        async with self._driver.session() as session:
            # neo4j typing expects LiteralString; we accept dynamic Cypher by design
            result: neo4j.AsyncResult = await session.run(query, dict(params) if params else {})  # type: ignore[arg-type]
            records: list[neo4j.Record] = []
            async for record in result:
                records.append(record)
                if max_records is not None and len(records) >= max_records:
                    break
        return records

    async def execute_write(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
    ) -> neo4j.ResultSummary:
        """Run a write query and return the result summary.

        Retries transient write conflicts (Memgraph aborts one side of concurrent
        writes to the same data) with exponential backoff before giving up.
        """
        logger.debug("execute_write: %s  params=%s", query, params)
        payload = dict(params) if params else {}
        for attempt in range(1, _WRITE_MAX_RETRIES + 1):
            try:
                async with self._driver.session() as session:
                    # neo4j typing expects LiteralString; we accept dynamic Cypher by design
                    result: neo4j.AsyncResult = await session.run(query, payload)  # type: ignore[arg-type]
                    return await result.consume()
            except neo4j.exceptions.TransientError as exc:
                if attempt >= _WRITE_MAX_RETRIES:
                    logger.warning("Write failed after %d attempts on transient conflict", attempt)
                    raise
                delay = _WRITE_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.info(
                    "Transient write conflict (attempt %d/%d), retrying in %.2fs: %s",
                    attempt, _WRITE_MAX_RETRIES, delay, exc.code,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def execute_write_query(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
    ) -> list[neo4j.Record]:
        """Run a write query that returns rows, retrying transient conflicts.

        Same retry policy as :meth:`execute_write` (Memgraph aborts one side of a write
        conflict with a ``TransientError``), but the result rows are read before the result
        is consumed — needed for compare-and-set writes like the project lease, whose return
        value ("did I acquire it?") is the whole point.
        """
        logger.debug("execute_write_query: %s  params=%s", query, params)
        payload = dict(params) if params else {}
        for attempt in range(1, _WRITE_MAX_RETRIES + 1):
            try:
                async with self._driver.session() as session:
                    # neo4j typing expects LiteralString; we accept dynamic Cypher by design
                    result: neo4j.AsyncResult = await session.run(query, payload)  # type: ignore[arg-type]
                    return [record async for record in result]
            except neo4j.exceptions.TransientError as exc:
                if attempt >= _WRITE_MAX_RETRIES:
                    logger.warning("Write query failed after %d attempts on transient conflict", attempt)
                    raise
                delay = _WRITE_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.info(
                    "Transient write conflict (attempt %d/%d), retrying in %.2fs: %s",
                    attempt, _WRITE_MAX_RETRIES, delay, exc.code,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def health_check(self) -> bool:
        """Return True if the database responds to a trivial query."""
        try:
            records = await self.execute_query("RETURN 1 AS ok")
            return len(records) == 1 and records[0]["ok"] == 1
        except neo4j.exceptions.Neo4jError:
            logger.warning("Health check failed (Neo4jError)", exc_info=True)
            return False
        except OSError:
            logger.warning("Health check failed (OSError)", exc_info=True)
            return False

    async def close(self) -> None:
        """Close the underlying driver and release resources."""
        await self._driver.close()
