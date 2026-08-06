from __future__ import annotations

import neo4j
import pytest

from inflorescence.config import Settings
from inflorescence.db import connection as connection_module
from inflorescence.db.connection import MemgraphConnection


def _transient() -> neo4j.exceptions.TransientError:
    return neo4j.exceptions.TransientError("Cannot resolve conflicting transactions")


class _FakeResult:
    def __init__(self, summary: object) -> None:
        self._summary = summary

    async def consume(self) -> object:
        return self._summary


class _FakeSession:
    def __init__(self, driver: _FakeDriver) -> None:
        self._driver = driver

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def run(self, query: str, params: dict) -> _FakeResult:
        self._driver.calls += 1
        if self._driver.fail_times > 0:
            self._driver.fail_times -= 1
            raise _transient()
        return _FakeResult(self._driver.summary)


class _FakeDriver:
    def __init__(self, fail_times: int, summary: str = "ok") -> None:
        self.fail_times = fail_times
        self.summary = summary
        self.calls = 0

    def session(self) -> _FakeSession:
        return _FakeSession(self)

    async def close(self) -> None:
        return None


def _conn_with_driver(driver: _FakeDriver, monkeypatch: pytest.MonkeyPatch) -> MemgraphConnection:
    # No backoff sleep in tests.
    monkeypatch.setattr(connection_module, "_WRITE_RETRY_BASE_DELAY_S", 0.0)
    conn = MemgraphConnection(Settings(_env_file=None))
    conn._driver = driver  # type: ignore[assignment]
    return conn


@pytest.mark.asyncio
async def test_execute_write_retries_transient_conflict_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver(fail_times=2)
    conn = _conn_with_driver(driver, monkeypatch)

    summary = await conn.execute_write("MERGE (n)", {"x": 1})

    assert summary == "ok"
    assert driver.calls == 3  # 2 conflicts + 1 success


@pytest.mark.asyncio
async def test_execute_write_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver(fail_times=99)
    conn = _conn_with_driver(driver, monkeypatch)

    with pytest.raises(neo4j.exceptions.TransientError):
        await conn.execute_write("MERGE (n)")

    assert driver.calls == connection_module._WRITE_MAX_RETRIES
