from __future__ import annotations

import neo4j
import pytest

from inflorescence.config import Settings
from inflorescence.db import connection as connection_module
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.schema import PROPERTY_INDEXES, setup_schema


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


class _IndexRow:
    """Mimics a neo4j.Record row from ``SHOW INDEX INFO`` (schema.py reads it via .data())."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def data(self) -> dict[str, object]:
        return self._data


class RecordingSchemaConnection:
    """Records every execute_write call; SHOW INDEX INFO answers with pre-seeded rows.

    Same pattern as ``RecordingSchemaConnection`` in test_rag_and_tools.py, kept local here
    so this file has no dependency on a test module a parallel change is actively editing.
    """

    def __init__(self, existing_index_rows: list[_IndexRow] | None = None) -> None:
        self._existing_index_rows = existing_index_rows or []
        self.writes: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, query: str, params: dict[str, object] | None = None) -> list[object]:
        if query == "SHOW INDEX INFO;":
            return self._existing_index_rows
        # The only other read setup_schema issues is the node_type backfill probe; empty
        # means "nothing to backfill".
        assert "n.node_type IS NULL" in query
        return []

    async def execute_write(self, query: str, params: dict[str, object] | None = None) -> None:
        self.writes.append((query, params or {}))


@pytest.mark.asyncio
async def test_setup_schema_creates_code_chunk_id_and_project_property_indexes() -> None:
    """Without these, MERGE (c:CodeChunk {id, project}) in STORE_CODE_CHUNKS_BATCH scans every
    stored chunk on every write. A live re-store benchmark of ~8.1k chunks dropped from 109.3s
    to 47.5s once both indexes existed."""
    conn = RecordingSchemaConnection()

    await setup_schema(conn)

    property_index_writes = [query for query, _params in conn.writes if "CREATE INDEX ON" in query]
    assert "CREATE INDEX ON :CodeChunk(id);" in property_index_writes
    assert "CREATE INDEX ON :CodeChunk(project);" in property_index_writes


def test_property_indexes_include_code_chunk_id_and_project() -> None:
    assert ("CodeChunk", "id") in PROPERTY_INDEXES
    assert ("CodeChunk", "project") in PROPERTY_INDEXES


@pytest.mark.asyncio
async def test_setup_schema_skips_code_chunk_indexes_that_already_exist() -> None:
    """Re-running setup_schema against a DB that already has the CodeChunk indexes must not
    re-issue CREATE INDEX for them, and must not raise — the app-level existence check (not
    reliance on Memgraph's own CREATE INDEX being a no-op) is what makes repeat calls safe."""
    conn = RecordingSchemaConnection(
        existing_index_rows=[
            _IndexRow({"index type": "label+property", "label": "CodeChunk", "property": ["id"], "name": ""}),
            _IndexRow({"index type": "label+property", "label": "CodeChunk", "property": ["project"], "name": ""}),
        ]
    )

    await setup_schema(conn)

    code_chunk_property_writes = [
        query
        for query, _params in conn.writes
        if query in ("CREATE INDEX ON :CodeChunk(id);", "CREATE INDEX ON :CodeChunk(project);")
    ]
    assert code_chunk_property_writes == []
