"""Tests for the dashboard service helpers and app wiring."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from inflorescence.dashboard.service import (
    language_for_path,
    stitch_chunks,
    verify_root_path,
)
from inflorescence.tools.cypher import validate_readonly_query


class TestVerifyRootPath:
    def test_accepts_matching_directory(self, tmp_path: Path):
        digest = hashlib.md5(str(tmp_path.resolve()).encode()).hexdigest()[:8]
        project = f"{tmp_path.name}_{digest}"
        assert verify_root_path(project, str(tmp_path)) == str(tmp_path.resolve())

    def test_rejects_wrong_directory(self, tmp_path: Path):
        assert verify_root_path("other_00000000", str(tmp_path)) is None

    def test_rejects_missing_directory(self):
        assert verify_root_path("x_00000000", "/nonexistent/definitely/not/here") is None


class TestStitchChunks:
    def test_empty(self):
        assert stitch_chunks([]) is None
        assert stitch_chunks([{"content": "", "start_line": 1}]) is None

    def test_single_chunk(self):
        content, first = stitch_chunks([{"content": "a\nb\nc", "start_line": 5, "end_line": 7}])
        assert content == "a\nb\nc"
        assert first == 5

    def test_overlapping_chunks_deduplicated(self):
        chunks = [
            {"content": "l1\nl2\nl3\nl4", "start_line": 1, "end_line": 4},
            {"content": "l3\nl4\nl5\nl6", "start_line": 3, "end_line": 6},
        ]
        content, first = stitch_chunks(chunks)
        assert content == "l1\nl2\nl3\nl4\nl5\nl6"
        assert first == 1

    def test_fully_contained_chunk_skipped(self):
        chunks = [
            {"content": "l1\nl2\nl3\nl4", "start_line": 1, "end_line": 4},
            {"content": "l2\nl3", "start_line": 2, "end_line": 3},
        ]
        content, _ = stitch_chunks(chunks)
        assert content == "l1\nl2\nl3\nl4"

    def test_gap_between_chunks_marked(self):
        chunks = [
            {"content": "l1\nl2", "start_line": 1, "end_line": 2},
            {"content": "l10\nl11", "start_line": 10, "end_line": 11},
        ]
        content, _ = stitch_chunks(chunks)
        assert content == "l1\nl2\n\nl10\nl11"


class TestLanguageForPath:
    def test_known_extensions(self):
        assert language_for_path("src/foo.py") == "python"
        assert language_for_path("a/b.tsx") == "typescript"
        assert language_for_path("main.go") == "go"

    def test_unknown_extension(self):
        assert language_for_path("notes.txt") is None
        assert language_for_path(None) is None


class TestValidateReadonlyQuery:
    def test_valid_query_passes(self):
        assert validate_readonly_query("MATCH (n:Code {project: $project}) RETURN n") is None

    def test_write_blocked(self):
        assert validate_readonly_query("MATCH (n:Code {project: $project}) SET n.x = 1") is not None
        assert validate_readonly_query("CREATE (n:Code {project: $project})") is not None

    def test_limit_blocked(self):
        error = validate_readonly_query("MATCH (n:Code {project: $project}) RETURN n LIMIT 5")
        assert error is not None and "limit" in error.lower()

    def test_unscoped_match_blocked(self):
        assert validate_readonly_query("MATCH (n:Code) RETURN n") is not None


class TestAppRoutes:
    def test_create_app_builds(self):
        from inflorescence.config import Settings
        from inflorescence.dashboard.app import create_app

        app = create_app(Settings(_env_file=None))
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/api/projects" in paths
        assert "/api/projects/{project}/graph" in paths
        assert "/api/projects/{project}/query" in paths
        assert "/api/projects/{project}/status" in paths


class _StatusConn:
    """Stub connection dispatching project_status's three queries on canned rows."""

    def __init__(self, lock_rows: list[dict]) -> None:
        self._lock_rows = lock_rows

    async def execute_query(self, query: str, params=None, max_records=None):
        if "IndexLock" in query:
            return self._lock_rows
        if "AS edges" in query:
            return [{"edges": 7}]
        return [{"nodes": 5, "summarized": 3, "summarizable": 4}]


def _status_service(lock_rows: list[dict]):
    from inflorescence.config import Settings
    from inflorescence.dashboard.service import DashboardService

    return DashboardService(conn=_StatusConn(lock_rows), settings=Settings(_env_file=None))  # type: ignore[arg-type]


class TestProjectStatus:
    """indexing must reflect a *held, unexpired* lease — a released lease keeps its
    :IndexLock node with holder = NULL (db/lease.py), and a crashed holder leaves a
    past expires_at; neither means an index is running."""

    async def test_no_lock_node_is_not_indexing(self):
        status = await _status_service([]).project_status("p")
        assert status == {"project": "p", "indexing": False, "nodes": 5, "edges": 7,
                          "summarized": 3, "summarizable": 4}

    async def test_held_unexpired_lease_is_indexing(self):
        future = int(time.time() * 1000) + 60_000
        status = await _status_service([{"holder": "tok", "expires_at": future}]).project_status("p")
        assert status["indexing"] is True

    async def test_expired_lease_is_not_indexing(self):
        past = int(time.time() * 1000) - 60_000
        status = await _status_service([{"holder": "tok", "expires_at": past}]).project_status("p")
        assert status["indexing"] is False

    async def test_released_lease_is_not_indexing(self):
        status = await _status_service([{"holder": None, "expires_at": 0}]).project_status("p")
        assert status["indexing"] is False


class _RaisingSemanticRepo:
    async def search_summary_vectors(self, project, embedding, top_k):
        raise RuntimeError("vector_search.search: no such index (needs MAGE)")

    async def search_code_vectors(self, project, embedding, top_k):
        raise RuntimeError("vector_search.search: no such index (needs MAGE)")


class _PartialSemanticRepo:
    async def search_summary_vectors(self, project, embedding, top_k):
        return []

    async def search_code_vectors(self, project, embedding, top_k):
        raise RuntimeError("transient backend failure")


class _EmptySemanticRepo:
    async def search_summary_vectors(self, project, embedding, top_k):
        return []

    async def search_code_vectors(self, project, embedding, top_k):
        return []


def _semantic_service(monkeypatch, repo):
    """Build a DashboardService whose embedder is stubbed and repo is swapped for a fake."""
    from inflorescence.config import Settings
    from inflorescence.dashboard.service import DashboardService
    from inflorescence.rag import embeddings

    monkeypatch.setattr(embeddings.EmbeddingClient, "embed", lambda self, texts: [[0.1, 0.2]])
    settings = Settings(_env_file=None)
    settings.llm_api_key = "test-key"
    service = DashboardService(conn=object(), settings=settings)  # type: ignore[arg-type]
    service._repo = repo  # type: ignore[assignment]
    return service


class TestSemanticSearchHonesty:
    async def test_reports_note_when_all_backends_fail(self, monkeypatch):
        # Finding 13 (INV-7): both vector backends failing must not read as "no matches".
        service = _semantic_service(monkeypatch, _RaisingSemanticRepo())
        result = await service._semantic_search("proj", "query")
        assert result["hits"] == []
        assert "note" in result and "unavailable" in result["note"].lower()

    async def test_reports_partial_note_when_one_backend_fails(self, monkeypatch):
        service = _semantic_service(monkeypatch, _PartialSemanticRepo())
        result = await service._semantic_search("proj", "query")
        assert "note" in result and "partially" in result["note"].lower()

    async def test_genuine_empty_carries_no_note(self, monkeypatch):
        # Both backends succeed with zero hits -> a real empty, no failure note.
        service = _semantic_service(monkeypatch, _EmptySemanticRepo())
        result = await service._semantic_search("proj", "query")
        assert result["hits"] == []
        assert "note" not in result
