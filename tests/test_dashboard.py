"""Tests for the dashboard service helpers and app wiring."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from inflorescence.dashboard.service import (
    derived_path,
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
        assert "/api/projects/{project}/specimen" in paths
        assert "/api/projects/{project}/query" in paths
        assert "/api/projects/{project}/status" in paths
        assert "/api/projects/{project}/freshness" in paths


class TestDerivedPath:
    """pack.js reconstructs a node's path from its id; both sides must agree, or the
    specimen would label entities with the wrong file."""

    def test_directory_id(self):
        assert derived_path("dir:src/inflorescence") == "src/inflorescence"

    def test_module_id_is_its_own_path(self):
        assert derived_path("src/a.py") == "src/a.py"

    def test_symbol_id_drops_the_symbol(self):
        assert derived_path("src/a.py::Klass.method") == "src/a.py"


# --- the specimen payload -------------------------------------------------------------

def _node(node_id, node_type, name, parent, path=None, summary="", sig="", ls=None, le=None):
    return {
        "id": node_id, "t": node_type, "name": name, "parent": parent,
        "path": derived_path(node_id) if path is None else path,
        "ls": ls, "le": le, "summary": summary, "sig": sig,
    }


_SPECIMEN_NODES = [
    _node("dir:.", "directory", ".", None, path="."),
    _node("dir:src", "directory", "src", "dir:."),
    _node("dir:docs", "directory", "docs", "dir:."),
    _node("src/a.py", "module", "a", "dir:src", summary="module a", ls=1, le=40),
    _node("src/a.py::foo", "function", "foo", "src/a.py", summary="x" * 400, sig="def foo() -> None", ls=3, le=9),
    _node("src/a.py::Bar", "class", "Bar", "src/a.py", summary="", ls=12, le=30),
    _node("docs/readme.md", "document", "readme.md", "dir:docs", summary="the readme", ls=1, le=5),
    # a node whose stored path is not what its id encodes: it must keep its own `p`
    _node("weird", "module", "weird", "dir:src", path="src/renamed.py", summary="odd one"),
]
_SPECIMEN_EDGES = [
    {"src": "src/a.py::foo", "dst": "src/a.py::Bar", "kind": "CALLS", "res": "semantic"},
    {"src": "src/a.py", "dst": "docs/readme.md", "kind": "IMPORTS", "res": None},
    {"src": "src/a.py::foo", "dst": "src/a.py::foo", "kind": "CALLS", "res": ""},  # self edge: dropped
]


class _SpecimenConn:
    """Stub connection dispatching graph_specimen's four queries on canned rows."""

    def __init__(self, nodes=None, edges=None):
        self.nodes = _SPECIMEN_NODES if nodes is None else nodes
        self.edges = _SPECIMEN_EDGES if edges is None else edges

    async def execute_query(self, query: str, params=None, max_records=None):
        if "n.parent_id AS parent" in query:
            return self.nodes
        if "AS kind" in query:
            return self.edges
        if "CodeChunk" in query:
            return [{"chunks": 12}]
        if "External" in query:
            return [{"root": "typing", "n": 3}, {"root": None, "n": 1}]
        raise AssertionError(f"unexpected query: {query}")


def _specimen_service(conn=None):
    from inflorescence.config import Settings
    from inflorescence.dashboard.service import DashboardService

    return DashboardService(  # type: ignore[arg-type]
        conn=conn or _SpecimenConn(), settings=Settings(_env_file=None)
    )


class TestGraphSpecimen:
    """The payload pack.js lays out. It is the whole index in one response, so its
    compactness is load-bearing — and so is its honesty about anything it left out."""

    async def test_nodes_are_index_addressed_and_abbreviated(self):
        payload = await _specimen_service().graph_specimen("p")
        by_id = {n["i"]: n for n in payload["nodes"]}
        index = {n["i"]: i for i, n in enumerate(payload["nodes"])}

        assert payload["root"] == "dir:."
        assert by_id["dir:."]["pa"] == -1
        assert by_id["src/a.py"]["pa"] == index["dir:src"]
        assert by_id["src/a.py::foo"]["pa"] == index["src/a.py"]
        # `p` is omitted whenever the id already encodes it, and kept when it does not
        assert "p" not in by_id["src/a.py::foo"]
        assert by_id["weird"]["p"] == "src/renamed.py"
        assert by_id["src/a.py::foo"]["l"] == [3, 9]
        assert by_id["src/a.py::foo"]["g"] == "def foo() -> None"
        # a node with no summary carries no `s` at all rather than an empty string
        assert "s" not in by_id["src/a.py::Bar"]

    async def test_summaries_are_clipped_for_the_list_payload(self):
        payload = await _specimen_service().graph_specimen("p")
        summary = next(n for n in payload["nodes"] if n["i"] == "src/a.py::foo")["s"]
        assert len(summary) <= 300
        assert summary.endswith("…")  # clipped, and says so

    async def test_edges_address_nodes_by_index_and_drop_self_edges(self):
        payload = await _specimen_service().graph_specimen("p")
        index = {n["i"]: i for i, n in enumerate(payload["nodes"])}
        assert [index["src/a.py::foo"], index["src/a.py::Bar"], "C", "semantic"] in payload["edges"]
        assert [index["src/a.py"], index["docs/readme.md"], "I", ""] in payload["edges"]
        assert len(payload["edges"]) == 2  # the self-CALLS is not drawable

    async def test_totals_describe_the_index_including_contains(self):
        payload = await _specimen_service().graph_specimen("p")
        assert payload["totals"]["directory"] == 3
        assert payload["totals"]["module"] == 2
        assert payload["edge_totals"]["CALLS"] == 2
        # CONTAINS never appears in `edges` — it is the `pa` column — but the legend
        # still has to be able to name how many the index holds.
        assert payload["edge_totals"]["CONTAINS"] == 7
        assert payload["chunks"] == 12
        assert payload["external"] == [["typing", 3]]

    async def test_prefix_scope_keeps_the_directories_that_position_what_is_left(self):
        payload = await _specimen_service().graph_specimen("p", prefix="src")
        ids = {n["i"] for n in payload["nodes"]}
        assert "src/a.py::foo" in ids
        assert "docs/readme.md" not in ids
        # nesting is the whole point of this view: the root and src/ must survive
        assert {"dir:.", "dir:src"} <= ids
        assert payload["scope"] == "src"
        assert any("src" in note for note in payload["notes"])

    async def test_prefix_respects_directory_boundaries(self):
        nodes = [*_SPECIMEN_NODES, _node("srcextra/b.py", "module", "b", "dir:.")]
        payload = await _specimen_service(_SpecimenConn(nodes=nodes)).graph_specimen("p", prefix="src")
        assert "srcextra/b.py" not in {n["i"] for n in payload["nodes"]}

    async def test_modules_level_keeps_only_the_map(self):
        payload = await _specimen_service().graph_specimen("p", level="modules")
        types = {n["t"] for n in payload["nodes"]}
        assert types <= {"directory", "module", "document"}
        assert payload["level"] == "modules"
        assert any("files and directories" in note for note in payload["notes"])

    async def test_summarized_only_keeps_what_has_been_summarized(self):
        payload = await _specimen_service().graph_specimen("p", summarized_only=True)
        ids = {n["i"] for n in payload["nodes"]}
        assert "src/a.py::Bar" not in ids   # no summary yet
        assert "src/a.py::foo" in ids
        assert "dir:src" in ids             # directories still position what is left
        assert any("summary" in note for note in payload["notes"])

    async def test_over_the_cap_it_says_what_it_dropped(self):
        # INV-7: a smaller graph must never be indistinguishable from a small one.
        crowd = [
            *_SPECIMEN_NODES,
            *(_node(f"src/a.py::f{i}", "function", f"f{i}", "src/a.py", summary="s") for i in range(300)),
        ]
        payload = await _specimen_service(_SpecimenConn(nodes=crowd)).graph_specimen("p", cap=100)
        assert payload["truncated"] is True
        assert any("best-connected" in note for note in payload["notes"])
        assert len(payload["nodes"]) == 100
        # the map survives the cull: directories and files rank ahead of leaves
        assert {"dir:.", "dir:src", "dir:docs", "src/a.py"} <= {n["i"] for n in payload["nodes"]}
        # and every surviving node still resolves its parent, or the tree would fall apart
        ids = [n["i"] for n in payload["nodes"]]
        for node in payload["nodes"]:
            assert node["pa"] == -1 or 0 <= node["pa"] < len(ids)

    async def test_totals_still_describe_the_whole_index_when_scoped(self):
        """A scoped view has to be able to say what share of the index it is showing."""
        payload = await _specimen_service().graph_specimen("p", prefix="src")
        assert payload["totals"]["document"] == 1  # docs/readme.md is out of scope but counted
        assert sum(payload["totals"].values()) == len(_SPECIMEN_NODES)


class _StatusConn:
    """Stub connection dispatching project_status's queries on canned rows.

    The watch-state read must be matched *before* the index-lock one: it reads the lease node
    too (that is how it reports a live daemon), so a looser dispatch would feed it the
    indexing lock's rows and quietly conflate "an index is running" with "a daemon is
    following this project".
    """

    def __init__(self, lock_rows: list[dict], watch_rows: list[dict] | None = None) -> None:
        self._lock_rows = lock_rows
        self._watch_rows = watch_rows if watch_rows is not None else [
            {"watched": False, "holder": None, "expires_at": 0,
             "daemon_holder": None, "daemon_expires_at": 0}
        ]

    async def execute_query(self, query: str, params=None, max_records=None):
        if "AS watched" in query:
            return self._watch_rows
        if "IndexLock" in query:
            return self._lock_rows
        if "AS edges" in query:
            return [{"edges": 7}]
        return [{"nodes": 5, "summarized": 3, "summarizable": 4}]


def _status_service(lock_rows: list[dict], watch_rows: list[dict] | None = None):
    from inflorescence.config import Settings
    from inflorescence.dashboard.service import DashboardService

    return DashboardService(  # type: ignore[arg-type]
        conn=_StatusConn(lock_rows, watch_rows), settings=Settings(_env_file=None)
    )


class TestProjectStatus:
    """indexing must reflect a *held, unexpired* lease — a released lease keeps its
    :IndexLock node with holder = NULL (db/lease.py), and a crashed holder leaves a
    past expires_at; neither means an index is running."""

    async def test_no_lock_node_is_not_indexing(self):
        status = await _status_service([]).project_status("p")
        assert status == {
            "project": "p", "indexing": False, "nodes": 5, "edges": 7,
            "summarized": 3, "summarizable": 4,
            "watch": {"watched": False, "live": False, "holder": None, "daemon_live": False},
        }

    async def test_status_says_whether_a_daemon_keeps_this_project_current(self):
        """`indexing` is about a run happening now; `watch` is about the rest of the time.

        The dashboard draws a graph that looks equally authoritative either way, so the
        difference between "a daemon re-indexes every save" and "this was true last Tuesday"
        has to travel with the data rather than be assumed by whoever is looking.
        """
        future = int(time.time() * 1000) + 60_000
        status = await _status_service(
            [],
            [{"watched": True, "holder": "mac:99:abc", "expires_at": future,
              "daemon_holder": "mac:99:abc", "daemon_expires_at": future}],
        ).project_status("p")

        assert status["indexing"] is False
        assert status["watch"] == {
            "watched": True, "live": True, "holder": "mac:99:abc", "daemon_live": True,
        }

    async def test_a_registration_nobody_runs_is_not_reported_as_watched(self):
        """The state after a reboot: registered, and nothing following it."""
        status = await _status_service(
            [],
            [{"watched": True, "holder": None, "expires_at": 0,
              "daemon_holder": None, "daemon_expires_at": 0}],
        ).project_status("p")

        assert status["watch"] == {
            "watched": True, "live": False, "holder": None, "daemon_live": False,
        }

    async def test_a_running_daemon_that_has_not_taken_this_project_is_not_absent(self):
        """"Not followed" and "nothing is running" are different, and only one is fixed by
        starting a daemon. A project registered seconds ago is always in the first state until
        the daemon's next registry pass; so is one the daemon deliberately refused."""
        future = int(time.time() * 1000) + 60_000
        status = await _status_service(
            [],
            [{"watched": True, "holder": None, "expires_at": 0,
              "daemon_holder": "mac:99:abc", "daemon_expires_at": future}],
        ).project_status("p")

        assert status["watch"] == {
            "watched": True, "live": False, "holder": None, "daemon_live": True,
        }

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


class TestProjectFreshness:
    """The on-demand drift check behind the dashboard's watch badge.

    Kept out of the 2-second status poll on purpose: it walks the tree and hashes every file
    it admits. That is also why it must be *right* — someone clicks it to decide whether the
    graph in front of them can be trusted.
    """

    async def test_reports_files_the_index_has_never_seen(self, tmp_path: Path):
        svc = _status_service([])
        (tmp_path / "a.py").write_text("def a(): ...\n")
        # Big enough that the estimate does not round to zero: the point of the assertion
        # below is that a real price reaches the caller, not that pennies survive rounding.
        (tmp_path / "b.py").write_text("def b():\n" + "    x = 1  # padding\n" * 2000)

        async def _root(project: str):
            return tmp_path

        async def _scope(project: str):
            return [], []

        async def _checksums(project: str):
            return {}

        svc.resolved_root_path = _root  # type: ignore[method-assign]
        svc._repo.get_project_scope = _scope  # type: ignore[method-assign]
        svc._repo.get_checksums = _checksums  # type: ignore[method-assign]

        result = await svc.project_freshness("p")

        assert result["available"] is True
        assert result["pending"] == 2
        assert sorted(result["sample"]) == ["a.py", "b.py"]
        # An answer with a price on it is what makes "catch this up" a decision rather than a
        # leap: a project indexed before the extension set widened can be hundreds of files
        # behind without anyone having touched the code.
        assert result["estimated_cost_usd"] > 0

    async def test_a_project_whose_tree_is_gone_says_so_instead_of_claiming_freshness(self):
        svc = _status_service([])

        async def _root(project: str):
            return None

        svc.resolved_root_path = _root  # type: ignore[method-assign]

        result = await svc.project_freshness("p")

        assert result["available"] is False
        assert "no root path" in result["reason"]


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
