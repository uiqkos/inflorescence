"""Field-report round 3: a search never silently loses hits, and says what its page means.

Covers the repository's widening candidate window (INV-9), verbatim chunks for every line of a
code file, quoted matching lines, unmatched query words with the fuzzy retry, page-shape hints,
the ``"code"`` node-type group, archived paths, index drift and the price of a chunk-format
upgrade.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.code_indexer.models import NodeType
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.config import Settings
from inflorescence.db import repository as repository_module
from inflorescence.db.repository import GraphRepository, ProjectHits
from inflorescence.rag.chunker import CHUNK_MARKER_SUFFIX, VERBATIM_CHUNK_KINDS, Chunker
from inflorescence.tools import archive as archive_module
from inflorescence.tools import freshness
from inflorescence.tools.archive import demote_archived, load_archive
from inflorescence.tools.gates import NODE_TYPE_GROUPS, normalize_node_types
from inflorescence.tools.hints import page_shape_hints
from inflorescence.tools.search import register_search_tools
from inflorescence.tools.text_terms import (
    fuzzy_distance_for,
    matching_lines,
    query_words,
    tokens,
    within_edits,
)
from tests.test_rag_and_tools import (
    FakeManager,
    FakeTextSearchRepository,
    _payload,
    entity,
)
from tests.test_worktree import git, make_repo

# ---------------------------------------------------------------------------
# The widening window (INV-9)
# ---------------------------------------------------------------------------


class _GlobalIndexConn:
    """A global search index shared by several projects, answering window by window.

    ``rows`` is the whole index in score order (every project mixed together, as Memgraph's
    procedures return it); a statement sees only the first ``$fetch`` of them, and the project
    filter runs after the cut — exactly the shape that starved a project in the field.
    """

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.fetches: list[int] = []

    async def execute_query(self, query: str, params: dict[str, object] | None = None, max_records=None):  # noqa: ANN001, ANN201
        params = params or {}
        window = self.rows[: int(params["fetch"])]
        if "AS scanned" in query:
            return [{"scanned": len(window)}]
        self.fetches.append(int(params["fetch"]))
        return [row for row in window if row["project"] == params["project"]]


def _row(project: str, index: int) -> dict[str, object]:
    return {
        "project": project, "id": f"{project}/f{index}.py", "name": f"f{index}",
        "labels": ["Code", "Module"], "file_path": f"{project}/f{index}.py",
        "line_start": 1, "line_end": 2, "summary": "", "score": 100.0 - index / 1000,
    }


@pytest.mark.asyncio
async def test_text_search_widens_past_other_projects_instead_of_answering_empty() -> None:
    # 3000 rows of another project out-score every row of ours: the procedure's default
    # window of 1000 used to hold none of ours, and the project filter then returned nothing.
    rows = [_row("other", i) for i in range(3000)] + [_row("mine", i) for i in range(3000, 3005)]
    conn = _GlobalIndexConn(rows)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    hits = await repo.text_hits("symbol", "mine", "run", 5)

    assert [row["id"] for row in hits.rows] == [f"mine/f{i}.py" for i in range(3000, 3005)]
    assert hits.complete is True
    assert conn.fetches[0] == 1000 and conn.fetches[-1] > 3000


@pytest.mark.asyncio
async def test_widening_stops_as_soon_as_the_window_comes_back_short() -> None:
    rows = [_row("other", i) for i in range(50)] + [_row("mine", 60)]
    conn = _GlobalIndexConn(rows)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    hits = await repo.text_hits("summary", "mine", "run", 10)

    assert len(hits.rows) == 1 and hits.complete is True
    assert conn.fetches == [1000]  # the first window already held the whole index


@pytest.mark.asyncio
async def test_widening_reports_incomplete_at_its_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repository_module, "_SEARCH_WINDOW_CEILING", 2000)
    rows = [_row("other", i) for i in range(5000)] + [_row("mine", 9000)]
    conn = _GlobalIndexConn(rows)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    hits = await repo.text_hits("code", "mine", "run", 10)

    assert hits.rows == []
    assert hits.complete is False  # rows may exist past the widest window: not "no matches"


@pytest.mark.asyncio
async def test_widening_applies_the_callers_filters_inside_the_loop() -> None:
    # Every row in the first window is ours but outside the include path, so a filter applied
    # after the repository returned would have starved the page; inside the loop it widens.
    rows = [_row("mine", i) for i in range(4000)]
    for row in rows[3990:]:
        row["file_path"] = f"rare/{row['file_path']}"
    conn = _GlobalIndexConn(rows)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    def keep(found: list[dict[str, object]]) -> list[dict[str, object]]:
        return [row for row in found if str(row["file_path"]).startswith("rare/")]

    hits = await repo.text_hits("symbol", "mine", "run", 5, keep)

    assert len(hits.rows) == 5 and hits.complete is True


@pytest.mark.asyncio
async def test_the_window_counts_entities_not_chunks() -> None:
    # 1500 chunks of one long function of ours out-rank our other functions: counted as rows,
    # the first window "had" the page and the answer was one entity marked complete.
    big = [{**_row("mine", 0), "chunk_id": f"c{i}"} for i in range(1500)]
    rest = [_row("mine", i) for i in range(1, 6)]
    conn = _GlobalIndexConn(big + rest)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    hits = await repo.text_hits("code", "mine", "run", 4)

    assert len({row["id"] for row in hits.rows}) == 4 and hits.complete is True
    assert conn.fetches[-1] > 1500


@pytest.mark.asyncio
async def test_vector_search_widens_the_same_way() -> None:
    rows = [_row("clone", i) for i in range(1000)] + [_row("mine", i) for i in range(1000, 1010)]
    for row in rows:
        row["score"] = 0.9
    conn = _GlobalIndexConn(rows)
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    hits = await repo.vector_hits("summary", "mine", [0.1], 10)

    assert len(hits.rows) == 10 and hits.complete is True
    assert conn.fetches[0] == 256


# ---------------------------------------------------------------------------
# Every line of a code file reaches a verbatim chunk
# ---------------------------------------------------------------------------


def test_module_level_rule_table_past_line_80_is_in_a_verbatim_chunk(tmp_path: Path) -> None:
    body = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(40))
    source = tmp_path / "triage.py"
    source.write_text(
        f'"""Triage rules."""\n\n{body}\n\nRULES = (\n    Rule("cache.stale_no_etag", LAYER2),\n'
        '    Rule("fetch.empty_page", TERMINAL),\n)\n\nMIN_BATCH_SIZE = 400\n',
        encoding="utf-8",
    )
    nodes, _ = PythonAstParser().parse_file(source, tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    module_body = [c for c in chunks if c.chunk_kind == "module_body"]
    assert any("cache.stale_no_etag" in c.content for c in module_body)
    assert any("MIN_BATCH_SIZE = 400" in c.content for c in module_body)
    rules_line = source.read_text().splitlines().index('    Rule("cache.stale_no_etag", LAYER2),') + 1
    holder = next(c for c in module_body if "cache.stale_no_etag" in c.content)
    assert holder.start_line <= rules_line <= holder.end_line
    assert holder.content.splitlines()[rules_line - holder.start_line].strip().startswith("Rule(")
    # A function's lines stay in its own body chunk and are not duplicated as module code.
    assert not any("return 7" in c.content for c in module_body)


def test_type_bodies_and_exported_constants_are_verbatim(tmp_path: Path) -> None:
    source = tmp_path / "bank.ts"
    source.write_text(
        "export interface Row {\n  op: string;\n  sides: number[];\n}\n\n"
        "export const NO_REFUND: Record<string, string> = {\n  t09: 'needs a drag step',\n};\n\n"
        "export function bankTasks() {\n  return Object.keys(NO_REFUND);\n}\n",
        encoding="utf-8",
    )
    nodes, _ = JavaScriptParser().parse_file(source, tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    kinds = {(c.chunk_kind, c.node_id) for c in chunks}
    assert ("type_body", "bank.ts::Row") in kinds
    const = next(c for c in chunks if c.chunk_kind == "module_body")
    assert "t09: 'needs a drag step'" in const.content and const.node_id == "bank.ts"
    assert {"module_body", "type_body"} <= VERBATIM_CHUNK_KINDS


def test_blank_runs_between_members_make_no_chunk(tmp_path: Path) -> None:
    source = tmp_path / "plain.py"
    source.write_text("def a():\n    return 1\n\n\n\ndef b():\n    return 2\n", encoding="utf-8")
    nodes, _ = PythonAstParser().parse_file(source, tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    assert not [c for c in chunks if c.chunk_kind == "module_body"]


# ---------------------------------------------------------------------------
# Words and lines
# ---------------------------------------------------------------------------


def test_tokens_mirror_the_index_tokenizer() -> None:
    assert tokens("parser_strict task.start R19 Выбрасывание") == [
        "parser", "strict", "task", "start", "r19", "выбрасывание",
    ]
    assert query_words("stale_no_etag stale_no_etag ::") == [("stale_no_etag", ["stale", "no", "etag"])]


def test_matching_lines_quote_file_lines_best_first() -> None:
    content = "x = 1\nRule('cache.stale_no_etag', LAYER2)\nstale = 2\n# stale no etag here too\n"

    lines = matching_lines(content, 390, query_words("stale_no_etag layer2"))

    assert lines[0] == {"line": 391, "text": "Rule('cache.stale_no_etag', LAYER2)"}
    assert {"line": 393, "text": "# stale no etag here too"} in lines
    assert all(item["line"] != 392 for item in lines)  # "stale" alone is not the phrase


def test_fuzzy_distance_grows_with_the_word_and_never_applies_to_phrases() -> None:
    assert fuzzy_distance_for("доск") == 0
    assert fuzzy_distance_for("keepsOrder") == 2
    assert fuzzy_distance_for("print") == 1
    assert fuzzy_distance_for("parser_strict") == 0
    assert within_edits("выбрасывание", "выбрасывания", 1)
    assert not within_edits("выбросить", "выбрасывания", 2)


# ---------------------------------------------------------------------------
# search_text: lines, unmatched words, the fuzzy retry
# ---------------------------------------------------------------------------


class _WordAwareTextRepo(FakeTextSearchRepository):
    """A text index that knows which words it holds, so probes and fuzzy retries are real."""

    def __init__(self, vocabulary: set[str], fuzzy_rows: list[dict[str, object]] | None = None) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.fuzzy_rows = fuzzy_rows or []
        self.fuzzy_calls: list[tuple[str, int]] = []

    async def text_hits(self, kind, project, query, needed, keep=lambda rows: rows, fuzzy_distance=0):  # noqa: ANN001, ANN201
        if fuzzy_distance:
            self.fuzzy_calls.append((query, fuzzy_distance))
            return ProjectHits(keep([dict(r) for r in self.fuzzy_rows]), complete=True)
        if needed == 1:  # a word probe
            known = all(part in self.vocabulary for part in tokens(query))
            return ProjectHits([{"id": "x"}] if known else [], complete=True)
        return await super().text_hits(kind, project, query, needed, keep)


def _chunk_row(node_id: str, path: str, content: str, start: int, score: float = 3.0) -> dict[str, object]:
    return {
        **entity(node_id, node_id.rsplit("::", 1)[-1], "module", path, 1, 200, "Module."),
        "score": score, "chunk_id": f"chunk:{node_id}:{start}", "chunk_kind": "module_body",
        "match_start_line": start, "match_end_line": start + content.count("\n"),
        "chunk_content": content,
    }


@pytest.mark.asyncio
async def test_search_text_quotes_the_matching_lines_and_never_leaks_chunk_text() -> None:
    mcp = FastMCP("test")
    repo = _WordAwareTextRepo({"stale", "no", "etag"})
    repo.code_rows = [_chunk_row("src/triage.py", "src/triage.py", "RULES = (\n    Rule('cache.stale_no_etag', L2),\n)", 389)]
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/p", "query": "stale_no_etag"}))

    match = payload["items"][0]["matches"][0]
    assert match["lines"] == [{"line": 390, "text": "Rule('cache.stale_no_etag', L2),"}]
    assert "chunk_content" not in str(payload)
    assert "unmatched_terms" not in payload["meta"]


@pytest.mark.asyncio
async def test_search_text_names_words_the_index_does_not_know_and_retries_them_fuzzily() -> None:
    mcp = FastMCP("test")
    fuzzy_row = _chunk_row("docs/vision.md", "docs/vision.md", "Переписывание — не выбрасывание.", 66)
    fuzzy_row["chunk_kind"] = "document"
    repo = _WordAwareTextRepo({"переписывание"}, fuzzy_rows=[fuzzy_row])
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool("search_text", {"directory": "/tmp/p", "query": "выбрасывания"}))

    assert payload["meta"]["unmatched_terms"] == ["выбрасывания"]
    assert repo.fuzzy_calls == [("выбрасывания", 2)]
    hit = payload["items"][0]
    assert hit["fuzzy"] is True
    assert hit["matches"][0]["lines"] == [{"line": 66, "text": "Переписывание — не выбрасывание."}]
    assert any("no stemming" in hint for hint in payload["meta"]["hints"])
    # A fuzzy page is not an empty page, so no coverage block claims anything about absence.
    assert "index_coverage" not in payload["meta"]


@pytest.mark.asyncio
async def test_search_text_says_when_known_words_were_filtered_out() -> None:
    mcp = FastMCP("test")
    repo = _WordAwareTextRepo({"harvest"})
    register_search_tools(mcp, FakeManager(), repo, Settings(_env_file=None))

    payload = _payload(await mcp.call_tool(
        "search_text", {"directory": "/tmp/p", "query": "harvest", "include_paths": ["deploy/"]},
    ))

    assert payload["items"] == []
    assert "unmatched_terms" not in payload["meta"]
    assert any("outside the scope" in hint for hint in payload["meta"]["hints"])


# ---------------------------------------------------------------------------
# Page-shape hints and the "code" group
# ---------------------------------------------------------------------------


def test_page_shape_hints_name_a_test_or_document_dominated_page() -> None:
    tests_page = [{"entity": {"node_type": "function"}, "is_test": True}] * 3 + [{"entity": {"node_type": "function"}}]
    docs_page = [{"entity": {"node_type": "document"}}] * 3 + [{"entity": {"node_type": "class"}}]

    assert "include_tests=false" in page_shape_hints(tests_page, include_tests=True, node_types=None)[0]
    assert page_shape_hints(tests_page, include_tests=False, node_types=None) == []
    assert 'node_types=["code"]' in page_shape_hints(docs_page, include_tests=True, node_types=None)[0]
    assert page_shape_hints(docs_page[:2], include_tests=True, node_types=None) == []


def _hit_at(path: str) -> dict[str, object]:
    return {"entity": {"node_type": "function", "file_path": path}}


def test_page_shape_hints_name_the_directory_that_buries_the_rest() -> None:
    # A shape from a field report: most of the page is copies from one sample tree, the two
    # real answers are elsewhere. The directory is read off the hits — nothing is predeclared.
    page = [_hit_at(f"samples/v{i}/app.js") for i in range(4)] + [
        _hit_at("backend/content/checker.go"),
        _hit_at("frontend/components/renderers.tsx"),
    ]

    (hint,) = page_shape_hints(page, include_tests=True, node_types=None)

    assert hint.startswith("4 of 6 hits come from samples/.")
    assert 'exclude_paths=["samples/"]' in hint


def test_dominant_directory_hint_picks_the_deepest_tree_and_respects_the_callers_scope() -> None:
    page = [_hit_at(f"web/src/tasks/bank/t{i}.json") for i in range(3)] + [
        _hit_at("web/src/engine/formula.ts"),
        _hit_at("web/src/engine/rules.ts"),
    ]

    # Scoped to web/src, the bank tree below it still buries the engine: name the bank.
    (hint,) = page_shape_hints(page, include_tests=True, node_types=None, include_paths=["web/src"])
    assert 'exclude_paths=["web/src/tasks/bank/"]' in hint
    # The caller asked about that very tree: nothing to say.
    assert page_shape_hints(page, include_tests=True, node_types=None, include_paths=["web/src/tasks/bank/**"]) == []
    # A homogeneous page (every hit in one tree) buries nothing.
    same = [_hit_at(f"web/src/engine/f{i}.ts") for i in range(5)]
    assert page_shape_hints(same, include_tests=True, node_types=None) == []


def test_search_descriptions_lead_with_scoping_advice() -> None:
    from inflorescence.tools.search import (
        SEARCH_CODE_DESCRIPTION,
        SEARCH_HYBRID_DESCRIPTION,
        SEARCH_SEMANTIC_DESCRIPTION,
        SEARCH_TEXT_DESCRIPTION,
    )

    for description in (
        SEARCH_CODE_DESCRIPTION, SEARCH_SEMANTIC_DESCRIPTION, SEARCH_HYBRID_DESCRIPTION, SEARCH_TEXT_DESCRIPTION
    ):
        scope_at = description.index("STRONGLY RECOMMENDED")
        assert scope_at < description.index("``node_types`` values")
        assert "exclude_paths" in description[scope_at : scope_at + 1500]


def test_code_group_expands_to_every_type_but_documents_and_directories() -> None:
    expanded = normalize_node_types(["code"])

    assert isinstance(expanded, list)
    assert set(expanded) == {t.value for t in NodeType} - {"document", "directory"}
    assert expanded == list(NODE_TYPE_GROUPS["code"])
    assert normalize_node_types(["Code", "function"]) == list(NODE_TYPE_GROUPS["code"])
    error = normalize_node_types(["codez"])
    assert isinstance(error, dict) and "'code'" in error["error"]["message"]


# ---------------------------------------------------------------------------
# Archived paths
# ---------------------------------------------------------------------------


def test_archived_hits_are_flagged_and_listed_after_live_ones(tmp_path: Path) -> None:
    archive_module._cache.clear()
    (tmp_path / ".inflorescencearchive").write_text("mocks/\ndocs/prototypes/**\n", encoding="utf-8")
    hits = [
        {"entity": {"file_path": "mocks/v4/app.js"}, "score": 0.9},
        {"entity": {"file_path": "src/registry.ts"}, "score": 0.6},
        {"entity": {"file_path": "docs/prototypes/step/app.js"}, "score": 0.5},
        {"entity": {"file_path": "backend/checker.go"}, "score": 0.4},
    ]

    ordered, archived = demote_archived(hits, load_archive(tmp_path))

    assert archived == 2
    assert [h["entity"]["file_path"] for h in ordered] == [
        "src/registry.ts", "backend/checker.go", "mocks/v4/app.js", "docs/prototypes/step/app.js",
    ]
    assert [h.get("archived") for h in ordered] == [None, None, True, True]


def test_no_archive_file_changes_nothing(tmp_path: Path) -> None:
    archive_module._cache.clear()
    hits = [{"entity": {"file_path": "mocks/a.js"}}]

    assert demote_archived(hits, load_archive(tmp_path)) == (hits, 0)


# ---------------------------------------------------------------------------
# Index drift
# ---------------------------------------------------------------------------


def test_drift_counts_indexed_files_that_differ_after_head_moved(tmp_path: Path) -> None:
    freshness._drift_cache.clear()
    repo_dir = make_repo(tmp_path / "repo")
    indexed_head = git("-C", str(repo_dir), "rev-parse", "HEAD").strip()
    tracked = sorted(p.name for p in repo_dir.iterdir() if p.is_file())
    stored = {
        name: freshness.FileChecksum.compute(repo_dir / name, repo_dir).md5 for name in tracked
    }
    # A commit lands without the index following it: one indexed file changes.
    changed = tracked[0]
    (repo_dir / changed).write_text("rewritten by a merge\n", encoding="utf-8")
    git("-C", str(repo_dir), "commit", "-am", "merge")

    drift = freshness.measure_drift(repo_dir, stored, indexed_head)

    assert drift == (1, [changed], False)
    # Same commit as the index: nothing to say, and nothing is hashed.
    head_now = git("-C", str(repo_dir), "rev-parse", "HEAD").strip()
    assert freshness.measure_drift(repo_dir, stored, head_now) is None
    assert freshness.measure_drift(repo_dir, stored, None) is None


def test_drift_cache_sees_a_reindex_that_kept_the_file_count(tmp_path: Path) -> None:
    freshness._drift_cache.clear()
    repo_dir = make_repo(tmp_path / "repo")
    indexed_head = git("-C", str(repo_dir), "rev-parse", "HEAD").strip()
    tracked = sorted(p.name for p in repo_dir.iterdir() if p.is_file())
    stale = {name: "0" * 32 for name in tracked}  # an index older than every file
    (repo_dir / tracked[0]).write_text("moved on\n", encoding="utf-8")
    git("-C", str(repo_dir), "commit", "-am", "move")
    assert freshness.measure_drift(repo_dir, stale, indexed_head)[0] == len(tracked)

    # The watcher re-indexes every file without moving HEAD: same count, new checksums.
    caught_up = {
        name: freshness.FileChecksum.compute(repo_dir / name, repo_dir).md5 for name in tracked
    }
    assert freshness.measure_drift(repo_dir, caught_up, indexed_head) is None


def test_a_partial_drift_measurement_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freshness._drift_cache.clear()
    repo_dir = make_repo(tmp_path / "repo")
    indexed_head = git("-C", str(repo_dir), "rev-parse", "HEAD").strip()
    (repo_dir / "new.txt").write_text("x\n", encoding="utf-8")
    git("-C", str(repo_dir), "add", "new.txt")
    git("-C", str(repo_dir), "commit", "-m", "add")
    monkeypatch.setattr(freshness, "DRIFT_TIME_BUDGET_S", -1.0)

    # Nothing was hashed: that is "unknown", not "no drift".
    assert freshness.measure_drift(repo_dir, {"new.txt": "0" * 32}, indexed_head) == (0, [], True)
    assert freshness._drift_cache == {}
    assert "unknown" in freshness._drift_notice(repo_dir, indexed_head, (0, [], True))


def test_drift_notice_names_the_commits_the_files_and_the_command(tmp_path: Path) -> None:
    notice = freshness._drift_notice(tmp_path, "a" * 40, (3, ["x.py", "y.py"], False))

    assert "aaaaaaa" in notice and "3 indexed files differ" in notice
    assert "x.py, y.py…" in notice
    assert f"index_directory({str(tmp_path)!r})" in notice


# ---------------------------------------------------------------------------
# A chunk-format upgrade is priced before it is spent (INV-10)
# ---------------------------------------------------------------------------


def test_chunk_marker_carries_the_format_version() -> None:
    assert CHUNK_MARKER_SUFFIX.startswith("@chunks-v")


@pytest.mark.asyncio
async def test_repository_stamps_and_detects_with_the_format_suffix() -> None:
    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def execute_write_query(self, query: str, params: dict[str, object]):  # noqa: ANN201
            self.calls.append((query, params))
            return None

        async def execute_query(self, query: str, params: dict[str, object]):  # noqa: ANN201
            self.calls.append((query, params))
            return [{"file_path": "a.py"}]

    conn = _Conn()
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    await repo.mark_files_chunks_covered("p", [{"file_path": "a.py", "checksum": "abc"}])
    assert await repo.get_files_with_stale_chunks("p") == ["a.py"]

    assert conn.calls[0][1]["marker_suffix"] == CHUNK_MARKER_SUFFIX
    assert "row.checksum + $marker_suffix" in conn.calls[0][0]
    assert conn.calls[1][1]["marker_suffix"] == CHUNK_MARKER_SUFFIX
    assert "m.checksum + $marker_suffix" in conn.calls[1][0]

