"""``find_text``: exhaustive line search over the project's files, tied back to the graph."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.tools import find as find_module
from inflorescence.tools.find import FIND_MAX_LINE_CHARS, _scan_files, register_find_tools
from tests.test_rag_and_tools import FakeGraphRepositoryV2, FakeManager, entity


def _payload(result: object) -> dict:
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def _write(root: Path, rel: str, text: str) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.md5(path.read_bytes()).hexdigest()


class ScanningManager(FakeManager):
    """Answers the "what is on disk and not indexed" scan from the real builder."""

    def __init__(self) -> None:
        self.max_examples: list[int | None] = []

    async def scan_uncovered(self, directory, indexed_paths, scope=None, max_examples=None):
        from inflorescence.code_indexer.graph_builder import GraphBuilder
        from inflorescence.code_indexer.models import IndexerConfig

        self.max_examples.append(max_examples)
        builder = GraphBuilder(
            repo=None, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)
        )
        return builder.scan_uncovered(Path(directory), list(indexed_paths), scope, max_examples=max_examples or 5)


class FileEntities(FakeGraphRepositoryV2):
    def __init__(self, entities: list[dict[str, object]]) -> None:
        super().__init__()
        self.entities = entities
        self.asked: list[list[str]] = []

    async def get_file_entities(self, project, file_paths):
        self.asked.append(list(file_paths))
        return [e for e in self.entities if e["file_path"] in file_paths]


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, FileEntities]:
    checksums = {
        "web/theme.css": _write(tmp_path, "web/theme.css", ":root {\n  --color-brand: #0a3;\n}\n.a { color: var(--color-brand); }\n"),
        "web/board.ts": _write(
            tmp_path,
            "web/board.ts",
            "export const PALETTE = {\n  bg: 'var(--color-brand)',\n}\n\nexport function paint() {\n"
            "  return 'var(--color-Brand)'\n}\n",
        ),
        "web/board.test.ts": _write(tmp_path, "web/board.test.ts", "expect(css).toContain('--color-brand')\n"),
        "web/logo.png": _write(tmp_path, "web/logo.png", "\0PNG--color-brand"),
    }
    repo = FileEntities([
        entity("web/board.ts", "board", "module", "web/board.ts", 1, 7, "Board."),
        entity("web/board.ts::PALETTE", "PALETTE", "variable", "web/board.ts", 1, 3, ""),
        entity("web/board.ts::paint", "paint", "function", "web/board.ts", 5, 7, "Paints."),
        entity("web/theme.css", "theme", "document", "web/theme.css", 1, 4, "Theme."),
    ])
    repo.checksums = checksums
    return tmp_path, repo


async def _call(repo, root: Path, **arguments) -> dict:
    mcp = FastMCP("test")
    register_find_tools(mcp, ScanningManager(), repo)
    return _payload(await mcp.call_tool("find_text", {"directory": str(root), **arguments}))


@pytest.mark.asyncio
async def test_every_occurrence_in_path_and_line_order_with_its_owner(project) -> None:
    root, repo = project
    payload = await _call(repo, root, pattern="--color-brand")

    assert [(i["file_path"], i["line"]) for i in payload["items"]] == [
        ("web/board.test.ts", 1),
        ("web/board.ts", 2),
        ("web/theme.css", 2),
        ("web/theme.css", 4),
    ]
    by_line = {(i["file_path"], i["line"]): i for i in payload["items"]}
    assert by_line[("web/board.ts", 2)]["entity"]["id"] == "web/board.ts::PALETTE"  # innermost span
    assert by_line[("web/board.ts", 2)]["text"] == "  bg: 'var(--color-brand)',"
    assert by_line[("web/board.test.ts", 1)]["is_test"] is True
    assert payload["meta"]["total_matches"] == 4 and payload["meta"]["files_matched"] == 3
    assert payload["meta"]["files_searched"] == 3, "the binary file is skipped, not searched"


@pytest.mark.asyncio
async def test_case_regex_tests_and_pagination(project) -> None:
    root, repo = project

    folded = await _call(repo, root, pattern="--COLOR-BRAND", case_sensitive=False, include_tests=False)
    assert folded["meta"]["total_matches"] == 4  # the `--color-Brand` line joins, the test file leaves
    assert all("is_test" not in item for item in folded["items"])

    anchored = await _call(repo, root, pattern=r"^\s+return", regex=True)
    assert [(i["file_path"], i["line"], i["entity"]["id"]) for i in anchored["items"]] == [
        ("web/board.ts", 6, "web/board.ts::paint")
    ]

    first = await _call(repo, root, pattern="brand", limit=2)
    assert first["page"]["has_more"] is True and first["meta"]["total_matches"] == 4
    rest = await _call(repo, root, pattern="brand", limit=2, offset=2)
    assert [(i["file_path"], i["line"]) for i in rest["items"]] == [("web/theme.css", 2), ("web/theme.css", 4)]


@pytest.mark.asyncio
async def test_facets_count_every_match_not_just_the_page(project) -> None:
    root, repo = project
    repo.checksums["legacy/v1/old.css"] = _write(
        root, "legacy/v1/old.css", ".x { color: var(--color-brand); }\n.y { color: var(--color-brand); }\n"
    )

    payload = await _call(repo, root, pattern="--color-brand", limit=1)

    assert len(payload["items"]) == 1
    facets = payload["meta"]["facets"]
    assert facets["counted"] == payload["meta"]["total_matches"] == 6
    assert facets["common_path"] == "./"
    assert facets["by_directory"] == [{"path": "web/", "count": 4}, {"path": "legacy/", "count": 2}]
    assert facets["tests"] == 1
    narrowed = await _call(repo, root, pattern="--color-brand", exclude_paths=["legacy/"], include_tests=False)
    assert narrowed["meta"]["facets"] == {
        "counted": 3, "common_path": "web/", "by_directory": [{"path": "web/", "count": 3, "direct": True}],
    }


@pytest.mark.asyncio
async def test_invalid_input_is_an_error_not_an_empty_page(project) -> None:
    root, repo = project

    broken = await _call(repo, root, pattern="(unclosed", regex=True)
    assert broken["error"]["code"] == "invalid_regex"
    assert "regex=false" in broken["error"]["message"]
    literal = await _call(repo, root, pattern="(unclosed")
    assert literal["items"] == [] and "error" not in literal
    assert (await _call(repo, root, pattern=""))["error"]["code"] == "invalid_argument"


@pytest.mark.asyncio
async def test_files_not_yet_indexed_are_read_and_flagged(project) -> None:
    root, repo = project
    _write(root, "web/new_panel.ts", "const x = 'var(--color-brand)'\n")

    payload = await _call(repo, root, pattern="--color-brand", include_paths=["web/"])

    new = [i for i in payload["items"] if i["file_path"] == "web/new_panel.ts"]
    assert new == [{"file_path": "web/new_panel.ts", "line": 1, "text": "const x = 'var(--color-brand)'", "indexed": False}]
    assert payload["meta"]["unindexed_files_searched"] == 1
    assert "web/new_panel.ts" not in sum(repo.asked, []), "no graph lookup for a file the graph does not hold"


@pytest.mark.asyncio
async def test_a_changed_file_is_read_as_it_is_now_and_flagged_stale(project) -> None:
    root, repo = project
    (root / "web/theme.css").write_text("/* moved */\n:root { --color-brand: #0a3; }\n", encoding="utf-8")

    payload = await _call(repo, root, pattern="--color-brand", include_paths=["web/theme.css"])

    assert payload["items"] == [
        {"file_path": "web/theme.css", "line": 2, "text": ":root { --color-brand: #0a3; }",
         "entity": {"id": "web/theme.css", "name": "theme", "node_type": "document", "line_start": 1, "line_end": 4},
         "stale": True},
    ]
    assert payload["meta"]["index_freshness"]["stale"] == ["web/theme.css"]


def test_long_lines_are_quoted_around_the_match_and_budgets_are_reported(tmp_path: Path) -> None:
    (tmp_path / "bundle.js").write_text("x" * 5000 + "NEEDLE" + "y" * 5000 + "\n", encoding="utf-8")
    (tmp_path / "b.js").write_text("NEEDLE\n", encoding="utf-8")
    compiled = find_module._compile("NEEDLE", is_regex=False, case_sensitive=True)

    scan = _scan_files(tmp_path, ["bundle.js", "b.js"], compiled, keep=10)
    text = scan.kept[0][2]
    assert "NEEDLE" in text and len(text) <= FIND_MAX_LINE_CHARS + 2

    # The budget is checked against a file's size before it is read: bundle.js (10 007 bytes)
    # fits, b.js no longer does.
    stopped = _scan_files(tmp_path, ["bundle.js", "b.js"], compiled, keep=10, byte_budget=10_010)
    assert stopped.files_searched == 1 and stopped.not_searched == 1
    # A file bigger than the whole budget is never read into memory at all.
    assert _scan_files(tmp_path, ["bundle.js"], compiled, keep=10, byte_budget=100).files_searched == 0


def test_a_catastrophic_regex_is_stopped_inside_the_line_and_named(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("ok\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x" * 5000 + "\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("ok\n", encoding="utf-8")
    compiled = find_module._compile(r"(x+x+)+y", is_regex=True, case_sensitive=True)

    scan = _scan_files(tmp_path, ["a.txt", "b.txt", "c.txt"], compiled, keep=10, time_budget=0.3)

    assert scan.regex_timed_out_in == "b.txt"
    assert scan.files_searched == 2 and scan.not_searched == 2


def test_line_numbers_count_newlines_only_like_grep(tmp_path: Path) -> None:
    # Form feed and U+2028 are line breaks to str.splitlines, not to grep or an editor.
    (tmp_path / "a.py").write_text("one\x0ctwo\nthree \u2028 four\nNEEDLE\n", encoding="utf-8")
    compiled = find_module._compile("NEEDLE", is_regex=False, case_sensitive=True)

    assert [line for _, line, _ in _scan_files(tmp_path, ["a.py"], compiled, keep=10).kept] == [3]


@pytest.mark.asyncio
async def test_a_budget_stop_marks_the_page_truncated_and_says_so(project, monkeypatch) -> None:
    root, repo = project
    monkeypatch.setattr(find_module, "FIND_MAX_BYTES", 1)
    original = find_module._scan_files
    monkeypatch.setattr(
        find_module, "_scan_files", lambda *a, **k: original(*a, **{**k, "byte_budget": 1})
    )

    payload = await _call(repo, root, pattern="brand")

    assert payload["page"]["truncated"] is True
    assert payload["meta"]["files_not_searched"] >= 1
    assert "hints" not in payload["meta"]
