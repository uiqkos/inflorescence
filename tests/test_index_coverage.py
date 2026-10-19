"""An empty answer must not read as "this does not exist".

Every honesty signal this server had was defined against something the index *holds*:
``stale`` compares a stored checksum to disk, ``file_missing`` reports a node whose file is
gone, ``page.truncated`` says a scan stopped early. A file that was never indexed has no
node, so none of them fire — the response is an empty page with ``"checked": 0``, empty
``stale``/``missing`` lists and a watcher line saying a daemon is following the tree.

The field report (10.09.2026) that these guard: ``search_entities(query="wave_queue")`` returned
exactly that over a project holding a 174-line ``scripts/wave_queue.py``, and the agent
concluded the file did not exist — one step from writing it a second time.

The contract:
  * an empty first page reports what the index covers and what it demonstrably does not;
  * a file on disk that the indexer *would* admit and does not hold is named;
  * a file it would never admit (ignored, generated, a lock file, an unindexable type) is
    not named — a coverage warning that cries wolf is read as noise and then ignored;
  * the check states its own bounds (capped scans, unseen directories) instead of implying
    it looked everywhere;
  * a page with results, or a page past the first, carries no such block;
  * and the whole thing is best-effort: it annotates an answer the caller already has, so a
    failure inside it must leave that answer untouched rather than raise.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import IndexerConfig
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import FreshnessChecker, describe_nested_checkouts

_NO_LLM = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)


class _Manager:
    """Stands in for ProjectManager: the real builder, none of the database."""

    def __init__(self, builder: GraphBuilder | None = None) -> None:
        self._builder = builder or GraphBuilder(repo=None, llm=None, config=_NO_LLM)
        self.calls: list[tuple[str, list[str], list[str] | None]] = []

    async def scan_uncovered(
        self, directory: str, indexed_paths: Any, scope: Any = None
    ) -> dict[str, object]:
        paths = list(indexed_paths)
        self.calls.append((directory, paths, list(scope) if scope else None))
        return self._builder.scan_uncovered(Path(directory), paths, scope)


class _FailingManager:
    async def scan_uncovered(self, directory: str, indexed_paths: Any, scope: Any = None) -> dict:
        raise RuntimeError("Memgraph went away mid-annotation")


def _empty_page(meta: dict[str, Any] | None = None, offset: int = 0) -> dict[str, Any]:
    return {
        "items": [],
        "page": {"limit": 10, "offset": offset, "next_offset": None, "has_more": False},
        "meta": dict(meta or {}),
    }


def _project(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "runner.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (tmp_path / "scripts" / "wave_queue.py").write_text("QUEUE = []\n" * 20, encoding="utf-8")


# ---------------------------------------------------------------------------
# GraphBuilder.scan_uncovered — the on-disk half
# ---------------------------------------------------------------------------


def test_scan_names_a_file_on_disk_that_the_index_does_not_hold(tmp_path: Path) -> None:
    _project(tmp_path)
    builder = GraphBuilder(repo=None, llm=None, config=_NO_LLM)

    report = builder.scan_uncovered(tmp_path, ["scripts/runner.py"])

    assert report["not_indexed"] == ["scripts/wave_queue.py"]
    assert report["not_indexed_total"] == 1
    assert report["files_indexed"] == 1
    assert report["capped"] is False


def test_scan_reports_nothing_when_every_file_is_indexed(tmp_path: Path) -> None:
    """The other half of the contract: a clean check is what licenses "it really is absent"."""
    _project(tmp_path)
    builder = GraphBuilder(repo=None, llm=None, config=_NO_LLM)

    report = builder.scan_uncovered(
        tmp_path, ["scripts/runner.py", "scripts/wave_queue.py"]
    )

    assert report["not_indexed"] == []
    assert report["not_indexed_total"] == 0
    # The project root plus scripts/: the root is always checked, indexed files under it or not.
    assert report["dirs_checked"] == 2


def test_scan_ignores_files_the_indexer_would_never_admit(tmp_path: Path) -> None:
    """Otherwise every empty page reports hundreds of "missing" files and means nothing.

    A lock file, an ignored path, a binary and a file type with no parser are all *correctly*
    absent from the index; listing them as coverage gaps would bury the one file that is not.
    """
    _project(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.md\n", encoding="utf-8")
    (tmp_path / "secret.md").write_text("ignored on purpose\n", encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "notes.xml").write_text("<notes/>\n", encoding="utf-8")

    report = GraphBuilder(repo=None, llm=None, config=_NO_LLM).scan_uncovered(
        tmp_path, ["scripts/runner.py", ".gitignore"]
    )

    assert report["not_indexed"] == ["scripts/wave_queue.py"]


def test_scan_skips_a_file_left_out_by_the_size_policy(tmp_path: Path) -> None:
    """An oversized file is reported by the indexer at index time, not as a coverage gap."""
    _project(tmp_path)
    (tmp_path / "huge.md").write_text("x" * 4096, encoding="utf-8")
    config = IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False, max_file_size_bytes=1024)

    report = GraphBuilder(repo=None, llm=None, config=config).scan_uncovered(
        tmp_path, ["scripts/runner.py", "scripts/wave_queue.py"]
    )

    assert report["not_indexed"] == []


def test_scope_narrows_the_directories_checked(tmp_path: Path) -> None:
    """The query's own path filters bound the check, so a scoped search pays a scoped price."""
    _project(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "runbook.md").write_text("how to launch a wave\n", encoding="utf-8")
    builder = GraphBuilder(repo=None, llm=None, config=_NO_LLM)

    scoped = builder.scan_uncovered(
        tmp_path, ["scripts/runner.py", "docs/index.md"], scope=["docs", "docs/**"]
    )

    # docs/ and the root above it — never scripts/, which the caller excluded by scoping.
    assert scoped["not_indexed"] == ["docs/runbook.md"]
    assert scoped["dirs_checked"] == 2


def test_a_capped_scan_says_it_was_capped(tmp_path: Path) -> None:
    """Stopping early is fine; stopping early *silently* is the failure mode being fixed."""
    indexed = []
    for i in range(6):
        directory = tmp_path / f"pkg{i}"
        directory.mkdir()
        (directory / "mod.py").write_text("x = 1\n", encoding="utf-8")
        indexed.append(f"pkg{i}/mod.py")

    report = GraphBuilder(repo=None, llm=None, config=_NO_LLM).scan_uncovered(
        tmp_path, indexed, max_dirs=2
    )

    assert report["capped"] is True
    assert report["dirs_checked"] == 2
    assert report["dirs_total"] == 7  # six packages plus the project root


def test_scan_never_opens_a_directory_the_index_does_not_know(tmp_path: Path) -> None:
    """The bound that makes this affordable: no walk into node_modules, ever.

    The price is that a brand-new directory is invisible to the check — stated in the block's
    own wording rather than papered over.
    """
    _project(tmp_path)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "left-pad.js").write_text("module.exports = 1\n", encoding="utf-8")
    opened: list[str] = []
    real_scandir = os.scandir

    def _recording_scandir(path):  # type: ignore[no-untyped-def]
        opened.append(str(path))
        return real_scandir(path)

    builder = GraphBuilder(repo=None, llm=None, config=_NO_LLM)
    try:
        os.scandir = _recording_scandir  # type: ignore[assignment]
        report = builder.scan_uncovered(tmp_path, ["scripts/runner.py"])
    finally:
        os.scandir = real_scandir  # type: ignore[assignment]

    assert not any("node_modules" in path for path in opened)
    assert all("node_modules" not in path for path in report["not_indexed"])  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# note_coverage — what the answer actually says
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_page_names_the_file_it_could_not_have_found(tmp_path: Path) -> None:
    """The field report, reproduced: query returns nothing, the file is right there."""
    _project(tmp_path)
    checker = FreshnessChecker(tmp_path, {"scripts/runner.py": "md5"})

    payload = await note_coverage(_empty_page(), _Manager(), str(tmp_path), checker)

    coverage = payload["meta"]["index_coverage"]
    assert coverage["not_indexed"] == ["scripts/wave_queue.py"]
    assert coverage["not_indexed_total"] == 1
    assert "NOT evidence of absence" in coverage["note"]
    assert "index_directory" in coverage["note"]


@pytest.mark.asyncio
async def test_an_empty_page_over_a_complete_index_says_so_positively(tmp_path: Path) -> None:
    """"Nothing matched" becomes a usable conclusion only once coverage is checked."""
    _project(tmp_path)
    checker = FreshnessChecker(
        tmp_path, {"scripts/runner.py": "md5", "scripts/wave_queue.py": "md5"}
    )

    payload = await note_coverage(_empty_page(), _Manager(), str(tmp_path), checker)

    coverage = payload["meta"]["index_coverage"]
    assert "not_indexed" not in coverage
    assert "evidence of absence" in coverage["note"]
    assert coverage["files_indexed"] == 2


@pytest.mark.asyncio
async def test_a_truncated_empty_page_is_never_called_evidence_of_absence(tmp_path: Path) -> None:
    """Complete coverage, but the search stopped at its candidate ceiling.

    Field case: search_semantic(node_types=["directory"]) came back empty and truncated over
    a project with 39 directories — they never reached the shared index's global top-k — and
    the note still said "evidence of absence". Coverage cannot vouch for a search that did
    not look at everything.
    """
    _project(tmp_path)
    checker = FreshnessChecker(
        tmp_path, {"scripts/runner.py": "md5", "scripts/wave_queue.py": "md5"}
    )
    page = _empty_page()
    page["page"]["truncated"] = True

    payload = await note_coverage(page, _Manager(), str(tmp_path), checker)

    note = payload["meta"]["index_coverage"]["note"]
    assert "NOT evidence of absence" in note
    assert "truncated" in note
    assert "not a coverage gap" in note


@pytest.mark.asyncio
async def test_the_checkers_inventory_is_reused_rather_than_re_read(tmp_path: Path) -> None:
    """The freshness checker already loaded the checksums; the coverage check costs no round trip."""
    _project(tmp_path)
    manager = _Manager()
    checker = FreshnessChecker(tmp_path, {"scripts/runner.py": "md5"})

    await note_coverage(_empty_page(), manager, str(tmp_path), checker)

    assert manager.calls[0][1] == ["scripts/runner.py"]


@pytest.mark.asyncio
async def test_the_scope_comes_from_the_filters_the_tool_echoed(tmp_path: Path) -> None:
    _project(tmp_path)
    manager = _Manager()
    page = _empty_page({"path_filters": {"include": ["scripts", "scripts/**"], "exclude": []}})

    await note_coverage(page, manager, str(tmp_path), FreshnessChecker(tmp_path, {}))

    assert manager.calls[0][2] == ["scripts", "scripts/**"]


@pytest.mark.asyncio
async def test_a_page_with_results_carries_no_coverage_block(tmp_path: Path) -> None:
    _project(tmp_path)
    manager = _Manager()
    page = {"items": [{"id": "scripts/runner.py"}], "page": {"offset": 0}, "meta": {}}

    payload = await note_coverage(page, manager, str(tmp_path), FreshnessChecker(tmp_path, {}))

    assert "index_coverage" not in payload["meta"]
    assert manager.calls == []


@pytest.mark.asyncio
async def test_a_page_past_the_end_is_not_a_claim_about_the_project(tmp_path: Path) -> None:
    """An empty page at offset=40 is pagination, not absence — no block, and no scan cost."""
    _project(tmp_path)
    manager = _Manager()

    payload = await note_coverage(
        _empty_page(offset=40), manager, str(tmp_path), FreshnessChecker(tmp_path, {})
    )

    assert "index_coverage" not in payload["meta"]
    assert manager.calls == []


@pytest.mark.asyncio
async def test_a_failing_coverage_check_leaves_the_answer_intact(tmp_path: Path) -> None:
    """It annotates an answer the caller already has; it may never take that answer away."""
    page = _empty_page({"project": "p"})

    payload = await note_coverage(page, _FailingManager(), str(tmp_path), FreshnessChecker(tmp_path, {}))

    assert payload["meta"] == {"project": "p"}
    assert payload["items"] == []


# ---------------------------------------------------------------------------
# Nested checkouts — answered_from names the outer tree, the files live elsewhere
# ---------------------------------------------------------------------------


def _fake_checkout(root: Path, branch: str, sha: str) -> None:
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    refs = git / "refs" / "heads"
    refs.mkdir(parents=True)
    (refs / branch).write_text(f"{sha}\n", encoding="utf-8")


def test_a_subproject_with_its_own_head_is_named(tmp_path: Path) -> None:
    """A monorepo's outer branch says nothing about code inside a nested checkout.

    Field report: ``answered_from`` read ``main @ 1a2b3c4`` while every file on the page
    came from a subproject on ``feature/nightly-batch`` at a different commit.
    """
    _fake_checkout(tmp_path, "master", "1a2b3c4d5e6f7")
    _fake_checkout(tmp_path / "worker", "feature-nightly-batch", "1234567abcdef")

    lines = describe_nested_checkouts(tmp_path, ["worker/scripts/runner.py"])

    assert lines == ["worker (feature-nightly-batch @ 1234567)"]


def test_files_of_the_outer_checkout_report_nothing(tmp_path: Path) -> None:
    """The ordinary case pays no tokens: only a genuine nested checkout is worth a sentence."""
    _fake_checkout(tmp_path, "master", "1a2b3c4d5e6f7")
    (tmp_path / "src").mkdir()

    assert describe_nested_checkouts(tmp_path, ["src/app.py", "README.md"]) == []


def test_the_nested_checkout_line_reaches_the_response(tmp_path: Path) -> None:
    _fake_checkout(tmp_path, "master", "1a2b3c4d5e6f7")
    _fake_checkout(tmp_path / "worker", "wave", "1234567abcdef")
    (tmp_path / "worker" / "app.py").write_text("x = 1\n", encoding="utf-8")
    checker = FreshnessChecker(tmp_path, {}, origin="/monorepo (main @ 1a2b3c4)")

    payload = checker.stamp({"items": [{"file_path": "worker/app.py", "id": "x"}], "meta": {}})

    assert "worker (wave @ 1234567)" in payload["meta"]["nested_checkouts"]


# ---------------------------------------------------------------------------
# The watcher's pre-filter has to know about name-routed files too
# ---------------------------------------------------------------------------


def test_the_watcher_follows_saves_of_name_routed_documents() -> None:
    """A suffix-only pre-filter drops every Makefile save before the builder hears about it.

    Same silence as never scanning the file, and harder to see: the index holds the file and
    simply stops following it.
    """
    from inflorescence.watcher import _DebouncedHandler

    handler = _DebouncedHandler(
        project="p",
        project_manager=None,
        debounce_seconds=0.0,
        exclude_patterns=[],
        loop=None,
        extensions={".py", ".md"},
        filenames={"Makefile", "Dockerfile"},
    )

    assert handler._is_excluded("/repo/Makefile") is False
    assert handler._is_excluded("/repo/deploy/Dockerfile") is False
    assert handler._is_excluded("/repo/app.py") is False
    assert handler._is_excluded("/repo/image.png") is True
