"""``find_text``: every line holding a literal or a regex, read from the files themselves.

The search tools rank. That is the right answer to "where is the code that does X", and the
wrong one to the questions field reports kept bringing to them anyway: "every use of the
design token ``--color-brand`` before I rename it", "is ``parser_strict`` referenced anywhere
under ``frontend/``", "which line defines ``WINDOW_SIZE``". Those need *all* occurrences and an
exact count, and a ranked top-k over a tokenized index gives neither: ``search_text`` splits
``--color-brand`` into three words and matches no partial word, and a page of 20 is not a
census. Agents fell back to ``grep -rn`` — which answers, but knows nothing about the project:
not which files are indexed, not which entity a line belongs to, not which trees are tests or
history.

So this reads the files — the index's own file inventory, plus files the indexer would admit
that are not in the index yet — line by line, and returns every matching line in path order
with an exact total, each one tied to the innermost indexed entity whose span holds it.
Nothing is ranked and nothing is embedded: the answer is as exhaustive as the file list, and
where it is not (a read budget reached, a directory the index has never seen) it says so.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

# ``regex`` rather than ``re``: its matching takes a timeout. A pattern with catastrophic
# backtracking (``(x+x+)+y`` on a long line) holds ``re.search`` for minutes inside one call, in a
# worker thread nothing can cancel — the time budget between files never gets a turn.
import regex
from mcp.server.fastmcp import FastMCP

from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.tools import descriptions as doc
from inflorescence.tools.archive import flag_archived, load_archive
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import build_checker
from inflorescence.tools.gates import ensure_indexed
from inflorescence.tools.hints import add_hints, page_shape_hints
from inflorescence.tools.path_filters import PathFilterConfig, is_test_path, path_is_allowed
from inflorescence.tools.responses import clamp_limit, clamp_offset, error_response, paginated_response

MAX_FIND_LIMIT = 500
# A matching line is quoted up to this many characters, cut around the match: a minified
# bundle's one 200 KB line must not become the page.
FIND_MAX_LINE_CHARS = 240
# The read budget of one call. A project of a few thousand source files is tens of MB; the
# budget exists for the tree that is not, and reaching it is reported, never silent.
FIND_MAX_BYTES = 256 * 1024 * 1024
FIND_TIME_BUDGET_S = 20.0
# Files the indexer would admit but the index does not hold yet, read in addition to the
# indexed ones (and flagged ``"indexed": false``). A tree with more unindexed files than this
# wants ``index_directory``, and the answer says so.
FIND_MAX_UNINDEXED = 500
_BINARY_SNIFF_BYTES = 8000


@dataclass
class _Scan:
    """What one pass over the file list found."""

    kept: list[tuple[str, int, str]] = field(default_factory=list)
    total_matches: int = 0
    files_matched: int = 0
    files_searched: int = 0
    binary_skipped: int = 0
    unreadable: list[str] = field(default_factory=list)
    not_searched: int = 0
    # The file in which the pattern used up the time budget inside a single match attempt:
    # the regex itself is too expensive (catastrophic backtracking), not the tree too big.
    regex_timed_out_in: str | None = None


def _quote(line: str, start: int) -> str:
    """The matching line, cut to ``FIND_MAX_LINE_CHARS`` around the match when it is longer."""
    text = line.rstrip("\r\n")
    if len(text) <= FIND_MAX_LINE_CHARS:
        return text
    lead = FIND_MAX_LINE_CHARS // 4
    begin = max(0, min(start - lead, len(text) - FIND_MAX_LINE_CHARS))
    piece = text[begin:begin + FIND_MAX_LINE_CHARS]
    return ("…" if begin else "") + piece + ("…" if begin + FIND_MAX_LINE_CHARS < len(text) else "")


def _lines(text: str) -> list[str]:
    """Lines split at ``\n`` only, numbered as grep and editors number them.

    ``str.splitlines`` also breaks at form feed, ``\x1c``–``\x1e``, ``\x85``, U+2028/9 — a
    file holding one of those would report every later match on the wrong line.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _scan_files(
    root: Path,
    paths: Sequence[str],
    compiled: regex.Pattern[str],
    keep: int,
    byte_budget: int = FIND_MAX_BYTES,
    time_budget: float = FIND_TIME_BUDGET_S,
) -> _Scan:
    """Read *paths* in order and collect matching lines; totals are exact for what was read.

    Only the first *keep* matches are stored (a page plus one); every match is counted.
    Stops at the byte or time budget and counts what it did not read.
    """
    scan = _Scan()
    deadline = monotonic() + time_budget
    remaining = byte_budget
    for index, rel in enumerate(paths):
        path = root / rel
        try:
            size = path.stat().st_size
        except OSError:
            scan.unreadable.append(rel)
            continue
        if monotonic() > deadline or size > remaining:
            scan.not_searched = len(paths) - index
            break
        try:
            data = path.read_bytes()
        except OSError:
            scan.unreadable.append(rel)
            continue
        remaining -= len(data)
        if b"\0" in data[:_BINARY_SNIFF_BYTES]:
            scan.binary_skipped += 1
            continue
        scan.files_searched += 1
        matched_here = False
        try:
            for number, line in enumerate(_lines(data.decode("utf-8", errors="replace")), start=1):
                match = compiled.search(line, timeout=max(deadline - monotonic(), 0.001))
                if match is None:
                    continue
                matched_here = True
                scan.total_matches += 1
                if len(scan.kept) < keep:
                    scan.kept.append((rel, number, _quote(line, match.start())))
        except TimeoutError:
            scan.regex_timed_out_in = rel
            scan.not_searched = len(paths) - index
            break
        finally:
            if matched_here:
                scan.files_matched += 1
    return scan


def _owner(entities: Sequence[dict[str, object]], line: int) -> dict[str, Any] | None:
    """The innermost entity whose span holds *line* (a method over its class over its module)."""
    best: dict[str, object] | None = None
    best_span = 0
    for entity in entities:
        start = entity.get("line_start")
        end = entity.get("line_end") or start
        if not isinstance(start, int) or not isinstance(end, int) or not start <= line <= end:
            continue
        span = end - start
        if best is None or span < best_span:
            best, best_span = entity, span
    if best is None:
        return None
    return {key: best[key] for key in ("id", "name", "node_type", "line_start", "line_end") if best.get(key) is not None}


def _compile(pattern: str, is_regex: bool, case_sensitive: bool) -> regex.Pattern[str] | dict:
    if not pattern:
        return error_response("invalid_argument", "pattern is empty — pass the literal (or regex) to find.")
    flags = 0 if case_sensitive else regex.IGNORECASE
    try:
        return regex.compile(pattern if is_regex else regex.escape(pattern), flags)
    except regex.error as exc:
        return error_response(
            "invalid_regex",
            f"{pattern!r} is not a valid Python regular expression ({exc}). Pass regex=false to "
            "search for it literally.",
        )


FIND_TEXT_DESCRIPTION = doc.join(
    f"""Every line that contains a literal (or matches a regex) — exhaustive, unranked, read from the files on disk.

Use it when the answer must be COMPLETE rather than ranked: every use of a name before a
rename, "is this referenced anywhere under X", the line that defines a constant, all
occurrences of a CSS token, an error string, a config key. ``search_text`` ranks whole words
from the index and misses partial words and punctuation-heavy literals (``--color-brand``,
``parser_strict(``, ``/api/v2/``); this does neither — it reads each file and tests each line.

``pattern`` is a literal by default (every character means itself); ``regex=true`` makes it a
Python regular expression, matched line by line (``^``/``$`` anchor a line); a regex that
backtracks catastrophically is stopped at the time budget and named in
``meta.regex_timed_out_in``. ``case_sensitive`` defaults to true. Each item is ``{{file_path, line, text, entity}}``: ``text`` is the line (cut
to {FIND_MAX_LINE_CHARS} characters around the match when longer) and ``entity`` the innermost
indexed entity whose span holds it (a method over its class over its module; ``variable`` for a
line inside a top-level constant), so the next call can be ``get_entity_context`` on it. Items
are in path and line order; ``meta.total_matches`` and ``meta.files_matched`` are exact counts
over everything read, whatever the page size.

WHAT IS READ: every file this index holds (code and documents), plus up to {FIND_MAX_UNINDEXED}
files on disk that the indexer would admit but the index does not hold yet — created since the
last index run — whose items carry ``"indexed": false`` and no ``entity``. Directories are
found through the index, so a directory the index has never seen is not opened (the coverage
block of an empty page says so). A file that changed since indexing is read as it is now, and
its items carry ``"stale": true``: the text is current, the entity's line span may be older.
Binary files are skipped. The read stops at {FIND_MAX_BYTES // (1024 * 1024)} MB or
{FIND_TIME_BUDGET_S:g} s; if it stops, ``page.truncated`` is true, ``meta.files_not_searched``
counts what was left, and a sentence in ``meta.hints`` says so — narrow with
``include_paths``. Costs nothing: no model, no embedding.""",
    """Scope it like a search: ``include_paths`` for the area you mean, ``exclude_paths`` for
mocks, fixtures, copies of an old version, and ``include_tests=false`` unless the tests are part
of the answer (for a rename they are). Example: ``find_text(directory="/repo",
pattern="--color-brand", include_paths=["frontend/src/"])``.""",
    doc.PATH_FILTERS,
    doc.LISTING_FLAGS,
    doc.WORKTREE_FIRST_QUERY,
    doc.PROVENANCE,
    doc.EMPTY_PAGE,
)


def register_find_tools(mcp: FastMCP, manager: ProjectManager, repo: GraphRepository) -> None:
    """Register ``find_text``."""

    @mcp.tool(description=FIND_TEXT_DESCRIPTION)
    async def find_text(
        directory: str,
        pattern: str,
        regex: bool = False,
        case_sensitive: bool = True,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Exhaustive line search over the project's files (published description: FIND_TEXT_DESCRIPTION)."""
        compiled = _compile(pattern, regex, case_sensitive)
        if isinstance(compiled, dict):
            return compiled
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit, MAX_FIND_LIMIT)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        checker = await build_checker(repo, project, directory)
        root = Path(directory).resolve()

        def wanted(path: str) -> bool:
            return path_is_allowed(path, filters) and (include_tests or not is_test_path(path))

        indexed = sorted(path for path in checker.indexed_files if wanted(path))
        gap = await manager.scan_uncovered(
            directory, checker.indexed_files, include_paths, max_examples=FIND_MAX_UNINDEXED
        )
        unindexed = sorted(path for path in gap.get("not_indexed") or [] if wanted(str(path)))
        unindexed_set = set(unindexed)
        universe = sorted([*indexed, *unindexed])

        scan = await asyncio.to_thread(
            _scan_files, root, universe, compiled, bounded_offset + bounded_limit + 1
        )
        window = scan.kept[bounded_offset:]
        on_page = window[:bounded_limit]
        page_files = sorted({path for path, _, _ in on_page if path not in unindexed_set})
        entities_by_file: dict[str, list[dict[str, object]]] = {}
        for entity in await repo.get_file_entities(project, page_files):
            entities_by_file.setdefault(str(entity.get("file_path")), []).append(entity)

        items: list[dict[str, Any]] = []
        for path, line, text in window:
            item: dict[str, Any] = {"file_path": path, "line": line, "text": text}
            if path in unindexed_set:
                item["indexed"] = False
            else:
                owner = _owner(entities_by_file.get(path, []), line)
                if owner is not None:
                    item["entity"] = owner
            if is_test_path(path):
                item["is_test"] = True
            items.append(item)

        truncated = scan.not_searched > 0
        meta: dict[str, Any] = {
            "project": project,
            "pattern": pattern,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "path_filters": filters.meta(),
            "include_tests": include_tests,
            "files_searched": scan.files_searched,
            "files_matched": scan.files_matched,
            "total_matches": scan.total_matches,
        }
        if unindexed:
            meta["unindexed_files_searched"] = len(unindexed)
        if scan.unreadable:
            meta["unreadable"] = scan.unreadable[:20]
        if truncated:
            meta["files_not_searched"] = scan.not_searched
        if scan.regex_timed_out_in is not None:
            meta["regex_timed_out_in"] = scan.regex_timed_out_in
        payload = paginated_response(items, bounded_limit, bounded_offset, meta, truncated=truncated)
        flag_archived(payload["items"], load_archive(root))
        for item in payload["items"]:
            checker.annotate(item)

        hints: list[str] = []
        if scan.regex_timed_out_in is not None:
            hints.append(
                f"The regex used up the whole {FIND_TIME_BUDGET_S:g} s budget inside a single line of "
                f"{scan.regex_timed_out_in}: it backtracks catastrophically (nested quantifiers such as "
                "(a+)+ or (x+x+)+). The totals cover only the files before it. Simplify the pattern, "
                "or pass the literal with regex=false."
            )
        elif truncated:
            hints.append(
                f"The read budget ({FIND_MAX_BYTES // (1024 * 1024)} MB / {FIND_TIME_BUDGET_S:g} s) "
                f"ran out with {scan.not_searched} of {len(universe)} files unread, so the totals "
                "cover only what was read. Narrow with include_paths to make the answer complete."
            )
        gap_total = int(gap.get("not_indexed_total") or 0)
        if gap.get("capped") or gap_total > FIND_MAX_UNINDEXED:
            hints.append(
                f"{gap_total} files on disk are not in this index"
                f"{' (the check stopped early)' if gap.get('capped') else ''}, and at most "
                f"{FIND_MAX_UNINDEXED} of them were read. Run index_directory to bring the rest "
                "in, or narrow with include_paths."
            )
        if bounded_offset == 0:
            hints.extend(page_shape_hints(
                payload["items"], include_tests=include_tests, node_types=None,
                include_paths=include_paths, documents=False,
            ))
        add_hints(payload, hints)
        return await note_coverage(checker.stamp(payload), manager, directory, checker)
