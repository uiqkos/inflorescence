"""What an empty answer is allowed to claim.

Every other honesty signal in this server is defined against something the index *has*:
``stale`` compares a node's stored checksum to its file, ``file_missing`` reports a node
whose file is gone, ``page.truncated`` says a scan stopped early. A file that was never
indexed has no node, so none of them can fire — and the response that comes back is a
confident empty page with ``"checked": 0, "stale": [], "missing": []`` and a watcher line
saying a daemon is following this tree. Field report, 10.09.2026 (names changed): *"Empty
items, zero stale, zero missing, plus a cheerful 'live watch daemon is following this tree'.
Every signal says nothing named wave_queue exists in this project. The truth was that a
174-line Python module named wave_queue.py was sitting in scripts/ right now."* The agent
nearly wrote a file that already existed.

So an empty first page is answered with evidence instead of silence: the index's own file
count, and a bounded on-disk check (:meth:`GraphBuilder.scan_uncovered`) of the directories
it already knows. Both outcomes are worth saying, and both are said here —

* files exist on disk that the index does not hold → the empty page is a coverage gap, and
  the block names the files and how to fix it;
* nothing is missing under that scope → the empty page is real evidence of absence, which
  is exactly the conclusion an agent could not safely draw before.

The check is deliberately not exhaustive, and says where it stops rather than implying it
went everywhere: it opens only directories the index already knows (a brand-new *directory*
is invisible to it), and it is capped in directories and in wall-clock. An honest partial
check reads as a partial check; that is the whole point of the block.

Only on an empty first page: a page with results carries its own evidence, and a page past
the first is a pagination artifact, not a claim about the project.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from inflorescence.project_manager import ProjectManager
from inflorescence.tools.freshness import FreshnessChecker

logger = logging.getLogger(__name__)


def _scope_patterns(meta: Mapping[str, Any]) -> list[str]:
    """The include patterns the tool already echoed in ``meta.path_filters``, if any.

    Reusing the echoed form rather than the raw argument keeps the coverage check and the
    filter that produced the empty page talking about the same paths — including the literal
    expansion ``docs/`` -> ``docs`` + ``docs/**`` that PathFilterConfig applies.
    """
    filters = meta.get("path_filters")
    if not isinstance(filters, Mapping):
        return []
    include = filters.get("include")
    if isinstance(include, Sequence) and not isinstance(include, str | bytes):
        return [str(pattern) for pattern in include]
    return []


def _note(report: Mapping[str, Any], directory: str) -> str:
    """One sentence a reader can act on, whichever way the check came out.

    Written as prose rather than as counts because it is read mid-task by something deciding
    what to do next, often without the tool description in context: the conclusion the
    numbers license has to be *in* the sentence, along with the command that closes the gap.
    """
    missing = int(report.get("not_indexed_total") or 0)
    examples = list(report.get("not_indexed") or [])
    indexed = int(report.get("files_indexed") or 0)
    checked = int(report.get("dirs_checked") or 0)

    if missing:
        shown = ", ".join(str(path) for path in examples)
        more = f" (the first {len(examples)} of {missing})" if missing > len(examples) else ""
        subject = "1 file is" if missing == 1 else f"{missing} files are"
        them = "it" if missing == 1 else "them"
        return (
            f"An empty result here is NOT evidence of absence: {subject} on disk in the "
            f"directories checked and absent from this index, so nothing about {them} is "
            f"searchable — {shown}{more}. Read {them} from the filesystem, and run "
            f"index_directory({directory!r}) to make {them} queryable."
        )
    tail = (
        " The check stopped at its bound before covering every directory it knows, so it is "
        "not exhaustive."
        if report.get("capped")
        else ""
    )
    directories = "1 directory" if checked == 1 else f"{checked} directories"
    return (
        f"This index holds {indexed} files, and every file in the {directories} checked just "
        f"now is one of them — so an empty result here is evidence of absence rather than a "
        f"coverage gap. Files in directories the index has never seen are outside this "
        f"check.{tail}"
    )


def _block(report: Mapping[str, Any], directory: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "files_indexed": report.get("files_indexed", 0),
        "dirs_checked": report.get("dirs_checked", 0),
        "note": _note(report, directory),
    }
    missing = int(report.get("not_indexed_total") or 0)
    if missing:
        # Present only when it fired: an absent list is the "nothing missing" case, which the
        # note states in words, and a permanent `"not_indexed": []` would read as noise on
        # every empty page.
        block["not_indexed"] = list(report.get("not_indexed") or [])
        block["not_indexed_total"] = missing
    if report.get("capped"):
        block["partial"] = True
    return block


async def note_coverage(
    payload: MutableMapping[str, Any],
    manager: ProjectManager,
    directory: str,
    checker: FreshnessChecker | None = None,
    indexed_paths: Sequence[str] | None = None,
) -> MutableMapping[str, Any]:
    """Add ``meta.index_coverage`` when *payload* is an empty first page; return it either way.

    Pass the ``FreshnessChecker`` the response was stamped with — it already holds the
    project's stored checksums, so the file inventory costs nothing extra — or an explicit
    *indexed_paths* when there is no checker.

    Never raises: this annotates an answer the caller already has, and a failed coverage
    check must not turn a working empty search into an error. It is silently skipped (with a
    debug log) instead, which is safe because the block's presence is what carries meaning —
    nothing else reads it.
    """
    items = payload.get("items")
    if items:
        return payload
    page = payload.get("page")
    if isinstance(page, Mapping) and int(page.get("offset") or 0) > 0:
        return payload  # past the end of a real result set, not a claim about the project

    meta = payload.setdefault("meta", {})
    scope = _scope_patterns(meta if isinstance(meta, Mapping) else {})
    paths = list(indexed_paths) if indexed_paths is not None else sorted(checker.indexed_files) if checker else []
    try:
        report = dict(await manager.scan_uncovered(directory, paths, scope))
    except Exception:  # noqa: BLE001 — an annotation must never break the answer it annotates
        logger.debug("Coverage check failed for %s", directory, exc_info=True)
        return payload
    meta["index_coverage"] = _block(report, directory)
    return payload
