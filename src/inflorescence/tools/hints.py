"""``meta.hints``: what the shape of a page says the caller should do next.

A page can be technically correct and still point the wrong way. Field reports kept describing
the same few shapes, each with a one-parameter fix the caller would have used had it noticed:

* every hit is a weak vector match — "a normal-looking page of 0.45–0.49 scores" read as an
  answer, while ``meta.weak_results: true`` beside it "reads like a formality, not a warning";
* half the page is test code, on a query looking for an implementation;
* half the page is documents, on a question about code (a keyword popular in a repository's own
  design notes buried the one class that answered it);
* most of a mixed page comes from one directory — a mock tree, a fixture dump, four copies of
  an old prototype — while the real answers sit 7th and 8th;
* hits from paths the repository declares archived were moved below live code;
* words of a ``search_text`` query that matched nothing at all, and hits found only by
  tolerating an edit or two in such a word.

Each becomes one sentence naming the evidence and the parameter that changes it. Sentences, not
flags, for the same reason as ``index_coverage.note``: the reader is mid-task, often without
the tool description in context, and the conclusion has to be *in* the text.

Advisory only. Nothing here filters or reorders a page (``archive.demote_archived`` does the
reordering, and says so through here), and ``meta.hints`` is absent when nothing applies.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from inflorescence.code_indexer.models import NodeType

# Below this many hits a page is too small for its composition to mean anything.
_MIN_PAGE_FOR_SHAPE = 4


def _entity(item: Mapping[str, Any]) -> Mapping[str, Any]:
    entity = item.get("entity")
    return entity if isinstance(entity, Mapping) else item


def _literal_prefix(pattern: str) -> str:
    """The directory part of a path filter before its first glob character (``a/b/**`` → ``a/b``)."""
    head = pattern.replace("\\", "/")
    for index, char in enumerate(head):
        if char in "*?[{":
            head = head[:index]
            break
    return head.strip("/").removeprefix("./")


def _dominant_directory(
    items: Sequence[Mapping[str, Any]], include_paths: Sequence[str] | None
) -> tuple[str, int] | None:
    """The deepest directory holding at least half of a mixed page's hits.

    ``None`` when no directory does, when it holds *every* hit (then the page is homogeneous and
    there is no "rest" it buries), or when it is the area the caller asked about — at or above
    one of their ``include_paths``. Nothing here names a directory in advance: whatever tree
    dominates this page (a mock tree, a fixture dump, copies of an old prototype) is read off
    the hits themselves.
    """
    total = len(items)
    counts: dict[str, int] = {}
    for item in items:
        path = str(_entity(item).get("file_path") or item.get("file_path") or "").replace("\\", "/").strip("/")
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            directory = "/".join(parts[:depth])
            counts[directory] = counts.get(directory, 0) + 1
    candidates = [(d, c) for d, c in counts.items() if c * 2 >= total and c < total]
    if not candidates:
        return None
    directory, count = max(candidates, key=lambda pair: (pair[0].count("/"), pair[1]))
    scopes = [_literal_prefix(p) for p in include_paths or ()]
    if any(scope and (scope == directory or scope.startswith(directory + "/")) for scope in scopes):
        return None
    return directory, count


def page_shape_hints(
    items: Sequence[Mapping[str, Any]],
    *,
    include_tests: bool,
    node_types: Sequence[str] | None,
    include_paths: Sequence[str] | None = None,
    documents: bool = True,
) -> list[str]:
    """Hints about what dominates a page: test code, documents, or one directory.

    ``documents=False`` for a tool with no ``node_types`` to offer (``find_text``): a hint must
    name a parameter the caller can actually pass.
    """
    hints: list[str] = []
    total = len(items)
    if total < _MIN_PAGE_FOR_SHAPE:
        return hints
    if include_tests:
        tests = sum(1 for item in items if item.get("is_test"))
        if tests * 2 >= total:
            hints.append(
                f"{tests} of {total} hits are test code (\"is_test\": true). If you are looking for "
                "the implementation, re-run with include_tests=false — a test restates the "
                "vocabulary of the code it exercises and out-ranks it."
            )
    if documents and not node_types:
        documents_count = sum(1 for item in items if _entity(item).get("node_type") == NodeType.DOCUMENT.value)
        if documents_count * 2 >= total:
            hints.append(
                f"{documents_count} of {total} hits are documents (prose, configuration, scripts), not "
                "code. If the question is whether code does this, re-run with "
                "node_types=[\"code\"]; a word that is popular in the repository's own notes "
                "otherwise buries the code that answers it."
            )
    dominant = _dominant_directory(items, include_paths)
    if dominant is not None:
        directory, count = dominant
        hints.append(
            f"{count} of {total} hits come from {directory}/. If that tree is not the answer (mocks, "
            "fixtures, sample data, copies of an old version), re-run with "
            f"exclude_paths=[\"{directory}/\"] to give the page to the rest; if it is the area you "
            f"are asking about, include_paths=[\"{directory}/\"] ranks within it alone."
        )
    return hints


def weak_page_hint(threshold: float) -> str:
    """The sentence that goes with ``meta.weak_results``."""
    return (
        f"Every hit on this page scored below the weak-similarity threshold ({threshold:g}): "
        "nothing matched closely. Do not read the top hit as the answer. The code may not exist "
        "under this scope, or it is described in other words — try search_text with a "
        "distinctive identifier or word, or find_text for a literal."
    )


def archived_hint(count: int) -> str:
    noun = "hit comes" if count == 1 else "hits come"
    return (
        f"{count} {noun} from paths this repository declares history (.inflorescencearchive); "
        "they carry \"archived\": true and are listed after every live hit. Pass exclude_paths "
        "to drop them from this query."
    )


def unmatched_terms_hint(words: Sequence[str], fuzzy: Mapping[str, int]) -> str:
    """Explain words that matched nothing as typed, and what the fuzzy retry did with them."""
    quoted = ", ".join(repr(word) for word in words)
    subject = "word matches" if len(words) == 1 else "words match"
    sentence = (
        f"{quoted}: this {subject} nothing indexed in this project as typed. search_text matches "
        "whole words exactly — no stemming in any language, and an identifier is split at every "
        "non-alphanumeric character and searched as a phrase — so a different inflection, a "
        "typo or a guessed name finds nothing."
    )
    if fuzzy:
        tried = ", ".join(f"{word!r} within {distance} edit{'s' if distance > 1 else ''}"
                          for word, distance in fuzzy.items())
        sentence += (
            f" Retried {tried}: hits found that way carry \"fuzzy\": true and are listed after "
            "exact ones."
        )
    return sentence


def add_hints(payload: MutableMapping[str, Any], hints: Sequence[str]) -> MutableMapping[str, Any]:
    """Append *hints* to ``meta.hints`` (created only when there is something to say)."""
    if not hints:
        return payload
    meta = payload.setdefault("meta", {})
    existing = meta.get("hints")
    meta["hints"] = [*(existing or []), *hints]
    return payload
