"""How a ``search_text`` query is split into words, and where those words sit inside a chunk.

The BM25 indexes behind ``search_text`` tokenize with Tantivy's default tokenizer: text is cut
at every character that is not a letter or a digit, and each piece is lower-cased. There is no
stemming, in any language — ``выбрасывания`` and ``выбрасывание`` are two unrelated terms to the
index, and so are ``run`` and ``running``. A query word that the tokenizer cuts into several
pieces (``parser_strict``, ``task.start``) is searched as a *phrase*: the pieces must be
adjacent, in order.

Everything here mirrors that behaviour, so that what the tools say about a query — which of
its words matched nothing, which lines of a chunk matched — describes what the index actually
did rather than what a reader might assume it did.
"""

from __future__ import annotations

import re

# One run of letters and digits: the unit Tantivy's default tokenizer produces. `\w` would
# also admit "_", which the tokenizer treats as a separator.
_TOKEN = re.compile(r"[^\W_]+")

# How many matching lines one chunk match lists. A long function mentions a common word on
# dozens of lines; the first few locate it, and the chunk span is still reported for the rest.
MAX_LINES_PER_MATCH = 3
# A matching line is quoted, not reproduced: enough to recognise it and to anchor an edit.
MAX_LINE_CHARS = 200


def tokens(text: str) -> list[str]:
    """The index's tokens for *text*: letter/digit runs, lower-cased."""
    return [token.lower() for token in _TOKEN.findall(text)]


def query_words(query: str) -> list[tuple[str, list[str]]]:
    """The whitespace-separated words of a sanitized query, each with its tokens.

    A word with several tokens is one phrase to the index; a word with none (punctuation
    only) is not searched at all and is dropped. Duplicates collapse, first spelling wins.
    """
    words: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for word in query.split():
        parts = tokens(word)
        key = tuple(parts)
        if not parts or key in seen:
            continue
        seen.add(key)
        words.append((word, parts))
    return words


def _contains_phrase(haystack: list[str], phrase: list[str]) -> bool:
    width = len(phrase)
    if width == 1:
        return phrase[0] in haystack
    return any(haystack[i : i + width] == phrase for i in range(len(haystack) - width + 1))


def matching_lines(
    content: str, first_line: int, words: list[tuple[str, list[str]]]
) -> list[dict[str, object]]:
    """The lines of a chunk that contain a query word, as ``{"line", "text"}``, best first.

    *first_line* is the chunk's own starting line in its file, so the numbers are file line
    numbers. A line that holds more of the query's words ranks above one that holds fewer;
    ties keep file order. At most :data:`MAX_LINES_PER_MATCH` are returned.
    """
    scored: list[tuple[int, int, str]] = []
    for offset, line in enumerate(content.splitlines()):
        line_tokens = tokens(line)
        if not line_tokens:
            continue
        hits = sum(1 for _word, phrase in words if _contains_phrase(line_tokens, phrase))
        if hits:
            scored.append((hits, first_line + offset, line.strip()))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"line": number, "text": text[:MAX_LINE_CHARS] + ("…" if len(text) > MAX_LINE_CHARS else "")}
        for _hits, number, text in scored[:MAX_LINES_PER_MATCH]
    ]


def fuzzy_distance_for(word: str) -> int:
    """How many edits a word that matched nothing may be off by, or 0 for "do not try".

    Proportional to length, because an edit is a large share of a short word: one edit away
    from a four-letter word is most of the vocabulary, while one or two edits away from a long
    one is almost always an inflection or a typo of *that* word — a Russian case ending
    (``выбрасывания`` → ``выбрасывание``), a plural (``keepsOrder`` → ``keepsOrder``). Tantivy
    caps the distance at 2.
    """
    parts = tokens(word)
    if len(parts) != 1:
        return 0  # a phrase is matched as adjacent pieces; a fuzzy phrase is not what was asked
    length = len(parts[0])
    if length >= 8:
        return 2
    if length >= 5:
        return 1
    return 0


def within_edits(candidate: str, word: str, distance: int) -> bool:
    """True when *candidate* is at most *distance* Levenshtein edits from *word*.

    Mirrors Tantivy's fuzzy term match closely enough to point at the line a fuzzy hit came
    from; a banded computation, so a long token far from the word costs a few comparisons.
    """
    if abs(len(candidate) - len(word)) > distance:
        return False
    previous = list(range(len(word) + 1))
    for i, char in enumerate(candidate, 1):
        current = [i] + [0] * len(word)
        for j, other in enumerate(word, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other))
        if min(current) > distance:
            return False
        previous = current
    return previous[-1] <= distance


def fuzzy_matching_lines(content: str, first_line: int, word: str, distance: int) -> list[dict[str, object]]:
    """The lines of a chunk holding a token within *distance* edits of *word*."""
    target = word.lower()
    found: list[dict[str, object]] = []
    for offset, line in enumerate(content.splitlines()):
        if any(within_edits(token, target, distance) for token in tokens(line)):
            text = line.strip()
            found.append({
                "line": first_line + offset,
                "text": text[:MAX_LINE_CHARS] + ("…" if len(text) > MAX_LINE_CHARS else ""),
            })
            if len(found) >= MAX_LINES_PER_MATCH:
                break
    return found
