"""Glob-style include/exclude path filters, and the test-file path heuristic.

Patterns are matched with :func:`fnmatch.fnmatch` against root-relative, forward-slashed
paths. A pattern that carries no glob metacharacter (``*``, ``?``, ``[``) is a *literal*
path, and a literal path is ambiguous by nature: ``"docs/"`` and ``"docs"`` both read as
"the docs tree" to a caller, yet neither matches ``docs/readme.md`` under plain fnmatch.
Echoing such a pattern back in ``meta.path_filters`` while filtering nothing is the silent
no-op INV-7 forbids, so every literal pattern is normalized *at construction* into the two
forms that cover both readings — the path itself and everything under it — and the
normalized form is what ``meta()`` reports. Patterns that do contain metacharacters are
left exactly as written: the caller spelled out what they meant.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch, fnmatchcase

# The characters that make a pattern a glob rather than a literal path.
_GLOB_METACHARS = ("*", "?", "[")


def _expand_pattern(pattern: str) -> list[str]:
    """Return the match forms of one user pattern (a literal expands to two)."""
    normalized = pattern.replace("\\", "/").strip()
    if not normalized:
        return []
    if any(char in normalized for char in _GLOB_METACHARS):
        return [normalized]
    # A trailing slash only ever meant "directory"; the subtree form carries that meaning.
    stripped = normalized.rstrip("/")
    if not stripped:
        return []
    return [stripped, f"{stripped}/**"]


def normalize_patterns(patterns: list[str]) -> list[str]:
    """Expand literal patterns, preserving order and dropping duplicates."""
    expanded: list[str] = []
    for pattern in patterns:
        for form in _expand_pattern(pattern):
            if form not in expanded:
                expanded.append(form)
    return expanded


@dataclass(frozen=True)
class PathFilterConfig:
    include: list[str]
    exclude: list[str]
    include_all: bool = True

    def __post_init__(self) -> None:
        # Normalizing here, not in ``from_paths``, keeps the invariant true for every
        # construction path — no caller can hold a config whose patterns silently no-op.
        object.__setattr__(self, "include", normalize_patterns(self.include))
        object.__setattr__(self, "exclude", normalize_patterns(self.exclude))

    @classmethod
    def from_paths(
        cls,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
    ) -> PathFilterConfig:
        return cls(
            include=list(include_paths or []),
            exclude=list(exclude_paths or []),
            include_all=include_paths is None,
        )

    def meta(self) -> dict[str, object]:
        """The filters as actually applied — normalized, not as typed."""
        return {
            "include": self.include,
            "exclude": self.exclude,
        }


def path_is_allowed(file_path: str, config: PathFilterConfig) -> bool:
    normalized = file_path.replace("\\", "/").lstrip("./")
    if not config.include_all and not config.include:
        return False
    if not config.include_all and not any(_matches(normalized, pattern) for pattern in config.include):
        return False
    return not any(_matches(normalized, pattern) for pattern in config.exclude)


def filter_records_by_path(records: list[dict[str, object]], config: PathFilterConfig) -> list[dict[str, object]]:
    return [record for record in records if path_is_allowed(str(record.get("file_path", "")), config)]


def _matches(path: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/").lstrip("./")
    if fnmatch(path, normalized):
        return True
    return normalized.startswith("**/") and fnmatch(path, normalized[3:])


# Directory names that mean "test code" in every ecosystem we index, and the filename
# shapes that do. Both are conventions, not facts: this is a *path* heuristic and cannot
# know what a file contains — see ``is_test_path``.
_TEST_DIR_SEGMENTS = frozenset({"tests", "test", "__tests__", "testdata", "spec"})
_TEST_FILE_GLOBS = ("test_*.*", "*_test.*", "conftest.py", "*.spec.*", "*.test.*")


def is_test_path(file_path: str) -> bool:
    """True when a path looks like test code, judged by the path alone.

    Deliberately a heuristic and nothing more: it reads directory segments
    (``tests/``, ``test/``, ``__tests__/``, ``testdata/``, ``spec/``) and filename shapes
    (``test_*.*``, ``*_test.*``, ``conftest.py``, ``*.spec.*``, ``*.test.*``), never the
    file's content. A production file living under ``spec/`` is flagged; a test written
    beside the code it exercises under an unconventional name is not. That is why the
    flag it feeds is advisory (``is_test`` on a hit) and why excluding it is opt-in.
    """
    normalized = file_path.replace("\\", "/").lstrip("./").lower()
    if not normalized:
        return False
    segments = normalized.split("/")
    if any(segment in _TEST_DIR_SEGMENTS for segment in segments):
        return True
    return any(fnmatchcase(segments[-1], glob) for glob in _TEST_FILE_GLOBS)
