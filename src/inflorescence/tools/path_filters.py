"""Glob-style include/exclude path filters for MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch


@dataclass(frozen=True)
class PathFilterConfig:
    include: list[str]
    exclude: list[str]
    include_all: bool = True

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
