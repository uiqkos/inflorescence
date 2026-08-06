"""Progress reporting for indexing.

Two concerns are kept separate:

* Per-batch phase counters (``parsed`` / ``summarized`` / ``embedded`` / ``upserted``
  out of N) are emitted with ``logging`` by the pipeline stages themselves, so they
  land in the log file for every caller — including the MCP ``index_directory`` tool,
  which has no console.
* A live progress bar is driven through the :class:`ProgressReporter` protocol below.
  The CLI ``index`` command passes a :class:`StderrProgress`; the MCP tool and library
  callers pass ``None`` (normalised to :class:`NullProgress`) and get logs only.
"""

from __future__ import annotations

import sys
from typing import Protocol, TextIO, runtime_checkable


@runtime_checkable
class ProgressReporter(Protocol):
    """Receives ``(phase, current, total)`` updates as a stage advances."""

    def update(self, phase: str, current: int, total: int) -> None: ...


class NullProgress:
    """No-op reporter used when no live progress surface is available."""

    def update(self, phase: str, current: int, total: int) -> None:
        return None


class StderrProgress:
    """Dependency-free progress bar: one rewriting line per phase on a stream.

    Avoids adding a runtime dependency (tqdm is not a declared dep). A phase change
    finalises the previous line with a newline; reaching ``total`` closes the line.
    """

    def __init__(self, stream: TextIO | None = None, width: int = 28) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._width = max(1, width)
        self._phase: str | None = None
        self._line_open = False

    def update(self, phase: str, current: int, total: int) -> None:
        if total <= 0:
            return
        if phase != self._phase:
            if self._line_open:
                self._stream.write("\n")
            self._phase = phase
            self._line_open = False
        current = max(0, min(current, total))
        pct = int(current * 100 / total)
        filled = int(self._width * current / total)
        bar = "#" * filled + "-" * (self._width - filled)
        self._stream.write(f"\r{phase:<9} [{bar}] {pct:3d}% ({current}/{total})")
        self._line_open = True
        if current >= total:
            self._stream.write("\n")
            self._line_open = False
        self._stream.flush()


def resolve(progress: ProgressReporter | None) -> ProgressReporter:
    """Return ``progress`` or a shared no-op reporter, so call sites can update unconditionally."""
    return progress if progress is not None else NullProgress()
