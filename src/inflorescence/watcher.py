"""File watcher with debounced change batching for project updates."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from pathlib import Path

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from inflorescence.config import Settings
from inflorescence.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Collects file-change paths over a debounce window, then triggers an async update."""

    def __init__(
        self,
        project: str,
        project_manager: ProjectManager,
        debounce_seconds: float,
        exclude_patterns: list[str],
        loop: asyncio.AbstractEventLoop,
        extensions: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._project_manager = project_manager
        self._debounce_seconds = debounce_seconds
        self._exclude_patterns = exclude_patterns
        self._extensions = extensions
        self._loop = loop

        self._lock = threading.Lock()
        self._changed_files: set[str] = set()
        self._timer: threading.Timer | None = None

    # ------------------------------------------------------------------
    # watchdog callbacks
    # ------------------------------------------------------------------

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        self._handle(event)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        self._handle(event)

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        self._handle(event)

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        self._handle(event)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, path: str) -> bool:
        """Return True if *path* can never affect the index.

        Filters, as early as possible (INV-6), events that cannot change the index: a suffix no
        parser handles (when an extension set is configured), an exclude pattern matching a
        path *component* (``node_modules``, ``.git``), or a ``*.ext`` glob matching the path
        tail (``*.log``, ``*.min.js``). Editor temp files, logs, and build artifacts otherwise
        debounce into flushes. This is a coarse pre-filter, not the authoritative scan: the
        builder still applies gitignore/globs, and a no-op flush is now cheap regardless, so a
        pattern this misses costs a scan+checksum diff at worst, never a paid embedding.
        """
        p = Path(path)
        if self._extensions is not None and p.suffix not in self._extensions:
            return True
        parts = p.parts
        for pattern in self._exclude_patterns:
            if pattern.startswith("*"):
                if path.endswith(pattern[1:]):
                    return True
            elif pattern in parts:
                return True
        return False

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path: str = str(event.src_path)
        if self._is_excluded(src_path):
            logger.debug("Ignored (excluded) %s event: %s", event.event_type, src_path)
            return

        paths_to_add: list[str] = [src_path]
        if isinstance(event, FileMovedEvent) and event.dest_path:
            dest = str(event.dest_path)
            if not self._is_excluded(dest):
                paths_to_add.append(dest)

        with self._lock:
            for p in paths_to_add:
                self._changed_files.add(p)
            pending = len(self._changed_files)
            self._reset_timer()

        logger.debug(
            "Observed %s event for project=%s: %s (%d path(s) pending flush)",
            event.event_type,
            self._project,
            " -> ".join(paths_to_add),
            pending,
        )

    def _reset_timer(self) -> None:
        """Cancel any pending timer and start a fresh debounce window.

        Must be called while holding ``self._lock``.
        """
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        """Collect accumulated paths and schedule the async update on the event loop.

        The timer thread does not block on the result: updates are serialized by the
        ProjectManager's per-project lock, and blocking here with a timeout used to
        declare long updates failed while they were still running.
        """
        with self._lock:
            if not self._changed_files:
                return
            files = sorted(self._changed_files)
            self._changed_files.clear()
            self._timer = None

        logger.info("Flushing %d changed file(s) for project=%s", len(files), self._project)
        future = asyncio.run_coroutine_threadsafe(
            self._project_manager.update_project(self._project, files),
            self._loop,
        )

        def _log_outcome(fut: concurrent.futures.Future) -> None:
            try:
                result = fut.result()
                logger.info("Project update finished for project=%s: %s", self._project, result)
            except Exception:
                logger.exception("Incremental update failed for project=%s", self._project)

        future.add_done_callback(_log_outcome)


class FileWatcher:
    """Watches project directories for file changes and triggers incremental re-indexing."""

    def __init__(
        self,
        project_manager: ProjectManager,
        settings: Settings,
        extensions: set[str] | None = None,
    ) -> None:
        self._project_manager = project_manager
        self._settings = settings
        self._extensions = extensions
        # project name -> (Observer, handler)
        self._watchers: dict[str, tuple[Observer, _DebouncedHandler]] = {}

    def start(self, project: str, path: str) -> None:
        """Start watching a directory for file changes.

        If a watcher is already running for *project*, it is stopped first.
        """
        if project in self._watchers:
            logger.warning("Watcher already active for project=%s — restarting", project)
            self.stop(project)

        loop = asyncio.get_event_loop()

        handler = _DebouncedHandler(
            project=project,
            project_manager=self._project_manager,
            debounce_seconds=self._settings.watcher_debounce_seconds,
            exclude_patterns=self._settings.exclude_patterns,
            loop=loop,
            extensions=self._extensions,
        )

        observer = Observer()
        observer.schedule(handler, path, recursive=True)
        observer.daemon = True
        observer.start()

        self._watchers[project] = (observer, handler)
        logger.info("Started file watcher for project=%s path=%s", project, path)

    def stop(self, project: str) -> None:
        """Stop watching a project's directory."""
        entry = self._watchers.pop(project, None)
        if entry is None:
            logger.warning("No active watcher for project=%s", project)
            return

        observer, handler = entry

        # Cancel any pending debounce timer so it does not fire after shutdown.
        with handler._lock:
            if handler._timer is not None:
                handler._timer.cancel()
                handler._timer = None

        observer.stop()
        observer.join(timeout=5)
        logger.info("Stopped file watcher for project=%s", project)

    def stop_all(self) -> None:
        """Stop all watchers."""
        projects = list(self._watchers)
        for project in projects:
            self.stop(project)
        logger.info("All file watchers stopped")
