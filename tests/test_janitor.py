"""The worktree janitor — the only code path in the system that deletes a whole project.

An index costs real money (LLM summaries + embeddings), so every test here is written from
the deleter's point of view: what has to be true for a graph to be destroyed, and — far more
important — every way in which the janitor must decline. The five defense layers named in
:mod:`inflorescence.janitor` are exercised one at a time, so a regression in any single one
fails a test of its own instead of hiding behind the other four.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import pytest

from inflorescence import janitor
from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeJanitorRepository:
    """The janitor's slice of :class:`GraphRepository`, with the real guard reproduced.

    ``delete_worktree_project`` mirrors the production wrapper (re-read the marker, refuse
    when absent) so a janitor bug that reaches the delete call on an unmarked project is
    caught here too; the wrapper itself is tested against the real repository below.
    ``delete_project`` is present and explodes: the janitor must never call the unguarded
    deleter.
    """

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.list_calls = 0
        self.deleted: list[str] = []
        self.refused: list[str] = []
        self.meta_reads: list[str] = []
        self.fail_meta_for: set[str] = set()

    async def list_projects(self) -> list[dict[str, object]]:
        self.list_calls += 1
        return [dict(row) for row in self.rows]

    def _row(self, project: str) -> dict[str, object] | None:
        for row in self.rows:
            if row["project"] == project:
                return row
        return None

    async def get_project_worktree_meta(self, project: str) -> dict[str, str] | None:
        self.meta_reads.append(project)
        if project in self.fail_meta_for:
            raise RuntimeError(f"boom for {project}")
        row = self._row(project)
        if row is None:
            return None
        worktree_of = row.get("worktree_of")
        main_root_path = row.get("main_root_path")
        if not worktree_of or not main_root_path:
            return None
        return {"worktree_of": str(worktree_of), "main_root_path": str(main_root_path)}

    async def delete_worktree_project(self, project: str) -> bool:
        if await self.get_project_worktree_meta(project) is None:
            self.refused.append(project)
            return False
        self.deleted.append(project)
        self.rows = [row for row in self.rows if row["project"] != project]
        return True

    async def delete_project(self, project: str) -> None:  # pragma: no cover - must not run
        raise AssertionError("janitor called the unguarded delete_project")


class FakeConnection:
    """Records queries; returns canned rows per query fragment."""

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.reads: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []
        self.write_rows: list[dict[str, object]] = []

    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        self.reads.append((query, dict(params or {})))
        return list(self.rows)

    async def execute_write(self, query: str, params: dict | None = None) -> None:
        self.writes.append((query, dict(params or {})))

    async def execute_write_query(self, query: str, params: dict | None = None) -> list[dict]:
        self.writes.append((query, dict(params or {})))
        return list(self.write_rows)


def settings_with(**kwargs: object) -> Settings:
    return Settings(**kwargs)


def make_git_dir(root: Path) -> Path:
    """A directory that passes the "donor looks like a checkout" test, without git."""
    (root / ".git").mkdir(parents=True, exist_ok=True)
    return root


def register_worktree(main_root: Path, name: str, worktree_path: Path) -> None:
    """Hand-write the registry entry git keeps for a linked worktree.

    ``worktree.registered_worktree_paths`` only reads files, so a hand-built registry is
    indistinguishable from git's own — and lets a test place an entry for a path that never
    existed, which is exactly the state ``rm -rf`` leaves behind.
    """
    entry = main_root / ".git" / "worktrees" / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "gitdir").write_text(f"{worktree_path}/.git\n", encoding="utf-8")


def row(
    project: str,
    root_path: Path | str,
    *,
    worktree_of: str | None = None,
    main_root_path: Path | str | None = None,
) -> dict[str, object]:
    return {
        "project": project,
        "root_path": str(root_path),
        "worktree_of": worktree_of,
        "main_root_path": str(main_root_path) if main_root_path is not None else None,
    }


# --------------------------------------------------------------------------------------
# Layer 1 — only a marked project is ever a candidate
# --------------------------------------------------------------------------------------


async def test_layer1_project_without_the_marker_is_never_a_candidate(tmp_path: Path) -> None:
    repo = FakeJanitorRepository([row("proj_a", tmp_path / "gone")])

    report = await sweep(repo)

    assert report["scanned"] == 1
    assert report["candidates"] == 0
    assert report["deleted"] == []
    assert repo.deleted == []


async def test_layer1_half_written_marker_is_not_a_candidate(tmp_path: Path) -> None:
    """Donor id without its path (or vice versa) means "unknown", never "delete"."""
    repo = FakeJanitorRepository(
        [
            row("proj_a", tmp_path / "gone_a", worktree_of="main_1"),
            row("proj_b", tmp_path / "gone_b", main_root_path=tmp_path / "main"),
        ]
    )

    report = await sweep(repo)

    assert report["candidates"] == 0
    assert repo.deleted == []


async def test_unmounted_volume_never_costs_an_ordinary_project_its_index(tmp_path: Path) -> None:
    """The disaster scenario: a detached disk makes every root_path vanish at once.

    Ordinary projects carry no marker, so no number of sweeps may touch them — this is the
    single most expensive failure the janitor could produce, so it is swept repeatedly.
    """
    rows = [row(f"proj_{i}", f"/Volumes/external/repo{i}") for i in range(20)]
    repo = FakeJanitorRepository(rows)

    for _ in range(5):
        report = await sweep(repo)
        assert report["deleted"] == []

    assert repo.deleted == []
    assert len(repo.rows) == 20


# --------------------------------------------------------------------------------------
# Layer 2 — the candidate's own directory must be gone
# --------------------------------------------------------------------------------------


async def test_layer2_worktree_still_on_disk_is_kept(tmp_path: Path) -> None:
    main = make_git_dir(tmp_path / "main")
    live = tmp_path / "wt"
    live.mkdir()
    repo = FakeJanitorRepository(
        [row("wt_1", live, worktree_of="main_1", main_root_path=main)]
    )

    report = await sweep(repo)

    assert report["candidates"] == 1
    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_ON_DISK}]


# --------------------------------------------------------------------------------------
# Layer 3 — the donor must still be there and look like a repository
# --------------------------------------------------------------------------------------


async def test_layer3_missing_donor_is_skipped_with_a_warning(tmp_path: Path, caplog) -> None:
    """Both paths gone is indistinguishable from an unmounted volume — keep the index."""
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=tmp_path / "main")]
    )

    with caplog.at_level(logging.WARNING, logger="inflorescence.janitor"):
        report = await sweep(repo)

    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_DONOR_MISSING}]
    assert any("wt_1" in rec.getMessage() for rec in caplog.records)


async def test_layer3_donor_directory_without_git_is_skipped(tmp_path: Path) -> None:
    """A path that exists but is not a checkout proves nothing about the worktree."""
    main = tmp_path / "main"
    main.mkdir()  # no .git — could be a stale mountpoint or a re-created empty directory
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )

    report = await sweep(repo)

    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_DONOR_MISSING}]


# --------------------------------------------------------------------------------------
# Layer 4 — git's registry has the last word
# --------------------------------------------------------------------------------------


async def test_layer4_registered_worktree_that_is_alive_is_kept(tmp_path: Path) -> None:
    """A stored path that fails ``is_dir`` while the worktree itself is alive.

    Layer 2 tests the path as recorded; that string can fail to resolve for reasons other
    than "the worktree is gone" (an intermediate component removed, a permission blip on a
    parent, a mount in flux). Here the recorded path traverses a directory that does not
    exist, yet resolves onto a live, still-registered worktree — layer 4 is what stops the
    deletion.
    """
    main = make_git_dir(tmp_path / "main")
    live = tmp_path / "wt"
    live.mkdir()
    register_worktree(main, "wt", live)
    recorded = tmp_path / "ghost" / ".." / "wt"
    assert not recorded.is_dir()

    repo = FakeJanitorRepository(
        [row("wt_1", recorded, worktree_of="main_1", main_root_path=main)]
    )

    report = await sweep(repo)

    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_STILL_REGISTERED}]


async def test_layer4_unregistered_and_absent_worktree_is_deleted(tmp_path: Path) -> None:
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )

    report = await sweep(repo)

    assert report["deleted"] == ["wt_1"]
    assert repo.deleted == ["wt_1"]
    assert report["skipped"] == []


async def test_layer4_rm_rf_without_prune_is_still_deleted(tmp_path: Path) -> None:
    """``rm -rf`` leaves git's registry entry in place until ``git worktree prune``.

    The registration is live but the content is dead, and re-creating the worktree re-indexes
    almost free from the donor's cache, so this is the janitor's main job — proved against a
    registry that real git wrote.
    """
    main = tmp_path / "main"
    _git_init(main)
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", str(wt))
    assert (main / ".git" / "worktrees").is_dir()
    _rmtree(wt)
    assert not wt.exists()

    repo = FakeJanitorRepository([row("wt_1", wt, worktree_of="main_1", main_root_path=main)])

    report = await sweep(repo)

    assert report["deleted"] == ["wt_1"]


async def test_live_git_worktree_is_kept(tmp_path: Path) -> None:
    """The same real-git setup, without the ``rm -rf`` — nothing may be deleted."""
    main = tmp_path / "main"
    _git_init(main)
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", str(wt))

    repo = FakeJanitorRepository([row("wt_1", wt, worktree_of="main_1", main_root_path=main)])

    report = await sweep(repo)

    assert report["deleted"] == []


# --------------------------------------------------------------------------------------
# Layer 5 — the guarded repository wrapper, tested on its own
# --------------------------------------------------------------------------------------


async def test_layer5_wrapper_refuses_a_project_without_the_marker() -> None:
    """Called directly, as a janitor bug would: no marker, no delete, no write at all."""
    conn = FakeConnection(rows=[{"worktree_of": None, "main_root_path": None}])
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    deleted = await repo.delete_worktree_project("proj_a")

    assert deleted is False
    assert conn.writes == []


async def test_layer5_wrapper_refuses_when_the_root_node_is_absent() -> None:
    conn = FakeConnection(rows=[])
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    assert await repo.delete_worktree_project("proj_a") is False
    assert conn.writes == []


async def test_layer5_wrapper_deletes_a_marked_project_with_a_guarded_query() -> None:
    conn = FakeConnection(rows=[{"worktree_of": "main_1", "main_root_path": "/repo"}])
    conn.write_rows = [{"deleted": 12}]
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    assert await repo.delete_worktree_project("wt_1") is True
    assert len(conn.writes) == 1
    query, params = conn.writes[0]
    assert params["project"] == "wt_1"
    # The delete itself re-asserts the marker in the same transaction, so a marker removed
    # between the read above and this write cannot be raced into a deletion.
    assert "worktree_of" in query
    assert "DETACH DELETE" in query


async def test_layer5_wrapper_reports_false_when_the_guarded_query_matches_nothing() -> None:
    """Marker read succeeded, then the guard matched nothing (raced marker removal)."""
    conn = FakeConnection(rows=[{"worktree_of": "main_1", "main_root_path": "/repo"}])
    conn.write_rows = []
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    assert await repo.delete_worktree_project("wt_1") is False


async def test_sweep_never_calls_the_unguarded_delete(tmp_path: Path) -> None:
    """FakeJanitorRepository.delete_project raises; a deletion still has to succeed."""
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )

    report = await sweep(repo)

    assert report["deleted"] == ["wt_1"]


async def test_a_refusing_wrapper_is_reported_as_a_skip(tmp_path: Path) -> None:
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )

    async def refuse(project: str) -> bool:
        return False

    repo.delete_worktree_project = refuse  # type: ignore[assignment]

    report = await sweep(repo)

    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_REFUSED}]


# --------------------------------------------------------------------------------------
# Lease, logging, isolation, off switch
# --------------------------------------------------------------------------------------


async def test_a_held_lease_skips_the_project_without_waiting(tmp_path: Path) -> None:
    """An index may be running against this very project — never delete under its feet.

    Uses the real :class:`ProjectLease` against a connection that reports the lease held by
    somebody else, so the no-wait behaviour of the janitor's lease is what is measured, not
    a stand-in.
    """
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )
    conn = FakeConnection()
    conn.write_rows = [{"acquired": False, "holder": "other-host:1:abc"}]

    started = time.monotonic()
    report = await sweep(repo, conn=conn)
    elapsed = time.monotonic() - started

    assert report["deleted"] == []
    assert report["skipped"] == [{"project": "wt_1", "reason": janitor.REASON_LEASE_HELD}]
    assert elapsed < 1.0  # did not poll or wait out any timeout


async def test_a_free_lease_is_taken_and_released_around_the_delete(tmp_path: Path) -> None:
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )
    conn = FakeConnection()
    conn.write_rows = [{"acquired": True, "holder": "me"}]

    report = await sweep(repo, conn=conn)

    assert report["deleted"] == ["wt_1"]
    # acquire + release both went through the lease node.
    assert len(conn.writes) == 2
    assert "IndexLock" in conn.writes[0][0]
    assert "IndexLock" in conn.writes[1][0]


async def test_every_deletion_is_logged_at_warning_with_paths(tmp_path: Path, caplog) -> None:
    """A deletion is irreversible and expensive; it must be visible in the log."""
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_17", main_root_path=main)]
    )

    with caplog.at_level(logging.WARNING, logger="inflorescence.janitor"):
        await sweep(repo)

    messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(messages) == 1
    assert "wt_1" in messages[0]
    assert str(tmp_path / "wt") in messages[0]
    assert "main_17" in messages[0]
    assert str(main) in messages[0]


async def test_one_broken_project_does_not_stop_the_sweep(tmp_path: Path) -> None:
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [
            row("wt_bad", tmp_path / "wt_bad", worktree_of="main_1", main_root_path=main),
            row("wt_good", tmp_path / "wt_good", worktree_of="main_1", main_root_path=main),
        ]
    )
    repo.fail_meta_for = {"wt_bad"}

    report = await sweep(repo)

    assert report["deleted"] == ["wt_good"]
    assert report["skipped"] == [{"project": "wt_bad", "reason": janitor.REASON_ERROR}]


async def test_a_failing_listing_is_reported_not_raised(tmp_path: Path) -> None:
    class BrokenRepo(FakeJanitorRepository):
        async def list_projects(self) -> list[dict[str, object]]:
            raise RuntimeError("memgraph is down")

    report = await janitor.sweep(BrokenRepo([]), settings_with(worktree_janitor_enabled=True))

    assert report == {"scanned": 0, "candidates": 0, "deleted": [], "skipped": []}


async def test_disabled_flag_makes_the_sweep_a_no_op(tmp_path: Path) -> None:
    """Second belt: even if something calls sweep, the off switch stops it dead."""
    main = make_git_dir(tmp_path / "main")
    repo = FakeJanitorRepository(
        [row("wt_1", tmp_path / "wt", worktree_of="main_1", main_root_path=main)]
    )

    report = await janitor.sweep(repo, settings_with(worktree_janitor_enabled=False))

    assert report == {"scanned": 0, "candidates": 0, "deleted": [], "skipped": []}
    assert repo.list_calls == 0
    assert repo.deleted == []


async def test_malformed_rows_are_ignored(tmp_path: Path) -> None:
    """A row with no project id or no root_path is data we cannot act on — never a candidate."""
    repo = FakeJanitorRepository(
        [
            {"project": "", "root_path": "/gone", "worktree_of": "m", "main_root_path": "/m"},
            {"project": "wt_1", "root_path": "", "worktree_of": "m", "main_root_path": "/m"},
        ]
    )

    report = await sweep(repo)

    assert report["candidates"] == 0
    assert repo.deleted == []


# --------------------------------------------------------------------------------------
# Repository listing
# --------------------------------------------------------------------------------------


async def test_list_projects_returns_marker_aware_rows() -> None:
    conn = FakeConnection(
        rows=[
            {"project": "a_1", "root_path": "/a", "worktree_of": None, "main_root_path": None},
            {"project": "b_1", "root_path": "/b", "worktree_of": "a_1", "main_root_path": "/a"},
        ]
    )
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    projects = await repo.list_projects()

    assert projects == [
        {"project": "a_1", "root_path": "/a", "worktree_of": None, "main_root_path": None},
        {"project": "b_1", "root_path": "/b", "worktree_of": "a_1", "main_root_path": "/a"},
    ]
    query, _ = conn.reads[0]
    assert "dir:." in query
    assert conn.writes == []


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


async def sweep(repo: object, *, conn: object | None = None) -> dict[str, object]:
    """Sweep with the janitor enabled — the default for every test that does not say otherwise."""
    return await janitor.sweep(repo, settings_with(worktree_janitor_enabled=True), conn=conn)


_GIT_PRELUDE = [
    "-c", "user.email=tests@inflorescence.invalid",
    "-c", "user.name=Inflorescence Tests",
    "-c", "init.defaultBranch=main",
]


def _git(cwd: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    subprocess.run(
        ["git", *_GIT_PRELUDE, "-C", str(cwd), *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _git_init(path: Path) -> None:
    """A repository with one commit — ``git worktree add`` refuses an unborn HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path)


@pytest.fixture(autouse=True)
async def _no_stray_tasks():
    """A sweep must leave no background work behind — a lease heartbeat least of all."""
    yield
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert pending == []
