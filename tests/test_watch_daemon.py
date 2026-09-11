"""The watch daemon — the process that keeps an index current after the session ends.

The failure this code exists to end is silent by construction: a watcher started inside an
MCP server dies with the session that started it, nothing says so, and every later answer is
served from a graph describing older code in confident present tense. So the tests here are
written around three questions, in order of how much they cost to get wrong:

1. **Does it spend money it was not allowed to?** The catch-up is an ordinary incremental
   re-index and is gated like one; a refusal must leave the daemon watching rather than dead.
2. **Does it touch the right project?** A registered path that no longer derives its project
   id (a moved or renamed tree) must be released, not followed — following it would apply one
   tree's edits to another tree's graph.
3. **Does it tell the truth about itself?** ``watch status``, ``index_directory`` and every
   read tool report the same three states, and each must distinguish "followed right now"
   from "registered but nothing is running" — the state after a reboot, and the one that
   looks fine in a registry while the index quietly rots.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from inflorescence.config import Settings
from inflorescence.cost import IndexCostExceededError
from inflorescence.db import lease
from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import IndexPreflightError, _project_name
from inflorescence.watch_daemon import WatchDaemon

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeConnection:
    """Answers the two queries this feature reads, keyed by the fragment that identifies them."""

    def __init__(self) -> None:
        self.watch_rows: list[dict[str, object]] = []
        self.reads: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []
        self.write_rows: list[dict[str, object]] = [{"matched": 1}]

    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        self.reads.append((query, dict(params or {})))
        return list(self.watch_rows)

    async def execute_write(self, query: str, params: dict | None = None) -> None:
        self.writes.append((query, dict(params or {})))

    async def execute_write_query(self, query: str, params: dict | None = None) -> list[dict]:
        self.writes.append((query, dict(params or {})))
        return list(self.write_rows)


class FakeWatchRepository:
    """The daemon's slice of the repository: a registry and a per-project scan scope."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
        self.scopes: dict[str, tuple[list[str], list[str]]] = {}
        self.list_calls = 0

    async def list_projects(self) -> list[dict[str, object]]:
        self.list_calls += 1
        return [dict(row) for row in self.rows]

    async def get_project_scope(self, project: str) -> tuple[list[str], list[str]]:
        return self.scopes.get(project, ([], []))

    def set_watched(self, project: str, watched: bool) -> None:
        for row in self.rows:
            if row["project"] == project:
                row["watched"] = watched


class FakeManager:
    """Derives project ids the way the real manager does, and records every index call."""

    def __init__(self, *, fail_with: Exception | None = None, result: dict | None = None) -> None:
        self.index_calls: list[tuple[str, list[str] | None, list[str] | None]] = []
        self.fail_with = fail_with
        self.result = result if result is not None else {"reused": True}

    def get_project(self, path: str) -> str:
        return _project_name(Path(path))

    async def index_directory(
        self,
        path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict:
        self.index_calls.append((path, include_patterns, exclude_patterns))
        if self.fail_with is not None:
            raise self.fail_with
        return dict(self.result)


class FakeWatcher:
    """Records which projects a watcher was started and stopped for."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def start(self, project: str, path: str) -> None:
        self.started.append((project, path))

    def stop(self, project: str) -> None:
        self.stopped.append(project)


def make_daemon(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    manager: FakeManager | None = None,
    catch_up: bool = True,
) -> tuple[WatchDaemon, FakeWatchRepository, FakeManager, FakeWatcher]:
    repo = FakeWatchRepository(rows if rows is not None else [])
    mgr = manager if manager is not None else FakeManager()
    watcher = FakeWatcher()
    daemon = WatchDaemon(
        manager=mgr,  # type: ignore[arg-type]
        repo=repo,  # type: ignore[arg-type]
        conn=None,  # no cross-process claim: exercised separately against the lease itself
        settings=Settings(_env_file=None),
        watcher=watcher,  # type: ignore[arg-type]
        catch_up=catch_up,
    )
    return daemon, repo, mgr, watcher


def watched_row(root: Path, *, watched: bool = True) -> dict[str, object]:
    return {
        "project": _project_name(root),
        "root_path": str(root),
        "worktree_of": None,
        "main_root_path": None,
        "watched": watched,
    }


# --------------------------------------------------------------------------------------
# Adoption
# --------------------------------------------------------------------------------------


async def test_adopts_a_registered_project_and_reconciles_it_before_trusting_the_watcher(
    tmp_path: Path,
) -> None:
    """Adoption watches first, then reconciles — a watcher hears nothing about the past.

    Everything that changed while no daemon ran is invisible to a fresh watcher, so adopting
    a project without reconciling it would leave the daemon faithfully tracking a graph that
    was already wrong.
    """
    root = tmp_path / "repo"
    root.mkdir()
    daemon, _repo, manager, watcher = make_daemon(tmp_path, rows=[watched_row(root)])

    report = await daemon.refresh()

    project = _project_name(root)
    assert report.adopted == [project]
    assert watcher.started == [(project, str(root))]
    assert manager.index_calls == [(str(root), None, None)]
    assert daemon.watching[project].catch_up == "already current"


async def test_ignores_projects_that_were_never_registered(tmp_path: Path) -> None:
    """Indexing a tree does not enlist it. A machine accumulates worktrees for branches
    abandoned weeks ago, and following those spends money summarizing commits nobody reads."""
    root = tmp_path / "repo"
    root.mkdir()
    daemon, _repo, manager, watcher = make_daemon(
        tmp_path, rows=[watched_row(root, watched=False)]
    )

    report = await daemon.refresh()

    assert report.adopted == []
    assert watcher.started == []
    assert manager.index_calls == []


async def test_skips_a_registration_whose_directory_is_gone(tmp_path: Path) -> None:
    """Registration outlives the directory (an unmounted volume, a deleted checkout).

    Skipping costs nothing and reverses itself: the next pass adopts the project again once
    the path is back. Nothing about the *index* is touched on this path — a disk that is not
    mounted is not evidence that a project should lose anything.
    """
    daemon, _repo, manager, watcher = make_daemon(
        tmp_path, rows=[watched_row(tmp_path / "not-there")]
    )

    report = await daemon.refresh()

    assert report.adopted == []
    assert watcher.started == []
    assert manager.index_calls == []
    assert "root path is not a directory" in next(iter(report.skipped.values()))


async def test_refuses_a_root_path_that_now_derives_a_different_project(tmp_path: Path) -> None:
    """A moved or renamed tree must not be followed under its old project id.

    Project ids are derived from the absolute path, so a registry row can outlive the identity
    it names. Watching it anyway would apply this tree's edits to a *different* project's
    graph — the one failure here that silently corrupts an index rather than merely leaving
    it stale.
    """
    root = tmp_path / "repo"
    root.mkdir()
    row = watched_row(root)
    row["project"] = "someone_elses_project_id"
    daemon, _repo, manager, watcher = make_daemon(tmp_path, rows=[row])

    report = await daemon.refresh()

    assert report.adopted == []
    assert watcher.started == []
    assert manager.index_calls == []
    assert "now derives project" in report.skipped["someone_elses_project_id"]


async def test_drops_a_project_once_it_is_unregistered(tmp_path: Path) -> None:
    """`watch rm` is honoured on the next pass, and stops the watcher rather than the index."""
    root = tmp_path / "repo"
    root.mkdir()
    daemon, repo, _manager, watcher = make_daemon(tmp_path, rows=[watched_row(root)])
    await daemon.refresh()
    project = _project_name(root)

    repo.set_watched(project, False)
    report = await daemon.refresh()

    assert report.dropped == [project]
    assert watcher.stopped == [project]
    assert daemon.watching == {}


async def test_adopting_twice_does_not_restart_a_live_watcher(tmp_path: Path) -> None:
    """The registry pass is a reconcile, not a restart: an unchanged registry is a no-op.

    Re-adopting every pass would re-run the catch-up scan every ``watch_refresh_seconds`` —
    a scan and an MD5 pass over the whole project, every thirty seconds, forever.
    """
    root = tmp_path / "repo"
    root.mkdir()
    daemon, _repo, manager, watcher = make_daemon(tmp_path, rows=[watched_row(root)])

    await daemon.refresh()
    await daemon.refresh()

    assert len(watcher.started) == 1
    assert len(manager.index_calls) == 1


# --------------------------------------------------------------------------------------
# Catch-up: it spends money, so every refusal has to be survivable
# --------------------------------------------------------------------------------------


async def test_a_cost_capped_catch_up_keeps_watching_and_reports_why(tmp_path: Path) -> None:
    """A cap refusal is a result, not a crash.

    The daemon must keep watching: the files saved from now on are small, cheap and exactly
    what the user is working on. Dying instead would turn one expensive backlog into a
    permanently dead daemon — and the refusal must reach `watch status`, not only a log.
    """
    root = tmp_path / "repo"
    root.mkdir()
    manager = FakeManager(fail_with=IndexCostExceededError(42.0, 10.0, changed_files=900))
    daemon, _repo, _mgr, watcher = make_daemon(tmp_path, rows=[watched_row(root)], manager=manager)

    report = await daemon.refresh()

    project = _project_name(root)
    assert report.adopted == [project]
    assert watcher.started == [(project, str(root))]
    assert "blocked by cost cap" in daemon.watching[project].catch_up


async def test_a_failing_catch_up_does_not_take_the_daemon_down(tmp_path: Path) -> None:
    """One project's broken index must not stop the others from being followed."""
    root = tmp_path / "repo"
    root.mkdir()
    manager = FakeManager(fail_with=IndexPreflightError("no API key"))
    daemon, _repo, _mgr, watcher = make_daemon(tmp_path, rows=[watched_row(root)], manager=manager)

    report = await daemon.refresh()

    project = _project_name(root)
    assert report.adopted == [project]
    assert "dependencies unavailable" in daemon.watching[project].catch_up


async def test_catch_up_reproduces_the_projects_stored_scan_scope(tmp_path: Path) -> None:
    """The reconcile is scoped exactly like the index that created the project.

    Without this a project indexed with ``exclude=["vendor"]`` would have vendor/** parsed,
    summarized and embedded the first time a daemon adopted it — scope silently widened, and
    money spent, by a background process nobody asked.
    """
    root = tmp_path / "repo"
    root.mkdir()
    daemon, repo, manager, _watcher = make_daemon(tmp_path, rows=[watched_row(root)])
    repo.scopes[_project_name(root)] = (["src"], ["vendor"])

    await daemon.refresh()

    assert manager.index_calls == [(str(root), ["src"], ["vendor"])]


async def test_catch_up_can_be_turned_off_entirely(tmp_path: Path) -> None:
    """``WATCH_CATCH_UP=false`` for someone who wants zero unattended spend."""
    root = tmp_path / "repo"
    root.mkdir()
    daemon, _repo, manager, watcher = make_daemon(
        tmp_path, rows=[watched_row(root)], catch_up=False
    )

    await daemon.refresh()

    assert manager.index_calls == []
    assert watcher.started  # still followed from here on
    assert daemon.watching[_project_name(root)].catch_up == "disabled"


async def test_shutdown_stops_every_watcher(tmp_path: Path) -> None:
    """A stopped daemon leaves nothing running, so a restart re-adopts from a clean slate."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    daemon, _repo, _manager, watcher = make_daemon(
        tmp_path, rows=[watched_row(first), watched_row(second)]
    )
    await daemon.refresh()

    await daemon.shutdown()

    assert sorted(watcher.stopped) == sorted([_project_name(first), _project_name(second)])
    assert daemon.watching == {}


# --------------------------------------------------------------------------------------
# The registry and the claim, read back the way every reporting surface reads them
# --------------------------------------------------------------------------------------


async def test_watch_state_distinguishes_registered_from_actually_running() -> None:
    """The three states are different answers, and only one of them means "trust this".

    ``watched`` without a live claim is the state after a reboot: the registry still says the
    project is followed, and nothing is following it. Reporting that as "watched" is exactly
    how an index rots behind a green light.
    """
    conn = FakeConnection()
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    now_ms = int(time.time() * 1000)

    conn.watch_rows = [{"watched": False, "holder": None, "expires_at": 0}]
    assert await repo.get_watch_state("p") == {
        "watched": False, "live": False, "holder": None, "daemon_live": False
    }

    conn.watch_rows = [{"watched": True, "holder": None, "expires_at": 0}]
    assert await repo.get_watch_state("p") == {
        "watched": True, "live": False, "holder": None, "daemon_live": False
    }

    conn.watch_rows = [{"watched": True, "holder": "host:42:abc", "expires_at": now_ms + 60_000}]
    assert await repo.get_watch_state("p") == {
        "watched": True,
        "live": True,
        "holder": "host:42:abc",
        "daemon_live": False,  # this project's claim, not the machine's daemon claim
    }

    # A daemon runs, but has not taken this project: registered and unfollowed, with nothing
    # to fix by starting another one.
    conn.watch_rows = [{
        "watched": True, "holder": None, "expires_at": 0,
        "daemon_holder": "host:42:abc", "daemon_expires_at": now_ms + 60_000,
    }]
    state = await repo.get_watch_state("p")
    assert (state["live"], state["daemon_live"]) == (False, True)


async def test_an_expired_claim_is_not_a_live_daemon() -> None:
    """A daemon killed with SIGKILL leaves its claim behind; the TTL is what retires it.

    Without the expiry check the lease node alone would report a dead daemon as live forever,
    and nothing would ever adopt those projects again.
    """
    conn = FakeConnection()
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    conn.watch_rows = [
        {"watched": True, "holder": "host:42:abc", "expires_at": int(time.time() * 1000) - 1}
    ]

    assert (await repo.get_watch_state("p"))["live"] is False


async def test_watch_state_reads_the_claim_under_its_own_key() -> None:
    """The claim must never be the project's index lock.

    Held for the daemon's whole lifetime, the index lock would block every index and every
    flush for as long as the project is watched — so the claim lives under a namespaced key
    on the same node type, and this is the test that says so.
    """
    conn = FakeConnection()
    repo = GraphRepository(conn)  # type: ignore[arg-type]
    conn.watch_rows = [{"watched": True, "holder": None, "expires_at": 0}]

    await repo.get_watch_state("myproj_1234")

    _query, params = conn.reads[0]
    assert params["claim_key"] == "watch:myproj_1234"
    assert params["claim_key"] != params["project"]
    assert lease.watch_claim_key("myproj_1234") == params["claim_key"]


async def test_registering_a_project_writes_the_flag_on_its_root_node() -> None:
    conn = FakeConnection()
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    assert await repo.set_project_watched("p_1", True) is True

    query, params = conn.writes[0]
    assert "dir:." in query
    assert params == {"project": "p_1", "watched": True}


async def test_registering_a_project_with_no_index_reports_the_refusal() -> None:
    """No root node means no index — and the daemon must not conjure one.

    A first index costs real money and stays an explicit, previewed act, so the write matching
    nothing is reported to the caller rather than silently registering a project that nothing
    will ever update.
    """
    conn = FakeConnection()
    conn.write_rows = [{"matched": 0}]
    repo = GraphRepository(conn)  # type: ignore[arg-type]

    assert await repo.set_project_watched("never_indexed", True) is False


# --------------------------------------------------------------------------------------
# The launchd job — the part that makes "always running" true across reboots
# --------------------------------------------------------------------------------------


def test_launch_agent_runs_the_daemon_and_carries_a_usable_path() -> None:
    """launchd hands a job a minimal PATH, and the daemon shells out to docker.

    Inheriting the installing shell's PATH is what keeps the agent from starting, failing to
    find docker, and dying in a log nobody reads — the exact invisible failure this feature
    exists to end.
    """
    from inflorescence.__main__ import render_launch_agent

    plist = render_launch_agent("/opt/bin/inflorescence", path_env="/opt/homebrew/bin:/usr/bin")

    assert "<string>/opt/bin/inflorescence</string>" in plist
    assert "<string>watch</string>" in plist
    assert "<string>run</string>" in plist
    assert "/opt/homebrew/bin:/usr/bin" in plist
    assert "<key>KeepAlive</key>" in plist  # a crashed daemon must come back
    assert "<key>RunAtLoad</key>" in plist  # ...and so must a rebooted machine


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"watched": True, "live": True}, "live"),
        ({"watched": True, "live": False}, "no daemon is running"),
        # A daemon *is* running and simply has not taken this project — telling the reader to
        # start one would send them after a process that is already up.
        ({"watched": True, "live": False, "daemon_live": True}, "not yet followed"),
        ({"watched": False, "live": False}, "none"),
    ],
)
def test_freshness_block_names_who_keeps_the_index_current(state: dict, expected: str) -> None:
    """Every read tool carries this line, because a clean page cannot say it.

    Freshness only ever looks at the handful of files one answer returned. Whether anything
    is keeping the *rest* current is a different question, and the one that decides whether a
    fresh-looking answer can be trusted an hour from now.
    """
    from inflorescence.tools.freshness import FreshnessChecker

    block = FreshnessChecker(Path("."), {}, watch_state=state).meta()

    assert block["watcher"].startswith(expected) or expected in block["watcher"]


# --------------------------------------------------------------------------------------
# What `index_directory` leaves behind, and what it says about it
# --------------------------------------------------------------------------------------


class WatchStateRepository:
    """Just enough repository to answer "is anything already watching this?"."""

    def __init__(self, state: dict) -> None:
        self.state = state

    async def get_watch_state(self, project: str) -> dict:
        return dict(self.state)


async def test_a_live_daemon_makes_the_session_watcher_step_aside() -> None:
    """Two watchers on one tree duplicate every flush for nothing.

    They serialize on the project lease, so the cost is wasted scans rather than a corrupt
    graph — but the daemon is the one that survives this session, so it is the one that keeps
    the tree, and the caller is told the index outlives the conversation.
    """
    from inflorescence.tools.indexing import _start_watching

    watcher = FakeWatcher()
    repo = WatchStateRepository({"watched": True, "live": True, "holder": "host:9:x"})

    note = await _start_watching(repo, watcher, "p_1", "/repo")  # type: ignore[arg-type]

    assert watcher.started == []
    assert "watch daemon" in note
    assert "after this session ends" in note


async def test_without_a_daemon_the_session_watcher_still_starts_but_says_what_it_is() -> None:
    """A watcher for the length of this session beats none — as long as nobody is misled.

    "Indexed" used to read as "kept current from now on", and stayed wrong from the moment the
    window closed. The watcher still starts; the sentence carries its expiry date and the
    command that removes it.
    """
    from inflorescence.tools.indexing import _start_watching

    watcher = FakeWatcher()
    repo = WatchStateRepository({"watched": False, "live": False, "holder": None})

    note = await _start_watching(repo, watcher, "p_1", "/repo")  # type: ignore[arg-type]

    assert watcher.started == [("p_1", "/repo")]
    assert "this session only" in note
    assert "inflorescence watch add /repo" in note


async def test_a_registered_but_unrun_project_is_told_to_start_the_daemon() -> None:
    """The post-reboot state has its own advice: start the daemon, do not register again."""
    from inflorescence.tools.indexing import _start_watching

    watcher = FakeWatcher()
    repo = WatchStateRepository({"watched": True, "live": False, "holder": None})

    note = await _start_watching(repo, watcher, "p_1", "/repo")  # type: ignore[arg-type]

    assert watcher.started == [("p_1", "/repo")]
    assert "inflorescence watch run" in note


async def test_a_viewer_process_with_no_watcher_says_so_plainly() -> None:
    """The dashboard's tools console indexes with no watcher on purpose (it is a viewer).

    The response must not imply anything is following the tree afterwards.
    """
    from inflorescence.tools.indexing import _start_watching

    repo = WatchStateRepository({"watched": False, "live": False, "holder": None})

    note = await _start_watching(repo, None, "p_1", "/repo")  # type: ignore[arg-type]

    assert "nothing is watching" in note


# --------------------------------------------------------------------------------------
# One daemon per machine
# --------------------------------------------------------------------------------------


async def test_a_second_daemon_on_this_machine_is_refused_by_name() -> None:
    """The refusal has to name the holder, or "already running" is another opaque failure.

    The usual way to get here is not exotic: `watch install` puts a daemon under launchd, and
    then `watch run` gets typed in a terminal months later. The holder string carries host and
    pid, which is what turns the message into something actionable.
    """
    from inflorescence.watch_daemon import claim_singleton

    conn = FakeConnection()
    conn.write_rows = [{"acquired": False, "holder": "mac.local:4242:abc"}]
    conn.watch_rows = [{"holder": "mac.local:4242:abc", "expires_at": int(time.time() * 1000) + 60_000}]

    claim, refusal = await claim_singleton(conn)  # type: ignore[arg-type]

    assert claim is None
    assert refusal is not None
    assert "mac.local:4242:abc" in refusal
    assert "watch uninstall" in refusal


async def test_the_first_daemon_takes_the_machine_claim() -> None:
    from inflorescence.watch_daemon import claim_singleton

    conn = FakeConnection()
    conn.write_rows = [{"acquired": True, "holder": "me"}]

    claim, refusal = await claim_singleton(conn)  # type: ignore[arg-type]

    assert refusal is None
    assert claim is not None
    await claim.__aexit__(None, None, None)


async def test_the_machine_claim_is_scoped_to_the_host_not_the_database() -> None:
    """Memgraph can be shared, and a machine-wide singleton must not become a database-wide one.

    Two laptops pointed at one database watch different files; locking them against each other
    would leave the second unable to run a daemon at all, for trees the first cannot even see.
    """
    from inflorescence.db.lease import daemon_claim_key, watch_claim_key

    assert daemon_claim_key("laptop-a") != daemon_claim_key("laptop-b")
    # ...and it can never collide with a project's own claim, or a project id that happened to
    # look like a hostname would silently unseat the daemon.
    assert daemon_claim_key("host") != watch_claim_key("host")


def test_launchd_restarts_a_failed_daemon_but_not_a_stopped_one() -> None:
    """Unconditional KeepAlive would spin: a refused start exits, and launchd would relaunch it
    every ten seconds forever. Restarting only failures also makes the SIGKILL case self-heal —
    each throttled retry keeps failing until the dead daemon's claim expires and it can take over.
    """
    from inflorescence.__main__ import render_launch_agent

    plist = render_launch_agent("/opt/bin/inflorescence", path_env="/usr/bin")

    assert "<key>SuccessfulExit</key>" in plist
    assert "<key>ThrottleInterval</key>" in plist
    assert "<key>KeepAlive</key>\n    <true/>" not in plist


def test_launch_agent_pins_the_directory_its_settings_come_from() -> None:
    """A launchd job starts in ``/``, where a project-local ``.env`` does not exist.

    On a machine whose key lives only in a checkout's `.env` — the normal shape when the MCP
    server is launched with `--directory` — the daemon then fails its pre-flight with a bare
    401 and, being configured to retry failures, does it every ten seconds forever. Pinning the
    install-time directory is what makes the job load what the installing shell loads.
    """
    from inflorescence.__main__ import render_launch_agent

    plist = render_launch_agent(
        "/opt/bin/inflorescence", path_env="/usr/bin", working_dir="/Users/me/code/thing"
    )

    assert "<key>WorkingDirectory</key>" in plist
    assert "<string>/Users/me/code/thing</string>" in plist


def test_install_refuses_when_the_daemon_would_find_no_api_key(monkeypatch, tmp_path) -> None:
    """Refuse at the moment a person is still watching, not in a log they will never open."""
    import sys as _sys

    from inflorescence import __main__ as cli

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("inflorescence.config.user_env_file", lambda: tmp_path / "absent.env")
    # If the guard ever stops firing, these make the test fail loudly instead of installing
    # a launchd job on the machine running the suite.
    monkeypatch.setattr(cli, "launch_agent_path", lambda *a, **k: tmp_path / "should-not-exist.plist")
    monkeypatch.setattr(cli, "_launchctl", lambda *a: pytest.fail("launchctl must not run"))

    assert cli._cmd_watch_install() == 1
    assert not (tmp_path / "should-not-exist.plist").exists()
