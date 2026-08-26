"""Git worktree detection, exercised against a real ``git`` binary.

The production code deliberately never shells out to git (see :mod:`inflorescence.worktree`),
so these tests are the only place that proves the hand-rolled ``.git``-file parsing agrees
with what git itself writes on disk. Fixtures therefore build genuine repositories,
worktrees, submodules and bare clones rather than hand-crafting metadata files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from inflorescence.worktree import registered_worktree_paths, worktree_main_root

# A clean-room git: no ~/.gitconfig, no /etc/gitconfig, and an identity supplied per call,
# so `commit` works on a machine where the developer has never configured git.
_GIT_PRELUDE = [
    "-c", "user.email=tests@inflorescence.invalid",
    "-c", "user.name=Inflorescence Tests",
    "-c", "init.defaultBranch=main",
    "-c", "protocol.file.allow=always",
]


def git(*args: str) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *_GIT_PRELUDE, *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def make_repo(path: Path, filename: str = "README.md") -> Path:
    """Create a repository with one commit — `git worktree add` refuses an unborn HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    git("init", str(path))
    (path / filename).write_text("hello\n", encoding="utf-8")
    git("-C", str(path), "add", filename)
    git("-C", str(path), "commit", "-m", "initial")
    return path


@pytest.fixture
def main_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "main")


def test_plain_repository_root_is_not_a_worktree(main_repo: Path) -> None:
    assert worktree_main_root(main_repo) is None


def test_subdirectory_of_plain_repository_is_not_a_worktree(main_repo: Path) -> None:
    nested = main_repo / "src" / "pkg"
    nested.mkdir(parents=True)
    assert worktree_main_root(nested) is None


def test_worktree_resolves_to_main_repository_root(main_repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    git("-C", str(main_repo), "worktree", "add", str(wt))

    assert worktree_main_root(wt) == main_repo.resolve()


def test_deep_subdirectory_inside_worktree_resolves_to_main_root(main_repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    git("-C", str(main_repo), "worktree", "add", str(wt))
    deep = wt / "a" / "b" / "c"
    deep.mkdir(parents=True)

    assert worktree_main_root(deep) == main_repo.resolve()


def test_worktree_created_from_another_worktree_still_points_at_main_root(
    main_repo: Path, tmp_path: Path
) -> None:
    """git registers every worktree under the *main* .git, never under a sibling worktree."""
    first = tmp_path / "wt1"
    second = tmp_path / "wt2"
    git("-C", str(main_repo), "worktree", "add", str(first))
    git("-C", str(first), "worktree", "add", str(second))

    assert worktree_main_root(second) == main_repo.resolve()
    assert (main_repo / ".git" / "worktrees" / "wt2").is_dir()


def test_relative_gitdir_pointer_is_resolved_against_the_git_file(main_repo: Path, tmp_path: Path) -> None:
    """`git worktree add --relative-paths` (git >= 2.48) writes a relative gitdir pointer.

    Rewritten by hand so the test also covers git builds that predate the flag; the resulting
    layout is byte-identical to what the flag produces.
    """
    wt = tmp_path / "wt"
    git("-C", str(main_repo), "worktree", "add", str(wt))
    relative = os.path.relpath(main_repo / ".git" / "worktrees" / "wt", wt)
    (wt / ".git").write_text(f"gitdir: {relative}\n", encoding="utf-8")

    assert not Path(relative).is_absolute()
    assert worktree_main_root(wt) == main_repo.resolve()


def test_directory_outside_any_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert worktree_main_root(plain) is None


def test_nonexistent_path(tmp_path: Path) -> None:
    assert worktree_main_root(tmp_path / "nope" / "nowhere") is None


def test_garbage_git_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / ".git").write_text("\x00 not a gitdir pointer at all\n", encoding="utf-8")
    assert worktree_main_root(junk) is None


def test_git_file_pointing_nowhere(tmp_path: Path) -> None:
    junk = tmp_path / "dangling"
    junk.mkdir()
    (junk / ".git").write_text(f"gitdir: {tmp_path / 'absent'}\n", encoding="utf-8")
    assert worktree_main_root(junk) is None


def test_submodule_is_not_a_worktree(tmp_path: Path) -> None:
    sub = make_repo(tmp_path / "sub", filename="lib.py")
    super_repo = make_repo(tmp_path / "super")
    git("-C", str(super_repo), "submodule", "add", f"file://{sub}", "sub")
    git("-C", str(super_repo), "commit", "-m", "add submodule")

    # The submodule checkout has a `.git` *file* just like a worktree, but its gitdir lives
    # under .git/modules/ and carries no commondir — it must not be mistaken for a worktree.
    assert (super_repo / "sub" / ".git").is_file()
    assert worktree_main_root(super_repo / "sub") is None


def test_bare_repository_and_its_worktree_have_no_donor(tmp_path: Path) -> None:
    source = make_repo(tmp_path / "source")
    bare = tmp_path / "bare.git"
    git("clone", "--bare", str(source), str(bare))
    wt = tmp_path / "bare-wt"
    git("-C", str(bare), "worktree", "add", str(wt))

    # A bare repo has no working tree, so there is no directory to reuse a cache from.
    assert worktree_main_root(bare) is None
    assert worktree_main_root(wt) is None


def test_registered_worktree_paths_lists_added_worktrees(main_repo: Path, tmp_path: Path) -> None:
    first = tmp_path / "wt1"
    second = tmp_path / "wt2"
    git("-C", str(main_repo), "worktree", "add", str(first))
    git("-C", str(main_repo), "worktree", "add", str(second))

    assert registered_worktree_paths(main_repo) == {first.resolve(), second.resolve()}


def test_registered_worktree_paths_is_empty_for_repository_without_worktrees(main_repo: Path) -> None:
    assert registered_worktree_paths(main_repo) == set()


def test_registered_worktree_paths_still_lists_deleted_worktree_before_prune(
    main_repo: Path, tmp_path: Path
) -> None:
    """Deleting the checkout does not unregister it — this is what a pruner has to clean up."""
    wt = tmp_path / "wt"
    git("-C", str(main_repo), "worktree", "add", str(wt))
    shutil.rmtree(wt)

    assert registered_worktree_paths(main_repo) == {wt.resolve()}

    git("-C", str(main_repo), "worktree", "prune")
    assert registered_worktree_paths(main_repo) == set()


def test_registered_worktree_paths_skips_broken_entries(main_repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    git("-C", str(main_repo), "worktree", "add", str(wt))
    worktrees = main_repo / ".git" / "worktrees"
    (worktrees / "empty").mkdir()
    (worktrees / "empty" / "gitdir").write_text("\n", encoding="utf-8")
    (worktrees / "missing-gitdir").mkdir()

    assert registered_worktree_paths(main_repo) == {wt.resolve()}


def test_registered_worktree_paths_on_non_repository(tmp_path: Path) -> None:
    assert registered_worktree_paths(tmp_path) == set()
