"""Detect a linked git worktree and locate the main repository it belongs to.

A worktree is indexed as its own project (its checkout is a different tree), but almost all
of its content is shared with the main checkout, so the main project's index is the natural
cache donor. Finding that donor is pure filesystem archaeology: this module reads git's
on-disk layout directly and never shells out to ``git``. The indexer runs inside a container
where the binary may be absent or a different version, and a subprocess per indexed directory
is not something a hot path should pay for.

The layout being read (git ≥ 2.5)::

    main/.git/                       # the *common* dir — refs, objects, config
    main/.git/worktrees/<id>/gitdir  # abs path to the worktree's `.git` file
    main/.git/worktrees/<id>/commondir  # path back to main/.git, usually "../.."
    wt/.git                          # a *file*: "gitdir: /abs/main/.git/worktrees/<id>"

Every function here is total: unreadable, truncated, or hand-edited metadata yields "not a
worktree" rather than an exception. A wrong answer here would attach one project's cache to
an unrelated tree, so ambiguity always resolves to "no donor".
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GITDIR_PREFIX = "gitdir:"
# git's own truthy spelling for a valueless or explicit boolean in config.
_TRUE_VALUES = {"", "true", "yes", "on", "1"}


def _first_line(path: Path) -> str | None:
    """Read the single significant line of a git pointer file, or None if unusable.

    Pointer files (``.git``, ``gitdir``, ``commondir``) hold exactly one path. Anything that
    cannot be read as text, or holds nothing but whitespace, is treated as absent.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _resolve_relative_to(base: Path, raw: str) -> Path | None:
    """Interpret a path recorded in a git pointer file, which may be relative to *base*."""
    try:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base / candidate
        return candidate.resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def _nearest_dotgit(start: Path) -> Path | None:
    """Walk up from *start* to the first existing ``.git`` entry, directory or file."""
    for directory in (start, *start.parents):
        dotgit = directory / ".git"
        try:
            if dotgit.exists():
                return dotgit
        except OSError:
            return None
    return None


def _is_bare(git_dir: Path) -> bool:
    """True when *git_dir* is a bare repository — or when we cannot prove it is not.

    A bare repo has no working tree, so its parent directory is just wherever the ``.git``
    suffixed folder happens to live, not a checkout worth reusing an index from. Config is
    parsed by hand rather than with :mod:`configparser`, which mis-reads git's tab-indented
    entries as continuation lines. Failing closed on an unreadable config keeps the "no
    ambiguous donor" rule.
    """
    config = git_dir / "config"
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True

    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        if stripped.startswith("["):
            # "[core]" and '[remote "origin"]' alike — only the leading token names the section.
            section = stripped[1:].partition("]")[0].strip().split()[0].strip('"').lower()
            continue
        if section != "core":
            continue
        key, sep, value = stripped.partition("=")
        if key.strip().lower() == "bare":
            return value.strip().strip('"').lower() in _TRUE_VALUES if sep else True
    return False


def worktree_main_root(path: Path) -> Path | None:
    """Return the working-tree root of the main repository *path*'s worktree belongs to.

    ``None`` when *path* is not inside a linked worktree: an ordinary checkout (``.git`` is a
    directory), a submodule (``.git`` file pointing into ``.git/modules/...``, which has no
    ``commondir``), a path outside git entirely, or metadata too damaged to trust.

    A worktree of a *bare* repository also yields ``None`` — the main repo has no checkout, so
    there is no donor project to share a cache with.

    The membership test is ``commondir``, not a ``/worktrees/`` substring in the gitdir
    pointer: the pointer is a plain path that a directory named "worktrees" anywhere in a
    user's tree could imitate, while ``commondir`` is written by git only for real linked
    worktrees and names the main ``.git`` authoritatively. That also makes the answer
    independent of which checkout ``git worktree add`` was run from — git registers every
    worktree under the main ``.git``, so a worktree spawned from another worktree still
    resolves here to the same main root.
    """
    try:
        start = Path(path).resolve()
    except (OSError, ValueError, RuntimeError):
        return None

    dotgit = _nearest_dotgit(start)
    if dotgit is None or dotgit.is_dir():
        # No git above us, or an ordinary repository — either way, not a linked worktree.
        return None

    pointer = _first_line(dotgit)
    if pointer is None or not pointer.startswith(_GITDIR_PREFIX):
        return None
    raw_target = pointer[len(_GITDIR_PREFIX):].strip()
    if not raw_target:
        return None
    # A relative gitdir is relative to the directory holding the `.git` file, not to the
    # process cwd. `git worktree add --relative-paths` (and portable repo copies) rely on it.
    target = _resolve_relative_to(dotgit.parent, raw_target)
    if target is None or not target.is_dir():
        return None

    raw_common = _first_line(target / "commondir")
    if raw_common is None:
        return None
    common = _resolve_relative_to(target, raw_common)
    if common is None or not common.is_dir() or _is_bare(common):
        return None

    main_root = common.parent
    if not main_root.is_dir():
        return None
    logger.debug("Path %s is a linked worktree of main repository %s", start, main_root)
    return main_root


def registered_worktree_paths(main_root: Path) -> set[Path]:
    """Return the checkout roots of every worktree registered under *main_root*.

    Reads ``<main_root>/.git/worktrees/*/gitdir``, each of which records the absolute path of
    that worktree's ``.git`` file; the checkout root is its parent directory.

    Registration outlives the directory: git only drops an entry on ``git worktree prune``, so
    a returned path need not exist. That is deliberate — a caller reaping indexes for deleted
    worktrees needs to see exactly the entries git still considers live. Damaged or
    unreadable entries are skipped silently; a broken sibling must not hide the healthy ones.
    """
    roots: set[Path] = set()
    registry = Path(main_root) / ".git" / "worktrees"
    try:
        entries = list(registry.iterdir())
    except OSError:
        return roots

    for entry in entries:
        raw = _first_line(entry / "gitdir")
        if raw is None:
            continue
        # Relative entries are not written by git today, but resolve them against the registry
        # entry rather than discarding a worktree that a repo-relocation tool rewrote.
        gitdir_file = _resolve_relative_to(entry, raw)
        if gitdir_file is None or gitdir_file.parent == gitdir_file:
            continue
        roots.add(gitdir_file.parent)
    return roots
