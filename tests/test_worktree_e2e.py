"""S1-S9 · A git worktree is a cheap separate project — end-to-end live oracles.

The contract this file proves, on a **real git repository** and a **live Memgraph**:

  * S1  a main checkout indexes normally: the LLM and both embedders are actually paid.
  * S2  a worktree whose content is byte-identical to the donor's costs **exactly zero**
        LLM calls and **exactly zero** embeddings, yet ends up with its own complete index —
        real summary text and real vectors in the database, not a hollow skeleton — carries
        the ``worktree_of`` marker, and answers queries scoped to itself (INV-9).
  * S3  a branch that edited one file pays for that file and nothing else: every node id
        this run summarized belongs to the changed file or to a container above it, and
        every chunk embedded belongs to that file.
  * S4  a worktree spawned from another worktree still seeds from the *main* checkout
        (git's registration is flat) — again exactly zero calls.
  * S5  the first *query* against a never-indexed worktree indexes it inline, for free, and
        inherits the donor's persisted scan scope, so an excluded tree is not silently
        parsed, summarized and paid for by a read tool.
  * S6  the donor moving ahead pays only for what the donor's own commit changed, and does
        not touch the worktree's project at all.
  * S7  a rebase in the worktree pays no more than the content the rebase actually changed
        in that checkout, and an immediately repeated index pays nothing.
  * S8  the janitor reaps a dead worktree's index by both death paths (``git worktree
        remove`` and a bare ``rm -rf``) while the donor and an ordinary project whose root
        path vanished survive untouched; re-creating the worktree re-indexes it free again.
  * S9  ``preview_directory`` on a worktree quotes ~$0 and explains it (``worktree_of``,
        donor-covered vs uncovered file counts) instead of quoting the whole tree.

These are **live** oracles. Cache reuse across two projects, the marker on the project root
node, the janitor's Cypher-guarded delete and query scoping only exist against real graph
state, so they are skipped unless ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a
**throwaway** Memgraph-MAGE instance — a developer's real ``bolt://localhost:7687`` holds
indexed projects and is never touched. Git is real (``subprocess``, as in
``tests/test_worktree.py``): the whole feature is filesystem archaeology over git's on-disk
layout, and hand-crafted metadata would prove nothing about it.

What this file deliberately does not pin is *how* the donor's cache reaches the new project.
Since the server-side clone landed that is a copy made inside Memgraph, with the client-side
seeding kept as its fallback; both are free, and the oracles here hold for either. The clone
itself — exactness, atomicity, and what happens when it is interrupted or cannot be trusted —
is proven in ``tests/test_worktree_clone_live.py`` (S10-S12).

Two defects found while writing this file are *not* worktree defects, and are deliberately
not asserted here — they would fail this suite for something it does not own, and pinning
the current behavior would bless it. Both are documented at the assertion that had to work
around them: the chunk-coverage marker that lands on only the first file of an UNWIND batch
(in ``test_s6_s7_...``), and vector search failing for every project for one garbage-
collection cycle after any project is deleted (in ``_vector_search``).

Only the two paid dependencies are faked, and they are faked as **recorders**, not stubs:
the LLM tags every summary it writes with the id of the run that paid for it, so "which
nodes did this run summarize?" is answered by reading the database rather than by trusting a
counter; the embedders keep every text they were handed. That is what turns "0 calls" into
"0 calls *and* the index is still complete".
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from inflorescence import janitor
from inflorescence.code_indexer.graph_builder import GraphBuilder
from inflorescence.code_indexer.models import IndexerConfig
from inflorescence.config import Settings
from inflorescence.db.connection import MemgraphConnection
from inflorescence.db.repository import GraphRepository
from inflorescence.db.schema import setup_schema
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.indexer import RAGIndexer

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536


# ---------------------------------------------------------------------------
# Real git
# ---------------------------------------------------------------------------

# A clean-room git: no ~/.gitconfig, no /etc/gitconfig, identity supplied per call, so
# `commit` works on a machine where the developer never configured git.
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


# A small but structurally real repository: functions, a class with methods, an import
# edge between two modules, and a prose file (documents are indexed by default).
ALPHA_PY = '''"""Arithmetic helpers."""


def alpha_add(a, b):
    """Return the sum of two numbers."""
    return a + b


class AlphaBox:
    """A box holding one value."""

    def __init__(self, value):
        self.value = value

    def doubled(self):
        """Return the value, twice."""
        return self.value * 2
'''

BETA_PY = '''"""Computation built on alpha."""

import alpha


def beta_compute(x):
    """Add x to itself using alpha."""
    return alpha.alpha_add(x, x)
'''

GAMMA_PY = '''"""Reporting helpers."""


def gamma_report(values):
    """Join values into a report line."""
    return ", ".join(str(v) for v in values)
'''

DELTA_PY = '''"""Entry point."""

import beta
import gamma


def delta_main():
    """Run the tiny pipeline."""
    return gamma.gamma_report([beta.beta_compute(2)])
'''

README_MD = """# Sample project

A tiny repository used to exercise worktree indexing end to end.
"""

BASE_FILES = {
    "alpha.py": ALPHA_PY,
    "beta.py": BETA_PY,
    "gamma.py": GAMMA_PY,
    "delta.py": DELTA_PY,
    "README.md": README_MD,
}


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def make_repo(path: Path, files: dict[str, str] | None = None) -> Path:
    """Create a repository with one commit — `git worktree add` refuses an unborn HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    git("init", str(path))
    _write_tree(path, files or BASE_FILES)
    git("-C", str(path), "add", "-A")
    git("-C", str(path), "commit", "-m", "initial")
    return path


def commit_all(repo: Path, message: str) -> None:
    git("-C", str(repo), "add", "-A")
    git("-C", str(repo), "commit", "-m", message)


# ---------------------------------------------------------------------------
# Recording fakes for the two paid dependencies
# ---------------------------------------------------------------------------


def _vec(text: str) -> list[float]:
    """A deterministic, content-derived vector — same text, same vector."""
    v = [0.05] * _DIM
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    for i in range(0, 8, 2):
        v[(digest[i] << 8 | digest[i + 1]) % _DIM] = 0.9
    return v


class _LLM:
    """A chat LLM that *signs* every summary with the id of the run that paid for it.

    ``summary_calls`` counts summarization only — the pre-flight ping (INV-2's probe, one
    per index call) is counted separately, because "this index spent nothing" is a claim
    about summaries, not about a liveness check that costs a token.

    Every returned summary is globally unique and prefixed with ``[<run>]``, so after a run
    the database itself answers "which nodes did *this* run summarize?" — no counter to
    trust, and the check survives any change to batching or ordering inside the summarizer.
    """

    def __init__(self) -> None:
        self.run = "run-0"
        self.pings = 0
        self.summary_calls = 0
        self.prompts: list[str] = []
        self._serial = 0

    def start_run(self, run: str) -> None:
        self.run = run
        self.summary_calls = 0
        self.prompts = []

    @property
    def tag(self) -> str:
        return f"[{self.run}]"

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        if prompt == "ping":
            self.pings += 1
            return "ok"
        self.summary_calls += 1
        self._serial += 1
        self.prompts.append(prompt)
        return f"{self.tag} summary #{self._serial}"


class _Embedder:
    """An embedder that keeps every text it was asked to embed.

    ``batches`` counts API round-trips, ``texts`` the individual items; a strict "nothing was
    embedded" asserts on both, since a single empty batch would satisfy neither alone.
    ``preflight`` is deliberately not counted as an embed: it is INV-2's probe, not spend.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.model = f"fake-{kind}-embedder"
        self.batches = 0
        self.preflights = 0
        self.texts: list[str] = []

    def start_run(self) -> None:
        self.batches = 0
        self.texts = []

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        self.batches += 1
        self.texts.extend(texts)
        return [_vec(t) for t in texts]

    def preflight(self) -> None:
        self.preflights += 1


# ---------------------------------------------------------------------------
# One live world: connection, repository, manager, recorders
# ---------------------------------------------------------------------------


@dataclass
class _World:
    conn: MemgraphConnection
    repo: GraphRepository
    manager: ProjectManager
    settings: Settings
    llm: _LLM
    summary_embedder: _Embedder
    code_embedder: _Embedder
    extra_projects: list[str] = field(default_factory=list)

    def start_run(self, run: str) -> None:
        """Open a new accounting window: nothing paid yet, and summaries signed ``[run]``."""
        self.llm.start_run(run)
        self.summary_embedder.start_run()
        self.code_embedder.start_run()

    def assert_free(self, what: str) -> None:
        """The strict zero: no summary was generated and no vector was computed."""
        assert self.llm.summary_calls == 0, (
            f"{what}: {self.llm.summary_calls} LLM summarization call(s) were paid for, "
            f"expected exactly 0 (prompts: {self.llm.prompts[:3]})"
        )
        assert self.summary_embedder.batches == 0 and not self.summary_embedder.texts, (
            f"{what}: {len(self.summary_embedder.texts)} summary embedding(s) were paid for, "
            "expected exactly 0"
        )
        assert self.code_embedder.batches == 0 and not self.code_embedder.texts, (
            f"{what}: {len(self.code_embedder.texts)} code embedding(s) were paid for, "
            "expected exactly 0"
        )

    async def summarized_ids(self, project: str, run: str) -> set[str]:
        """Node ids whose summary in the database was written by *run* — read back, not counted."""
        records = await self.conn.execute_query(
            "MATCH (n:Code {project: $project}) WHERE n.summary STARTS WITH $tag RETURN n.id AS id",
            {"project": project, "tag": f"[{run}]"},
        )
        return {str(r["id"]) for r in records}

    async def close(self) -> None:
        projects = set(self.manager._projects.values()) | set(self.extra_projects)
        for project in projects:
            # One project that cannot be dropped must not leave the others behind in a
            # database the next test will read.
            with contextlib.suppress(Exception):
                await self.repo.delete_project(project)
        await self.conn.close()


async def _world(**overrides: object) -> _World:
    settings = Settings(_env_file=None, memgraph_url=LIVE_URL, **overrides)  # type: ignore[arg-type]
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    repo = GraphRepository(conn)
    llm = _LLM()
    # The production default registry (Python + prose documents), so this exercises the same
    # scan the server does. SCIP semantic enrichment is off: it shells out to containerized
    # indexers, which is a different subsystem's dependency, not this feature's.
    builder = GraphBuilder(
        repo=repo,
        llm=llm,
        settings=settings,
        config=IndexerConfig(
            use_llm_summaries=True,
            use_llm_fallback_parser=False,
            use_scip_semantic=False,
        ),
    )
    rag = RAGIndexer(repo=repo, settings=settings)
    summary_embedder = _Embedder("summary")
    code_embedder = _Embedder("code")
    rag._summary_embedder = summary_embedder  # type: ignore[assignment]
    rag._code_embedder = code_embedder  # type: ignore[assignment]
    manager = ProjectManager(
        repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn
    )
    return _World(
        conn=conn,
        repo=repo,
        manager=manager,
        settings=settings,
        llm=llm,
        summary_embedder=summary_embedder,
        code_embedder=code_embedder,
    )


async def _vector_search(repo: GraphRepository, project: str, vector: list[float]) -> list[dict]:
    """``search_summary_vectors`` with a bounded wait for Memgraph's garbage collector.

    Deleting a project (any project — a test's teardown, or a janitor sweep on the running
    server) leaves the vector indexes pointing at deleted nodes, and until Memgraph's GC
    cycle passes, *every* ``vector_search.search`` — for every project — fails outright with
    ``Memgraph.ClientError … Trying to get a property from a deleted object``. That is a real
    defect (see this file's report), not a property of worktrees, and it is deterministic:
    delete a project, search immediately, and it raises; ~20 s later it answers normally.

    So the retry is deliberately narrow: only that one message, only for as long as one GC
    cycle, and any other failure — or a timeout — fails the test. Without it, this oracle
    would be green or red depending on how recently some other test deleted something.
    """
    deadline = time.monotonic() + 45.0
    while True:
        try:
            return await repo.search_summary_vectors(project, vector, top_k=200)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless it is *that* error
            if "deleted object" not in str(exc) or time.monotonic() > deadline:
                raise
            await asyncio.sleep(2.0)


def _belongs_to(node_id: str, file_path: str) -> bool:
    """True when *node_id* is the file's own node or a symbol inside it."""
    return node_id == file_path or node_id.startswith(f"{file_path}::")


def _is_container(node_id: str) -> bool:
    """Directory nodes. A container legitimately re-summarizes when a child's summary changes."""
    return node_id.startswith("dir:")


# ---------------------------------------------------------------------------
# S1 + S2 + S9 · the free worktree
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s1_s2_s9_identical_worktree_is_indexed_for_free_and_completely(tmp_path: Path) -> None:
    """S1 the main checkout is paid for; S2 an identical worktree is not, yet is complete.

    S2 is the load-bearing claim of the whole feature, so "0 calls" is not asserted alone:
    the worktree project must end the run holding the *same* summary text and a real vector
    on every node (a hollow skeleton would also make zero calls), carry the marker naming
    its donor, and scope its own queries (INV-9). S9 checks the preview quotes ~$0 for it
    and explains why, before anything is written.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo

        # ---- S1: a first, ordinary index. Everything is paid for.
        world.start_run("main")
        main_project = manager.get_project(str(main))
        stats = await manager.index_directory(str(main))
        assert stats["reused"] is False
        assert int(stats["nodes"]) > 0 and int(stats["edges"]) > 0
        assert world.llm.summary_calls > 0, "S1 indexed a real repository without paying the LLM"
        assert world.code_embedder.texts, "S1 stored no code chunk embeddings"
        assert world.summary_embedder.texts, "S1 stored no summary embeddings"
        s1_llm, s1_code, s1_summary = (
            world.llm.summary_calls,
            len(world.code_embedder.texts),
            len(world.summary_embedder.texts),
        )
        donor_summaries = await repo.get_stored_summaries(main_project)
        donor_chunks = await repo.get_chunk_ids(main_project)
        donor_stats = await repo.get_project_stats(main_project)
        assert len(donor_summaries) == s1_llm, "S1 paid for summaries it did not store"

        # ---- S2: the same commit, checked out again. `--detach` because git refuses to
        # check out one branch twice; the tree is byte-identical, which is what matters.
        worktree = tmp_path / "wt-same"
        git("-C", str(main), "worktree", "add", "--detach", str(worktree))
        wt_project = manager.get_project(str(worktree))

        # ---- S9 (free half): the preview must not quote the whole tree.
        preview = await manager.preview_directory(str(worktree))
        assert preview["worktree_of"] == main_project
        assert preview["uncovered_files"] == 0, "an identical worktree has uncovered files"
        assert preview["files"] == 0
        assert int(preview["donor_covered_files"]) == len(BASE_FILES)
        assert float(preview["estimated_cost_usd"]) == pytest.approx(0.0, abs=1e-9)

        world.start_run("wt-same")
        wt_stats = await manager.index_directory(str(worktree))

        # The strict zeros.
        world.assert_free("S2 (worktree identical to its donor)")
        assert wt_stats["worktree_of"] == main_project
        # How the donor's cache was carried over depends on which path ran: a server-side
        # clone reports the copied counts, the client-side seeding the summaries it wrote
        # back. The claim here is the same either way — nothing was regenerated. Which
        # mechanism produced it, and whether the copy is exact, is S10's business
        # (tests/test_worktree_clone_live.py).
        assert wt_stats.get("cloned") or int(wt_stats.get("seeded_summaries", 0)) > 0, (
            f"the worktree index reused nothing from its donor: {wt_stats}"
        )

        # ...and the index is nonetheless complete: same nodes, same summary text, real vectors.
        wt_summaries = await repo.get_stored_summaries(wt_project)
        assert set(wt_summaries) == set(donor_summaries), "the worktree project is missing nodes"
        for node_id, row in wt_summaries.items():
            assert row["summary"], f"worktree node {node_id} ended up with an empty summary"
            assert row["summary"] == donor_summaries[node_id]["summary"], (
                f"worktree node {node_id} was re-summarized instead of reusing the donor's text"
            )
            assert row["summary_embedding"] is not None, (
                f"worktree node {node_id} has a summary but no vector — search would miss it"
            )
        assert await repo.get_chunk_ids(wt_project) == donor_chunks, (
            "the worktree's chunk vectors are not the donor's content-identical ones"
        )
        assert await repo.get_project_stats(wt_project) == donor_stats

        # The marker: this project is the index of a worktree of the main checkout.
        assert await repo.get_project_worktree_meta(wt_project) == {
            "worktree_of": main_project,
            "main_root_path": str(main.resolve()),
        }
        assert await repo.get_project_worktree_meta(main_project) is None, (
            "the ordinary main checkout was marked as a worktree"
        )

        # ---- INV-9: a query from the worktree sees the worktree, and nothing else. A third,
        # unrelated project holds a symbol that exists nowhere else; it must stay invisible.
        other = tmp_path / "other"
        other.mkdir()
        (other / "only_here.py").write_text(
            "def zzz_unrelated_symbol():\n    return 1\n", encoding="utf-8"
        )
        other_project = manager.get_project(str(other))
        await manager.index_directory(str(other))

        assert await repo.search_symbol_text(other_project, "zzz_unrelated_symbol", top_k=25), (
            "sanity: the unrelated symbol is not findable even in its own project"
        )
        assert await repo.search_symbol_text(wt_project, "zzz_unrelated_symbol", top_k=25) == [], (
            "a search scoped to the worktree returned another project's node (INV-9)"
        )
        wt_ids = await repo.get_node_ids(wt_project)
        hits = await _vector_search(repo, wt_project, _vec("anything"))
        assert hits, "semantic search over the seeded worktree returned nothing"
        assert {h["id"] for h in hits} <= wt_ids, "vector search leaked ids from another project"

        print(
            f"\nS1: llm={s1_llm} code_embeds={s1_code} summary_embeds={s1_summary} "
            f"nodes={donor_stats['nodes']}\nS2: llm=0 code_embeds=0 summary_embeds=0 "
            f"cloned={wt_stats.get('cloned')} "
            f"seeded_summaries={wt_stats.get('seeded_summaries')}"
        )
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S3 · a branch pays for exactly what it changed
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s3_branch_pays_only_for_the_file_it_changed(tmp_path: Path) -> None:
    """S3: one edited file is the only thing that reaches the LLM and the embedder.

    Asserted by identity, not by count: every node the run summarized must be the changed
    file, a symbol inside it, or a container above it (a directory whose child summary moved
    legitimately re-summarizes), and every chunk vector the run bought must belong to that
    file. A count alone would pass just as happily if the *wrong* file had been re-paid.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))
        donor_chunks = await repo.get_chunk_ids(main_project)

        worktree = tmp_path / "wt-feature"
        git("-C", str(main), "worktree", "add", "-b", "feature", str(worktree))
        (worktree / "beta.py").write_text(
            BETA_PY + '\n\ndef beta_branch_only(x):\n    """Only this branch has it."""\n    return x - 1\n',
            encoding="utf-8",
        )
        commit_all(worktree, "branch edits beta")
        wt_project = manager.get_project(str(worktree))

        # S9 (paid half): the preview prices exactly the one file the donor does not cover.
        preview = await manager.preview_directory(str(worktree))
        assert preview["worktree_of"] == main_project
        assert preview["uncovered_files"] == 1, (
            f"expected exactly beta.py to be uncovered, got {preview['uncovered_files']}"
        )
        assert int(preview["donor_covered_files"]) == len(BASE_FILES) - 1
        assert float(preview["estimated_cost_usd"]) > 0.0

        world.start_run("wt-feature")
        wt_stats = await manager.index_directory(str(worktree))
        assert wt_stats["worktree_of"] == main_project

        paid_ids = await world.summarized_ids(wt_project, "wt-feature")
        assert paid_ids, "the changed file was not summarized at all"
        stray = {
            nid for nid in paid_ids
            if not (_belongs_to(nid, "beta.py") or _is_container(nid))
        }
        assert not stray, f"nodes unrelated to the changed file were re-summarized: {sorted(stray)}"
        assert any(_belongs_to(nid, "beta.py") for nid in paid_ids)
        assert world.llm.summary_calls == len(paid_ids)

        # Untouched files keep the donor's text verbatim.
        wt_summaries = await repo.get_stored_summaries(wt_project)
        donor_summaries = await repo.get_stored_summaries(main_project)
        for node_id, row in wt_summaries.items():
            if node_id in paid_ids:
                continue
            assert row["summary"] == donor_summaries[node_id]["summary"], (
                f"{node_id} was not reused from the donor although its content is identical"
            )

        # Chunk vectors: only beta.py's chunks are new, and only those were embedded.
        wt_chunk_index = await repo.get_chunk_index(wt_project)
        new_chunks = {cid: fp for cid, fp in wt_chunk_index.items() if cid not in donor_chunks}
        assert new_chunks, "the edited file produced no new chunks"
        assert set(new_chunks.values()) == {"beta.py"}, (
            f"chunks were re-embedded for files the branch never touched: {sorted(set(new_chunks.values()))}"
        )
        assert len(world.code_embedder.texts) == len(new_chunks), (
            "the number of code embeddings paid for does not match the number of new chunks"
        )

        # Summary vectors: only the freshly summarized nodes were re-embedded.
        paid_summaries = {str(wt_summaries[nid]["summary"]) for nid in paid_ids}
        assert set(world.summary_embedder.texts) == paid_summaries, (
            "summary embeddings were bought for text this run did not generate"
        )

        # INV-9 again, now with a symbol that really exists in only one of the two projects.
        assert await repo.search_symbol_text(wt_project, "beta_branch_only", top_k=25), (
            "the branch-only symbol is not searchable in the worktree project"
        )
        assert await repo.search_symbol_text(main_project, "beta_branch_only", top_k=25) == [], (
            "the branch-only symbol leaked into the main checkout's project (INV-9)"
        )

        print(
            f"\nS3: llm={world.llm.summary_calls} (nodes {sorted(paid_ids)}) "
            f"code_embeds={len(world.code_embedder.texts)} "
            f"summary_embeds={len(world.summary_embedder.texts)}"
        )
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S4 · a worktree of a worktree still seeds from main
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s4_worktree_of_a_worktree_seeds_from_the_main_checkout(tmp_path: Path) -> None:
    """S4: git registers every worktree under the main ``.git``, so the donor is always main.

    The intermediate worktree is deliberately left **unindexed**: if detection walked to the
    nearest checkout instead of to ``commondir``'s owner, there would be no donor at all and
    this index would summarize the whole tree at full price.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))

        first = tmp_path / "wt-first"
        git("-C", str(main), "worktree", "add", "--detach", str(first))
        second = tmp_path / "wt-second"
        git("-C", str(first), "worktree", "add", "--detach", str(second))

        world.start_run("wt-second")
        second_project = manager.get_project(str(second))
        stats = await manager.index_directory(str(second))

        world.assert_free("S4 (worktree spawned from another worktree)")
        assert stats["worktree_of"] == main_project, (
            f"donor should be the main checkout, got {stats.get('worktree_of')}"
        )
        assert await repo.get_project_worktree_meta(second_project) == {
            "worktree_of": main_project,
            "main_root_path": str(main.resolve()),
        }
        assert await repo.get_project_stats(second_project) == await repo.get_project_stats(main_project)
        assert not await repo.project_is_indexed(manager.get_project(str(first))), (
            "sanity: the intermediate worktree must stay unindexed for this test to mean anything"
        )
        print("\nS4: llm=0 code_embeds=0 summary_embeds=0")
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S5 · the first query indexes the worktree, for free, within the donor's scope
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s5_first_query_auto_indexes_the_worktree_and_inherits_the_donor_scope(
    tmp_path: Path,
) -> None:
    """S5: a read tool's first query against a fresh worktree indexes it inline and free.

    "thirdparty" is deliberately not one of ``DEFAULT_EXCLUDE_PATTERNS`` (unlike "vendor"),
    so only the donor's *persisted* scope — not the global config — can keep the auto-index
    from parsing, summarizing and embedding it. If the scope were not inherited, the file
    would be indexed and the zero below would be a non-zero paid for by a mere query.
    """
    files = {**BASE_FILES, "thirdparty/junk.py": "def junk_only_symbol():\n    return 1\n"}
    main = make_repo(tmp_path / "main", files)
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main), exclude_patterns=["thirdparty/**"])
        main_ids = await repo.get_node_ids(main_project)
        assert not any(nid.startswith("thirdparty/") for nid in main_ids), (
            "sanity: the exclude was not honored at index time"
        )

        worktree = tmp_path / "wt-query"
        git("-C", str(main), "worktree", "add", "--detach", str(worktree))
        assert (worktree / "thirdparty" / "junk.py").is_file(), (
            "sanity: the excluded file must exist on disk in the worktree"
        )

        world.start_run("wt-query")
        # No index_directory call: this is the read tool's own entry point.
        project, reason = await manager.ensure_indexed_for_query(str(worktree))
        assert reason is None, f"the first query refused to index the worktree: {reason}"
        assert project == manager.get_project(str(worktree))
        world.assert_free("S5 (auto-index on first query)")

        assert await repo.project_is_indexed(project)
        wt_ids = await repo.get_node_ids(project)
        assert wt_ids == main_ids, "the auto-indexed worktree does not mirror its donor"
        assert not any(nid.startswith("thirdparty/") for nid in wt_ids), (
            "the auto-index widened the scan past the donor's persisted exclude — a query paid "
            "for files the operator excluded"
        )
        assert await repo.get_project_scope(project) == ([], ["thirdparty/**"]), (
            "the auto-index did not persist the scope it inherited, so a later watcher "
            "update would widen it again"
        )
        assert await repo.search_symbol_text(project, "junk_only_symbol", top_k=25) == []
        print("\nS5: llm=0 code_embeds=0 summary_embeds=0 (scope inherited)")
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S6 + S7 · the donor moves ahead; the worktree rebases onto it
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s6_s7_donor_moves_ahead_then_the_worktree_rebases(tmp_path: Path) -> None:
    """S6 a donor commit is paid for alone and leaves the worktree's project untouched;
    S7 a rebase pays no more than the content it actually changed, and a repeat pays nothing.

    S6's second half is the one that matters operationally: two projects share content but
    not identity, so re-indexing one must not rewrite, re-summarize or re-embed anything in
    the other. Equality of the whole stats dict plus of every stored checksum is the check.

    S7 is bounded, not zeroed, on purpose. Donor seeding is a *first-index* mechanism; once
    the worktree has its own project the incremental path reuses that project's summaries by
    input hash, so content the rebase pulled in from the donor is re-summarized here even
    though the donor already paid for it. The oracle therefore pins the honest bound — never
    more than what the rebase changed in this checkout — and the strict zero is asserted, for
    the immediately following no-op index, over everything this feature governs: no summary
    is generated and no summary vector is bought. Code-chunk re-embedding is excluded there
    and only reported; see the comment at that assertion for the pre-existing,
    worktree-independent reason.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))

        worktree = tmp_path / "wt-feature"
        git("-C", str(main), "worktree", "add", "-b", "feature", str(worktree))
        (worktree / "beta.py").write_text(
            BETA_PY + "\n\ndef beta_branch_only(x):\n    return x - 1\n", encoding="utf-8"
        )
        commit_all(worktree, "branch edits beta")
        wt_project = manager.get_project(str(worktree))
        world.start_run("wt-feature")
        await manager.index_directory(str(worktree))

        wt_stats_before = await repo.get_project_stats(wt_project)
        wt_checksums_before = await repo.get_checksums(wt_project)
        wt_summaries_before = {
            k: v["summary"] for k, v in (await repo.get_stored_summaries(wt_project)).items()
        }

        # ---- S6: the main checkout gains a commit touching exactly one file.
        (main / "alpha.py").write_text(
            ALPHA_PY + "\n\ndef alpha_scaled(a, factor):\n    return a * factor\n", encoding="utf-8"
        )
        commit_all(main, "main edits alpha")

        world.start_run("main-2")
        await manager.index_directory(str(main))
        donor_paid = await world.summarized_ids(main_project, "main-2")
        assert donor_paid, "the donor's own commit was not summarized"
        stray = {
            nid for nid in donor_paid
            if not (_belongs_to(nid, "alpha.py") or _is_container(nid))
        }
        assert not stray, f"the donor update re-summarized unrelated nodes: {sorted(stray)}"
        s6_llm = world.llm.summary_calls

        # The worktree's project is a different project: nothing about it moved.
        assert await repo.get_project_stats(wt_project) == wt_stats_before, (
            "re-indexing the donor changed the worktree project's graph"
        )
        assert await repo.get_checksums(wt_project) == wt_checksums_before
        assert {
            k: v["summary"] for k, v in (await repo.get_stored_summaries(wt_project)).items()
        } == wt_summaries_before, "re-indexing the donor rewrote the worktree's summaries"
        assert not await world.summarized_ids(wt_project, "main-2"), (
            "the donor's run wrote summaries into the worktree's project"
        )

        # ---- S7: the branch rebases onto the new main. alpha.py in this checkout becomes
        # the donor's new content; beta.py keeps the branch's edit.
        before = {rel: (worktree / rel).read_text(encoding="utf-8") for rel in BASE_FILES}
        git("-C", str(worktree), "rebase", "main")
        changed_by_rebase = {
            rel for rel, text in before.items()
            if (worktree / rel).read_text(encoding="utf-8") != text
        }
        assert changed_by_rebase == {"alpha.py"}, (
            f"sanity: the rebase was expected to move only alpha.py, moved {sorted(changed_by_rebase)}"
        )

        world.start_run("wt-rebased")
        await manager.index_directory(str(worktree))
        rebase_paid = await world.summarized_ids(wt_project, "wt-rebased")
        stray = {
            nid for nid in rebase_paid
            if not (any(_belongs_to(nid, rel) for rel in changed_by_rebase) or _is_container(nid))
        }
        assert not stray, (
            f"the rebase paid for content it did not change: {sorted(stray)}"
        )
        s7_llm = world.llm.summary_calls

        # ---- S7 (strict half): nothing changed since — the repeat generates nothing.
        rebased_summaries = {
            k: v["summary"] for k, v in (await repo.get_stored_summaries(wt_project)).items()
        }
        rebased_stats = await repo.get_project_stats(wt_project)
        rebased_chunk_files = set((await repo.get_chunk_index(wt_project)).values())
        world.start_run("wt-noop")
        stats = await manager.index_directory(str(worktree))
        assert stats["reused"] is True
        assert world.llm.summary_calls == 0, (
            "re-indexing an unchanged worktree re-summarized nodes: "
            f"{sorted(await world.summarized_ids(wt_project, 'wt-noop'))}"
        )
        assert not world.summary_embedder.texts, "an unchanged re-index bought summary vectors"
        assert await repo.get_project_stats(wt_project) == rebased_stats
        assert {
            k: v["summary"] for k, v in (await repo.get_stored_summaries(wt_project)).items()
        } == rebased_summaries
        # Code-chunk embeddings are deliberately NOT asserted to be zero here, and the number
        # is printed instead. On this Memgraph they are not zero, for a reason that has
        # nothing to do with worktrees: the chunk-coverage marker written by
        # `queries.MARK_CHUNKS_COVERED` lands on only the first file of each UNWIND batch,
        # so `get_files_with_stale_chunks` keeps reporting files that are in fact fully
        # chunked and `_repair_incomplete_index` re-chunks them on every run. The same
        # happens to an ordinary project that was never a worktree; asserting zero here
        # would make this file fail for a defect it does not own, and asserting the current
        # non-zero would bless it.
        assert set((await repo.get_chunk_index(wt_project)).values()) == rebased_chunk_files, (
            "the unchanged re-index changed which files have code chunks"
        )

        print(
            f"\nS6: llm={s6_llm} (donor nodes {sorted(donor_paid)}), worktree untouched\n"
            f"S7: llm={s7_llm} (rebase, nodes {sorted(rebase_paid)}); "
            f"repeat llm=0 summary_embeds=0 code_embeds={len(world.code_embedder.texts)} "
            "(see the chunk-coverage note above)"
        )
    finally:
        await world.close()


# ---------------------------------------------------------------------------
# S8 · the janitor reaps dead worktrees only
# ---------------------------------------------------------------------------


@live
@pytest.mark.asyncio
async def test_s8_janitor_reaps_both_death_paths_and_spares_everything_else(tmp_path: Path) -> None:
    """S8: both ways a worktree dies are reaped; the donor and an ordinary project are not.

    The two deaths differ in git's registry, which is layer 4 of the janitor's defence:
    ``git worktree remove`` de-registers the path, while ``rm -rf`` leaves the registration
    behind with nothing at the end of it. Both must be reaped, and the reaping must be the
    *only* change to the database — an ordinary project whose root path vanished (the
    unmounted-volume signature) has to survive, which is the accident the marker gate exists
    to prevent. Recreating the worktree afterwards must again cost nothing, because that is
    what makes deleting the index cheap enough to be allowed at all.
    """
    main = make_repo(tmp_path / "main")
    world = await _world()
    try:
        manager, repo = world.manager, world.repo
        world.start_run("main")
        main_project = manager.get_project(str(main))
        await manager.index_directory(str(main))

        removed = tmp_path / "wt-removed"      # dies via `git worktree remove`
        rmrfed = tmp_path / "wt-rmrfed"        # dies via `rm -rf`
        git("-C", str(main), "worktree", "add", "--detach", str(removed))
        git("-C", str(main), "worktree", "add", "--detach", str(rmrfed))
        removed_project = manager.get_project(str(removed))
        rmrfed_project = manager.get_project(str(rmrfed))
        await manager.index_directory(str(removed))
        await manager.index_directory(str(rmrfed))

        # An ordinary (non-worktree) project whose directory then disappears.
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "solo.py").write_text("def solo():\n    return 1\n", encoding="utf-8")
        plain_project = manager.get_project(str(plain))
        await manager.index_directory(str(plain))
        assert await repo.get_project_worktree_meta(plain_project) is None

        main_stats_before = await repo.get_project_stats(main_project)
        plain_stats_before = await repo.get_project_stats(plain_project)

        git("-C", str(main), "worktree", "remove", "--force", str(removed))
        shutil.rmtree(rmrfed)
        shutil.rmtree(plain)

        report = await janitor.sweep(repo, world.settings, conn=world.conn)
        deleted = set(report["deleted"])
        assert removed_project in deleted, "a worktree removed with `git worktree remove` was kept"
        assert rmrfed_project in deleted, "a worktree deleted with `rm -rf` was kept"
        assert main_project not in deleted, "the janitor deleted the donor's index"
        assert plain_project not in deleted, (
            "the janitor deleted an ordinary project whose root path vanished — the exact "
            "unmounted-volume accident the marker gate exists to prevent"
        )
        assert not await repo.project_is_indexed(removed_project)
        assert not await repo.project_is_indexed(rmrfed_project)
        assert await repo.get_project_stats(main_project) == main_stats_before
        assert await repo.get_project_stats(plain_project) == plain_stats_before

        # Re-creating the worktree at the same path re-indexes it free again: the donor is
        # alive, so everything the janitor threw away comes back out of its cache.
        git("-C", str(main), "worktree", "prune")
        git("-C", str(main), "worktree", "add", "--detach", str(rmrfed))
        world.start_run("wt-recreated")
        stats = await manager.index_directory(str(rmrfed))
        world.assert_free("S8 (worktree re-created after the janitor reaped it)")
        assert stats["worktree_of"] == main_project
        assert await repo.get_project_stats(rmrfed_project) == main_stats_before

        print(
            f"\nS8: deleted={sorted(deleted & {removed_project, rmrfed_project})} "
            f"donor+plain intact, re-index llm=0"
        )
    finally:
        await world.close()
