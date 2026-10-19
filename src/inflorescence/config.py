"""Configuration via Pydantic Settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


def user_config_dir() -> Path:
    """Per-user configuration directory: XDG on POSIX, ``%APPDATA%`` on Windows."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "inflorescence"


def user_env_file() -> Path:
    """The dotenv file ``init`` writes, readable from any working directory."""
    return user_config_dir() / ".env"


def feedback_file() -> Path:
    """Where ``send_feedback`` appends agent-written notes about using this server.

    Next to the user's ``.env`` rather than inside any repository: feedback is about the
    server, not about the project that happened to be open, and every session — whatever
    directory its client started the process in — must append to the same log. Overridable
    with ``FEEDBACK_PATH``.
    """
    return user_config_dir() / "feedback.jsonl"

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".worktrees",
    ".playwright-mcp",
    ".next",
    "dist",
    "build",
    "coverage",
    ".tox",
    "*.egg-info",
    ".beads",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "target",
    "vendor",
    ".idea",
    ".vscode",
    "out",
    ".cache",
]

# Files indexed as whole documents: one node per file, body chunked with overlap, no
# structure parsing (see parser/document_parser.py). Prose was the original case; styles,
# markup and configuration joined it because they answer the same *kind* of question and
# no parser here understands them either. A frontend bug lived in
# `@media (prefers-reduced-motion)` inside index.css and was unfindable — not ranked low,
# absent — because no scan ever opened the file. A cascade rule, a `<meta>` tag and a
# `vite.config.ts`'s sibling `tsconfig.json` are half of what a frontend question is about.
#
# They are NOT parsed: CSS has no call graph and a JSON object is not a class, so inventing
# symbols above them would produce nodes no edge points at. One node, chunked body, BM25 and
# chunk search over the text — which is exactly what finds a media query by its own words.
#
# The cost of admitting data formats is generated bulk, and it is bounded three ways:
# lock files are dropped by name (LOCKFILE_NAMES), anything over MAX_FILE_SIZE_BYTES lands
# in `skipped_large_files` instead of being summarized, and .gitignore/exclude patterns
# already keep node_modules, dist and build output out of the scan.
DOCUMENT_EXTENSION_GROUPS: dict[str, list[str]] = {
    "prose": [".md", ".markdown", ".mdx", ".rst", ".txt", ".text", ".adoc", ".asciidoc", ".org"],
    # styles — hand-written, prose-free, and where layout/theme/media-query answers live
    "stylesheets": [".css", ".scss", ".sass", ".less", ".styl"],
    # markup — index.html carries the meta tags, the mount point and the script tags
    "markup": [".html", ".htm"],
    # configuration and build/tool settings
    "configuration": [".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"],
    # interface and query definitions — the contract a service is built against. A field
    # report (14.09.2026) asked "how does search work" of a Go backend whose gRPC contracts
    # lived in .proto files and whose actual search SQL (full-text match, filters, ORDER BY)
    # lived in sqlc query files: neither was indexed, so the semantic search landed on the Go
    # wrappers and never on the query itself. No parser here builds symbols from them, so they
    # are documents: found by their own text and by meaning, which is what that question needed.
    "schemas and queries": [".proto", ".sql", ".graphql", ".gql"],
    # shell and operator scripts — the runbook layer. A field report (10.09.2026) traced a
    # whole failed research task to their absence: the task was "explain how a batch run
    # is launched", and the exact invocation, every flag, and the reason for each deviation
    # from the default recipe lived in a 38-line `scripts/nightly_run.sh` — including the
    # mandatory-flag warning, which was a bash *comment*. The graph held 105 Python modules
    # from that directory and not one of its nine shell scripts, so no query could reach
    # them: not ranked low, absent. Teams put operational truth in these files precisely
    # because they are executable; that makes them documentation, not noise.
    "shell and operator scripts": [".sh", ".bash", ".zsh", ".fish", ".ps1"],
}
DEFAULT_DOCUMENT_EXTENSIONS: list[str] = [
    extension for group in DOCUMENT_EXTENSION_GROUPS.values() for extension in group
]

# Documents admitted by **exact filename** rather than by extension: the build and runbook
# files that carry no suffix at all (``Makefile``, ``Dockerfile``) or whose suffix says
# nothing useful (``.env.example`` → ``.example``). Same treatment as any other document —
# one node, chunked body, no parsing.
#
# Exact names only, and never a bare ``.env``: a template committed for humans to copy
# (``.env.example`` and its siblings) documents which variables exist, while ``.env`` itself
# holds the filled-in secrets. Prefix matching would erase that distinction, so a variant
# this list does not name (``Dockerfile.prod``, ``.env.local``) stays out until someone adds
# it deliberately — via DOCUMENT_FILENAMES, which replaces this list wholesale.
DEFAULT_DOCUMENT_FILENAMES: list[str] = [
    "Makefile",
    "makefile",
    "GNUmakefile",
    "Dockerfile",
    "Containerfile",
    "Justfile",
    "justfile",
    "Procfile",
    "Rakefile",
    "Brewfile",
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
]

# Dependency lock files, dropped from every scan **by name**, whatever the extension set or
# the include patterns say. They are machine-written transcripts of a resolver's output:
# thousands of lines that mention every package name in the tree, so they win keyword
# searches they have no business winning, and they change on every install. `.lock` / `.sum`
# are not indexable extensions today, so only a few of these can currently match — the rest
# are listed so that admitting a new extension never silently admits its lock file too.
LOCKFILE_NAMES: frozenset[str] = frozenset({
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lock",
    "bun.lockb",
    "deno.lock",
    "poetry.lock",
    "pdm.lock",
    "uv.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
    "packages.lock.json",
})

GENERATED_FILE_PATTERNS: list[str] = [
    # protobuf / gRPC (many of these lack an in-file marker, esp. JS/TS/Python)
    "*.pb.go", "*.pb.gw.go", "*_grpc.pb.go", "*_pb.go",           # Go
    "*_pb2.py", "*_pb2_grpc.py",                                  # Python
    "*_pb.js", "*_pb.d.ts", "*_grpc_pb.js", "*_grpc_pb.d.ts",     # JS/TS grpc-web / grpc-js
    "*.pb.ts", "*_pb.ts", "*_connect.ts", "*_connectweb.ts",      # ts-proto / connect
    "*.pb.cc", "*.pb.h",                                          # C/C++
    # common build/codegen output
    "*.min.js", "*.bundle.js", "*.min.css", "*.bundle.css",
]


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    def __init__(self, **kwargs: Any) -> None:
        # Two dotenv sources, in ascending precedence: the per-user file `init` writes, then a
        # project-local .env. Real environment variables still outrank both.
        #
        # The user-level file is what makes an MCP-launched server work at all. A client starts
        # this process with its own working directory — not the one the key was configured in —
        # so a key that lived only in a project .env produced a server that came up clean and
        # then failed every LLM call with 401. Resolved per instantiation rather than on the
        # class so that HOME/XDG_CONFIG_HOME changes (tests, sandboxes) are honored.
        kwargs.setdefault("_env_file", (str(user_env_file()), ".env"))
        super().__init__(**kwargs)

    # Memgraph
    memgraph_url: str = "bolt://localhost:7687"
    memgraph_autostart: bool = True
    memgraph_user: str | None = None
    memgraph_password: str | None = None

    # LLM (OpenRouter)
    llm_api_key: str = Field(alias="OPENROUTER_API_KEY", default="")
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-2.5-flash-lite"

    # Embeddings
    code_embedding_model: str = "openai/text-embedding-3-small"
    summary_embedding_model: str = "openai/text-embedding-3-small"
    embedding_dimension: int = 1536
    # Code embeddings may use a different provider than the LLM. A code-specialized
    # model (e.g. Mistral `codestral-embed`, best on code-retrieval benchmarks) is not
    # served by OpenRouter, so it needs its own OpenAI-compatible endpoint + key. Both
    # default to the LLM endpoint (llm_base_url / llm_api_key) — leave unset to keep the
    # current behavior. If you switch models, keep embedding_dimension in sync (codestral
    # -embed defaults to 1536, matching the summary model) and re-index the project.
    code_embedding_base_url: str | None = None
    code_embedding_api_key: str | None = None
    # Below this cosine similarity a vector hit is reported with "weak": true, and a page
    # where *every* hit is weak also gets meta.weak_results (search_code / search_semantic /
    # search_hybrid). Vectors cannot prove absence — a top-k search always returns its k —
    # so this does not filter anything (that is `min_score`, a hard cut-off the caller
    # chooses per query); it only stops a page of far-away neighbours from reading like a
    # page of answers. 0.5 is the line for the default `text-embedding-3-small`: unrelated
    # English text pairs land in ~0.0–0.35 with that model, queries about code the repository
    # does not contain were measured at 0.43–0.46, and genuine matches sit at 0.6–0.9. Raise
    # it for a model whose similarities run hot (many older models floor around 0.7), lower
    # it if real matches start arriving flagged.
    semantic_weak_threshold: float = 0.5

    # Chunking
    chunk_size: int = 3000
    chunk_overlap: int = 1000
    # Documents (.md/.txt/...) are chunked as overlapping sentence windows over the whole
    # file rather than by structure, so these are the only knobs that decide their
    # retrieval granularity. Smaller than the code window on purpose: a prose window that
    # spans several sections embeds into a vector that matches nothing precisely.
    document_chunk_size: int = 800
    document_chunk_overlap: int = 200
    # RAG streaming window: chunks/summaries are embedded and stored this many at a
    # time so peak embedding-vector memory stays bounded and each window persists.
    rag_index_batch_size: int = 256

    # Indexing
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_EXCLUDE_PATTERNS))
    batch_size: int = 10
    respect_gitignore: bool = True
    max_file_size_bytes: int = 262144
    skip_generated: bool = True
    use_llm_summaries: bool = True
    # Index non-code files (README and docs/**.md, but also stylesheets, HTML,
    # JSON/YAML/TOML/INI configuration, shell scripts and Makefile/Dockerfile) alongside
    # code. They are the part of a repository that says *why*, *with what settings* and
    # *how it is actually run*, which no amount of AST parsing recovers — but they are also
    # summarized and embedded like code, so this is an off switch for a repo where they are
    # huge, generated, or not worth the spend. Narrow DOCUMENT_EXTENSIONS /
    # DOCUMENT_FILENAMES instead of turning this off to keep only some of them.
    index_documents: bool = True
    document_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_DOCUMENT_EXTENSIONS))
    # Documents matched by exact filename (see DEFAULT_DOCUMENT_FILENAMES). Setting this
    # replaces the default list rather than adding to it, same as DOCUMENT_EXTENSIONS.
    document_filenames: list[str] = Field(default_factory=lambda: list(DEFAULT_DOCUMENT_FILENAMES))
    # Rows per UNWIND write to Memgraph (nodes/edges/chunks/summaries/checksums).
    db_write_batch_size: int = 500

    # Summarization
    summary_max_source_chars: int = 12000

    # Cost guard (indexing). Every run that would spend is gated by this cap (INV-10: money is
    # spent consciously, and background must not silently outspend an explicit index): a
    # first-time index priced over the whole tree, and both a re-index and a background watcher
    # update priced over just the files whose content moved. Re-indexes are gated because an
    # indexer change — one new DOCUMENT_EXTENSIONS entry — can make thousands of files newly
    # billable behind a call that reads like a cheap refresh. Default $10 clears normal and even large repos (a full index
    # of a ~200-file Go service measured well under $1) while stopping a runaway monorepo, or a
    # thousand-file `git checkout` flush, cold. Raise it (env MAX_INDEX_COST_USD / --max-cost / max_cost_usd),
    # or set it empty (or null) to disable the cap entirely. Prices default to the v0.1 default models
    # (gemini-2.5-flash-lite + text-embedding-3-small); update if you switch.
    max_index_cost_usd: float | None = 10.0

    @field_validator("max_index_cost_usd", mode="before")
    @classmethod
    def _empty_cap_means_no_cap(cls, value: object) -> object:
        """``MAX_INDEX_COST_USD=`` (empty), ``null`` or ``none`` disables the cap, as the docs say.

        An environment variable cannot hold a JSON null, so without this the documented way to
        turn the cap off was a validation error at startup.
        """
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
            return None
        return value
    llm_price_input_usd_per_mtok: float = 0.25
    llm_price_output_usd_per_mtok: float = 1.50
    embedding_price_usd_per_mtok: float = 0.02

    # Search circuit (INV-2). The vector indexes that make embeddings searchable require the
    # Memgraph MAGE image; on a plain Memgraph their creation fails. When True (default) an
    # index/update aborts before spending on embeddings that could never be searched, instead
    # of paying and pretending search works. Set False to index anyway on a search-less DB
    # (graph-only use); the missing indexes are still logged at ERROR.
    require_search_indexes: bool = True

    # Semantic CALLS enrichment via SCIP indexers (scip-go / scip-python / scip-typescript).
    # These are external binaries the user installs separately; a missing one degrades that
    # language to the syntactic ladder rather than failing the run. Exposed as settings because
    # the SCIP pass shells out to a real build toolchain inside the repository being indexed,
    # which is not something a user can be expected to accept without an off switch.
    use_scip_semantic: bool = True
    scip_timeout_seconds: int = 600
    # How an indexer is obtained. "auto" prefers a binary on PATH and falls back to a
    # container image, which is what lets SCIP work on a machine where nothing was installed
    # by hand: the image carries the Go and Node toolchains the indexers need, and the
    # repository is mounted read-only, so a foreign build configuration never runs against the
    # host. "native" and "docker" pin one rung; either way, failure degrades to the ladder.
    scip_runner: str = "auto"
    # language -> image override, e.g. {"go": "ghcr.io/me/scip-go:pinned"}
    scip_images: dict[str, str] = Field(default_factory=dict)

    # Git worktrees. A linked worktree is indexed as its own project (its checkout is a
    # different tree), but nearly all of its content is byte-identical to the main checkout's,
    # so the main project's index is a free cache donor: summaries are reused by input hash and
    # chunk vectors by content-derived id, which makes the first index of a worktree cost ~$0
    # for everything the branch did not change (INV-10 — money is spent on new content only).
    # Set False to index every worktree from scratch, paying for content that is already indexed.
    worktree_auto_index: bool = True
    # How that first index is produced. True (default): the donor's graph is copied inside
    # Memgraph — summaries and vectors never leave the database — and only the files the
    # branch actually differs on are re-parsed and re-embedded, which turns minutes of
    # client-side re-writing into a copy plus a checksum scan. False falls back to the
    # original path: parse the whole tree in the client and write every node, chunk and
    # vector back over Bolt, reusing the donor's caches by content hash. Both cost the same
    # (nothing) in LLM and embedding calls; only wall time differs. The fallback is also
    # automatic — a clone that fails or fails verification is deleted and the client-side
    # path runs instead, so this switch exists for diagnosis, not for correctness.
    worktree_clone: bool = True
    # Reap the index of a worktree git no longer knows about. A worktree is a disposable
    # checkout: it is deleted far more often than a project is, and its index would otherwise
    # linger forever, answering queries about a tree that no longer exists. Only projects
    # carrying the worktree marker written at index time are ever considered — an ordinary
    # project is never a candidate, whatever this is set to.
    worktree_janitor_enabled: bool = True

    # Watcher
    watcher_debounce_seconds: float = 5.0
    # How often the watch daemon re-reads the registry (`inflorescence watch add/rm` writes it
    # from another process, so this is the delay before a newly added project is followed) and
    # re-checks that every root it watches is still on disk. Cheap: one row per indexed project.
    watch_refresh_seconds: float = 30.0
    # Whether the daemon reconciles a project against disk when it adopts it. This is what
    # makes "the watcher was down all night" recoverable rather than permanent: without it the
    # index keeps whatever staleness it accumulated while nothing was watching, and only files
    # touched *after* the daemon started are ever repaired. The catch-up is an ordinary
    # incremental re-index, so MAX_INDEX_COST_USD gates it exactly like a manual one.
    watch_catch_up: bool = True

    # Feedback. Where the send_feedback MCP tool appends what consuming agents report about
    # using this server (JSONL, append-only, read back with `inflorescence feedback`).
    # Unset means the per-user default, config.feedback_file() — deliberately a plain file
    # and not a Memgraph node: the reports worth having most are the ones written while the
    # database is unreachable.
    feedback_path: str | None = None

    # Logging
    log_level: str = "INFO"
