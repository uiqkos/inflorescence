"""Configuration via Pydantic Settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from pydantic import Field
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

# Prose files indexed as whole documents: one node per file, body chunked with overlap,
# no structure parsing (see parser/document_parser.py). Deliberately excludes data/config
# formats (.json, .yaml, .csv): those carry no prose an LLM summary or a semantic query can
# use, and a large lockfile would be paid for in summaries and embeddings for nothing.
DEFAULT_DOCUMENT_EXTENSIONS: list[str] = [
    ".md",
    ".markdown",
    ".mdx",
    ".rst",
    ".txt",
    ".text",
    ".adoc",
    ".asciidoc",
    ".org",
]

GENERATED_FILE_PATTERNS: list[str] = [
    # protobuf / gRPC (many of these lack an in-file marker, esp. JS/TS/Python)
    "*.pb.go", "*.pb.gw.go", "*_grpc.pb.go", "*_pb.go",           # Go
    "*_pb2.py", "*_pb2_grpc.py",                                  # Python
    "*_pb.js", "*_pb.d.ts", "*_grpc_pb.js", "*_grpc_pb.d.ts",     # JS/TS grpc-web / grpc-js
    "*.pb.ts", "*_pb.ts", "*_connect.ts", "*_connectweb.ts",      # ts-proto / connect
    "*.pb.cc", "*.pb.h",                                          # C/C++
    # common build/codegen output
    "*.min.js", "*.bundle.js",
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
    # Index prose documents (README, docs/**.md, notes.txt) alongside code. They are the
    # part of a repository that says *why*, which no amount of AST parsing recovers — but
    # they are also summarized and embedded like code, so this is an off switch for a repo
    # where documentation is huge, generated, or not worth the spend.
    index_documents: bool = True
    document_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_DOCUMENT_EXTENSIONS))
    # Rows per UNWIND write to Memgraph (nodes/edges/chunks/summaries/checksums).
    db_write_batch_size: int = 500

    # Summarization
    summary_max_source_chars: int = 12000

    # Cost guard (indexing). A first-time index AND every background watcher update are
    # gated by this cap (INV-10: money is spent consciously, and background must not silently
    # outspend an explicit index). Default $10 clears normal and even large repos (a full index
    # of a ~200-file Go service measured well under $1) while stopping a runaway monorepo, or a
    # thousand-file `git checkout` flush, cold. Raise it (env MAX_INDEX_COST_USD / --max-cost / max_cost_usd),
    # or set to null to disable the cap entirely. Prices default to the v0.1 default models
    # (gemini-2.5-flash-lite + text-embedding-3-small); update if you switch.
    max_index_cost_usd: float | None = 10.0
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

    # Watcher
    watcher_debounce_seconds: float = 5.0

    # Logging
    log_level: str = "INFO"
