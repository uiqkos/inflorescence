# Changelog

Notable changes per release. This project follows [semantic versioning](https://semver.org/).

## 0.1.0 — unreleased

First public release.

- **Code graph** — modules, classes, functions and methods parsed from Python (stdlib `ast`),
  JavaScript, TypeScript, Go and Rust (tree-sitter), stored in Memgraph with `CALLS`,
  `IMPORTS`, `INHERITS`, `IMPLEMENTS` and `CONTAINS` relations. A Python file that fails to
  parse falls back to an LLM extractor.
- **Honest call resolution** — `CALLS` edges come from a ladder: compiler-grade SCIP indexers
  where the project builds, a syntactic ladder measured at 100% precision elsewhere, and calls
  that cannot be resolved are recorded on the caller (`external_calls` / `unresolved_calls`)
  instead of being dropped or guessed. See [docs/calls-resolution.md](docs/calls-resolution.md).
- **External dependencies** — packages and base classes the project names but does not contain
  are stored as `:External` nodes rather than dropped at write time.
- **Documents** — non-code files are indexed as `Document` nodes: prose (`.md`, `.rst`, `.txt`,
  `.adoc`, `.org` and friends), stylesheets (`.css`, `.scss`, `.sass`, `.less`, `.styl`),
  markup (`.html`, `.htm`) and configuration (`.json`, `.jsonc`, `.yaml`, `.yml`, `.toml`,
  `.ini`, `.cfg`, `.conf`). One node per file, no structure parsing, body split into
  overlapping chunks and summarized by a prose-specific prompt — so a CSS media query or a
  build-config key is findable by its own text. Code that imports one by path
  (`import './index.css'`) gives it an incoming `IMPORTS` edge. Dependency lock files are
  never indexed; oversized files are reported as `skipped_large_files`. Turn off with
  `INDEX_DOCUMENTS=false`.
- **Answers that say where they came from** — every response carrying entities reports
  `meta.answered_from` (tree path, `branch @ commit`, and the checkout a git worktree belongs
  to) and `meta.index_freshness`, which re-hashes that page's files and flags any that changed
  or vanished since indexing. Vector hits below `SEMANTIC_WEAK_THRESHOLD` are marked
  `"weak": true`, since a top-k search cannot prove absence.
- **Dual RAG plus BM25** — code chunks and LLM summaries embedded into separate HNSW indexes
  and fused per entity; full-text indexes over names, signatures, summaries and raw code.
- **12 MCP tools** — search by meaning, by text, by name and by code similarity; entity
  deep-dives; relation traversal; a read-only Cypher console; and indexing.
- **Live incremental updates** — a file watcher re-indexes on save, reusing unchanged summaries
  and embeddings by content hash, so an edit costs pennies and an interrupted run resumes
  without repeating LLM spend.
- **Cost guard** — a first-time index is gated by a dollar estimate you can cap with
  `--max-cost`.
- **Dashboard** — overview, containment explorer, force-directed graph, Cypher console and an
  MCP tools console at `127.0.0.1:8321`.
- **Setup** — `inflorescence init` writes the config and registers the MCP server;
  `inflorescence doctor` checks config, key, database and schema; Memgraph and the SCIP
  indexers start as containers on demand.
- **Image supply chain** — the SCIP indexer images are reproducible (indexer versions
  pinned in the Dockerfiles) and buildable locally from Dockerfiles shipped inside the
  package: `inflorescence build-images` tags them exactly as the docker rung expects, and a
  locally present tag is used with no registry contact. Published images are built
  multi-arch by CI with build provenance attestations and referenced by immutable digest in
  code (`SCIP_IMAGES` still overrides); `doctor` names the image that will run and whether
  it is already local.
