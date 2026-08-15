"""Semantic CALLS enrichment via SCIP indexers (scip-go, scip-python, scip-typescript).

The syntactic ladder in resolver.py cannot bind calls through typed locals, interface
dispatch or metaprogramming — a compiler can. SCIP (Sourcegraph Code Intelligence
Protocol) is the unifying format: one adapter here consumes the output of any language's
indexer. The pass is strictly *best-effort*: a missing binary, a project that doesn't
build, or a timeout degrades that language back to the heuristic ladder — it never
fails or blocks the run (INV-3-style resilience; measured behavior in
docs/calls-resolution.md).

The .scip file is protobuf; only four field paths matter (Index.documents ->
Document.relative_path/occurrences -> Occurrence.range/symbol/symbol_roles), so a
~60-line wire-format reader below avoids a protobuf runtime dependency entirely.
Field numbers come from scip.proto (v0.3.x, stable since 2022).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from inflorescence.code_indexer.models import CallResolution, CodeNode, Edge, EdgeType, IndexerConfig
from inflorescence.docker_env import docker_cli_present

logger = logging.getLogger(__name__)

# Directories never worth descending into while looking for module roots.
_SKIP_DIRS = {".git", "node_modules", "vendor", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
_MAX_ROOTS_PER_LANGUAGE = 5
_CALLABLE_NODE_TYPES = ("function", "method")
_TARGET_NODE_TYPES = ("function", "method", "class", "struct")

# Default indexer invocations; "{output}" is replaced with the index path.
_DEFAULT_COMMANDS: dict[str, list[str]] = {
    "go": ["scip-go", "--output", "{output}"],
    # --project-version is not optional in practice: scip-python derives it from git and
    # crashes in symbol emission when that yields undefined — a directory that is not a git
    # checkout, or a container that cannot read the repository's git metadata. The value only
    # decorates symbol strings, which this adapter matches by path and name, so a constant is
    # correct as well as safe.
    "python": [
        "scip-python", "index", ".",
        "--project-name", "inflorescence-index",
        "--project-version", "0.0.0",
        "--output", "{output}",
    ],
    "typescript": ["scip-typescript", "index", "--infer-tsconfig", "--output", "{output}"],
}
_PROVENANCE = {"go": "scip-go", "python": "scip-python", "typescript": "scip-typescript"}

# Container images carrying the indexers *and* the toolchains they drive. Neither indexer is
# a self-contained binary: scip-go loads packages through `go list`, and both scip-typescript
# and scip-python are Node programs — so the image must ship the toolchain, not just the
# executable. Split by toolchain rather than by language, so indexing a Python repository
# never pulls the Go image.
#
# Two forms of the same reference. The tag is what `inflorescence build-images` creates and
# what the documentation shows; the digest pin is what gets pulled when nothing local
# exists — a tag can be silently re-pointed in the registry, a digest cannot. The digests
# are the manifest-list digests of the CI build (.github/workflows/images.yml); after
# rebuilding there, update them from the workflow's "Print the digest to pin" step, or with
# `docker buildx imagetools inspect ghcr.io/uiqkos/inflorescence-scip-go:v1`.
_SCIP_GO_IMAGE_TAG = "ghcr.io/uiqkos/inflorescence-scip-go:v1"
_SCIP_NODE_IMAGE_TAG = "ghcr.io/uiqkos/inflorescence-scip-node:v1"
_SCIP_GO_IMAGE = "ghcr.io/uiqkos/inflorescence-scip-go@sha256:f07cb6a53b249e88845a5dbe2efbeca01213ea1d643c63d56364bc13e6961955"
_SCIP_NODE_IMAGE = "ghcr.io/uiqkos/inflorescence-scip-node@sha256:c7ba7e55213574560f014d29b4d2e3b073e78f1d078c60ca5fbb83da5a9f3737"
_DEFAULT_IMAGE_TAGS: dict[str, str] = {
    "go": _SCIP_GO_IMAGE_TAG,
    "python": _SCIP_NODE_IMAGE_TAG,
    "typescript": _SCIP_NODE_IMAGE_TAG,
}
_DEFAULT_IMAGES: dict[str, str] = {
    "go": _SCIP_GO_IMAGE,
    "python": _SCIP_NODE_IMAGE,
    "typescript": _SCIP_NODE_IMAGE,
}

# Paths inside the container. The repository is mounted read-only and the index is written to
# a separate writable mount, so an indexer can never modify the code it was pointed at.
_CONTAINER_REPO = "/repo"
_CONTAINER_OUT = "/out"
_CONTAINER_OUTPUT_FILE = f"{_CONTAINER_OUT}/index.scip"

# Toolchain caches survive the container in a named volume. Measured on a 144-file Go service:
# 25.3s per run with an ephemeral cache against 1.7s native, because every run re-resolved and
# re-compiled the module graph from scratch. The watcher re-runs this pass on every batch of
# edits, so a cold cache is not a one-time cost.
_CONTAINER_CACHE = "/cache"
_SCIP_CACHE_VOLUME = "inflorescence_scip_cache"

# With the repository mounted read-only, a toolchain that defaults to writing beside the
# sources (GOCACHE, npm) would fail on its first write; point every cache at the volume.
_CONTAINER_ENV = {
    "HOME": "/tmp",
    "XDG_CACHE_HOME": f"{_CONTAINER_CACHE}/xdg",
    "GOMODCACHE": f"{_CONTAINER_CACHE}/go/mod",
    "GOCACHE": f"{_CONTAINER_CACHE}/go/build",
    "npm_config_cache": f"{_CONTAINER_CACHE}/npm",
}

# Which rung "auto" tries first, per language. Measured on real repositories:
#
#   Go          identical edges either way (866 on a 144-file service); container 0.7s warm
#               against 1.7s native, but a cold first run costs an image pull — native first.
#   TypeScript  identical edges (299); native 6.3s against 11.4s, and the container cache does
#               not help because the cost is type-checking, not resolution — native first.
#   Python      container is a strict superset: 1786 edges against 1202 on the same repository,
#               with nothing lost. A native scip-python resolves imports against whatever
#               interpreter environment it finds, so in a project installed into its own venv
#               it binds intra-project calls to the site-packages copy — an external symbol —
#               and drops roughly a third of the CALLS edges. The container has no such
#               environment and resolves against the repository. It is also marginally faster.
_AUTO_ORDER: dict[str, tuple[str, ...]] = {
    "go": ("native", "docker"),
    "typescript": ("native", "docker"),
    "python": ("docker", "native"),
}


# ---------------------------------------------------------------------------
# Minimal protobuf wire-format reader for SCIP
# ---------------------------------------------------------------------------


@dataclass
class ScipOccurrence:
    start_line: int  # 0-based
    symbol: str
    roles: int


@dataclass
class ScipDocument:
    relative_path: str
    occurrences: list[ScipOccurrence] = field(default_factory=list)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    """Yield (field_number, wire_type, value) over a protobuf message body.

    value is an int for varint fields and a bytes slice for length-delimited ones
    (callers dispatch on wire_type); fixed32/fixed64 are skipped (SCIP uses neither
    in the fields we read).
    """
    i = 0
    n = len(buf)
    while i < n:
        key, i = _read_varint(buf, i)
        field_no, wire_type = key >> 3, key & 0x7
        if wire_type == 0:  # varint
            value, i = _read_varint(buf, i)
            yield field_no, wire_type, value
        elif wire_type == 2:  # length-delimited
            length, i = _read_varint(buf, i)
            yield field_no, wire_type, buf[i : i + length]
            i += length
        elif wire_type == 5:  # fixed32
            i += 4
        elif wire_type == 1:  # fixed64
            i += 8
        else:  # groups (3/4) don't occur in scip.proto
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def _decode_packed_int32(buf: bytes) -> list[int]:
    values = []
    i = 0
    while i < len(buf):
        v, i = _read_varint(buf, i)
        values.append(v)
    return values


def _parse_occurrence(buf: bytes) -> ScipOccurrence | None:
    start_line = -1
    symbol = ""
    roles = 0
    for field_no, wire_type, value in _iter_fields(buf):
        if field_no == 1 and wire_type == 2:  # range: packed [line, sc, (el,) ec]
            rng = _decode_packed_int32(value)
            if rng:
                start_line = rng[0]
        elif field_no == 2 and wire_type == 2:  # symbol
            symbol = value.decode("utf-8", errors="replace")
        elif field_no == 3 and wire_type == 0:  # symbol_roles
            roles = value
        elif field_no in (8, 9) and wire_type == 2:  # single_/multi_line_range {(start_)line=1, ...}
            for f2, w2, v2 in _iter_fields(value):
                if f2 == 1 and w2 == 0:
                    start_line = v2
    if start_line < 0 or not symbol:
        return None
    return ScipOccurrence(start_line=start_line, symbol=symbol, roles=roles)


def _parse_document(buf: bytes) -> ScipDocument:
    doc = ScipDocument(relative_path="")
    for field_no, wire_type, value in _iter_fields(buf):
        if field_no == 1 and wire_type == 2:  # relative_path
            doc.relative_path = value.decode("utf-8", errors="replace")
        elif field_no == 2 and wire_type == 2:  # occurrence
            occ = _parse_occurrence(value)
            if occ is not None:
                doc.occurrences.append(occ)
    return doc


def parse_scip_index(data: bytes) -> list[ScipDocument]:
    """Parse a .scip file into documents with (line, symbol, roles) occurrences."""
    docs: list[ScipDocument] = []
    for field_no, wire_type, value in _iter_fields(data):
        if field_no == 2 and wire_type == 2:  # Index.documents
            doc = _parse_document(value)
            if doc.relative_path:
                docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Indexer discovery and execution (preflight + degradation)
# ---------------------------------------------------------------------------


def find_scip_roots(root: Path, language: str, files_by_language: dict[str, list[str]]) -> list[Path]:
    """Directories to run the language's indexer in (go.mod / tsconfig / project root)."""
    if not files_by_language.get(language):
        return []
    if language == "python":
        return [root]

    marker_names = ("go.mod",) if language == "go" else ("tsconfig.json", "package.json")
    markers: list[Path] = []
    stack = [root]
    while stack and len(markers) < _MAX_ROOTS_PER_LANGUAGE:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        found_marker = False
        for name in marker_names:
            if (current / name).is_file():
                markers.append(current)
                found_marker = True
                break
        if found_marker and current != root:
            continue  # nested modules under a found root are covered by that run
        for entry in entries:
            if entry.is_dir() and entry.name not in _SKIP_DIRS and not entry.name.startswith("."):
                stack.append(entry)
    return markers[:_MAX_ROOTS_PER_LANGUAGE]


def _expand_output(command: list[str], output: str) -> list[str]:
    """Substitute the output placeholder, appending `--output` when the command has none."""
    argv = [arg.replace("{output}", output) for arg in command]
    if "{output}" not in "".join(command):
        argv += ["--output", output]
    return argv


def _local_image_present(image: str) -> bool:
    """True when the local docker daemon already holds *image*. Never contacts a registry."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _prefer_local_tag(language: str, image: str) -> str:
    """Resolve the default reference for *language*: a locally present tag wins over the pin.

    A digest reference always names the registry's bytes — `docker run` on it ignores any
    locally built image and pulls, which is the opposite of what `inflorescence build-images`
    promises. So when the default pin is a digest, the human-friendly tag is consulted on the
    local daemon first (inspect only — an absent tag is not pulled, it falls through to the
    pinned digest). An explicit `SCIP_IMAGES` override is used verbatim, never rewritten.
    """
    if image != _DEFAULT_IMAGES.get(language):
        return image
    tag = _DEFAULT_IMAGE_TAGS.get(language)
    if tag and tag != image and _local_image_present(tag):
        return tag
    return image


def _docker_user() -> str | None:
    """Run the container as the invoking user, so the index it writes is not root-owned."""
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return None
    return f"{getuid()}:{getgid()}"


def docker_scip_argv(
    image: str,
    scip_root: Path,
    host_output_dir: Path,
    command: list[str],
    *,
    user: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """A ``docker run`` that indexes *scip_root* and leaves the index in *host_output_dir*.

    The repository is mounted read-only and nothing else from the host is exposed. This is
    the containment the native rung cannot offer: a SCIP indexer necessarily executes the
    indexed project's own build configuration, and that configuration is not the user's code
    in the case that matters — someone else's repository, opened to read it.
    """
    argv = [
        "docker", "run", "--rm",
        "-v", f"{scip_root}:{_CONTAINER_REPO}:ro",
        "-v", f"{host_output_dir}:{_CONTAINER_OUT}",
        "-v", f"{_SCIP_CACHE_VOLUME}:{_CONTAINER_CACHE}",
        "-w", _CONTAINER_REPO,
    ]
    if user:
        argv += ["--user", user]
    for key, value in {**_CONTAINER_ENV, **(extra_env or {})}.items():
        argv += ["-e", f"{key}={value}"]
    argv.append(image)
    return argv + _expand_output(command, _CONTAINER_OUTPUT_FILE)


@dataclass
class ScipInvocation:
    """How one indexer run will be executed, once a rung has been chosen."""

    argv: list[str]
    cwd: Path | None  # None for docker, which sets its working directory inside the container
    runner: str  # "native" | "docker"
    output: Path  # host path where the .scip file is expected to appear


def resolve_scip_invocation(
    language: str,
    scip_root: Path,
    host_output_dir: Path,
    config: IndexerConfig,
) -> ScipInvocation | None:
    """Pick a rung for *language*: an installed binary, else a container. None if neither.

    Native first — it has no container start-up cost and no image to pull. Docker exists so
    that a machine where nothing was installed by hand still gets compiler-grade edges.
    """
    override = config.scip_commands.get(language)
    command = shlex.split(override) if override else list(_DEFAULT_COMMANDS.get(language, ()))
    if not command:
        return None

    runner = (config.scip_runner or "auto").lower()
    output = host_output_dir / "index.scip"
    if runner == "native":
        order: tuple[str, ...] = ("native",)
    elif runner == "docker":
        order = ("docker",)
    else:
        order = _AUTO_ORDER.get(language, ("native", "docker"))

    for rung in order:
        if rung == "native" and shutil.which(command[0]) is not None:
            return ScipInvocation(_expand_output(command, str(output)), scip_root, "native", output)
        if rung == "docker":
            image = config.scip_images.get(language) or _DEFAULT_IMAGES.get(language)
            if image and docker_cli_present():
                image = _prefer_local_tag(language, image)
                argv = docker_scip_argv(
                    image, scip_root, host_output_dir, command,
                    user=_docker_user(), extra_env=config.scip_env,
                )
                return ScipInvocation(argv, None, "docker", output)

    logger.debug(
        "No SCIP rung available for %s (runner=%s, %r not on PATH, no usable image) — "
        "staying on the heuristic ladder", language, runner, command[0],
    )
    return None


def run_scip_indexer(
    language: str,
    scip_root: Path,
    config: IndexerConfig,
) -> list[ScipDocument] | None:
    """Run the indexer for *language* in *scip_root*; None on any failure (degrade)."""
    with tempfile.TemporaryDirectory(prefix="scip-") as tmp:
        invocation = resolve_scip_invocation(language, scip_root, Path(tmp), config)
        if invocation is None:
            return None

        env = None
        if config.scip_env and invocation.runner == "native":
            env = {**os.environ, **config.scip_env}  # docker carries these as -e instead
        logger.info(
            "SCIP %s via %s in %s%s", language, invocation.runner, scip_root,
            " (first run may pull the image)" if invocation.runner == "docker" else "",
        )
        try:
            proc = subprocess.run(  # noqa: S603 — command comes from config/defaults, not user input
                invocation.argv,
                cwd=invocation.cwd,
                env=env,
                capture_output=True,
                timeout=config.scip_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "SCIP %s indexer (%s) failed to run in %s (%s) — degrading to heuristic",
                language, invocation.runner, scip_root, exc,
            )
            return None
        if proc.returncode != 0 or not invocation.output.is_file():
            tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
            logger.warning(
                "SCIP %s indexer (%s) exited %d in %s — degrading to heuristic. stderr tail: %s",
                language, invocation.runner, proc.returncode, scip_root, tail,
            )
            return None
        try:
            return parse_scip_index(invocation.output.read_bytes())
        except (ValueError, IndexError) as exc:
            logger.warning("Unparseable SCIP index from %s (%s) — degrading to heuristic", language, exc)
            return None


def indexer_availability(config: IndexerConfig) -> dict[str, str]:
    """language -> "native" / "docker" / "none", for `doctor` to report before an index runs."""
    availability: dict[str, str] = {}
    for language in _DEFAULT_COMMANDS:
        invocation = resolve_scip_invocation(language, Path("."), Path("."), config)
        availability[language] = invocation.runner if invocation else "none"
    return availability


@dataclass
class ImageStatus:
    """One container image the docker rung would run, for `doctor` to show.

    `local` is the line that answers "what exactly will execute on my machine": a locally
    present reference is used by `docker run` with no registry contact at all, an absent
    one is pulled once on first use.
    """

    image: str
    languages: list[str]
    local: bool


def image_availability(config: IndexerConfig) -> list[ImageStatus]:
    """The images the docker rung would actually run, deduplicated, with local presence.

    Only languages whose chosen rung is "docker" contribute — a native binary on PATH or a
    wholly unavailable rung says nothing about images. The reference reported is the one
    `docker run` would receive, after the same local-tag preference the real run applies.
    """
    statuses: dict[str, ImageStatus] = {}
    for language, how in indexer_availability(config).items():
        if how != "docker":
            continue
        image = config.scip_images.get(language) or _DEFAULT_IMAGES.get(language)
        if not image:
            continue
        chosen = _prefer_local_tag(language, image)
        if chosen in statuses:
            statuses[chosen].languages.append(language)
        else:
            statuses[chosen] = ImageStatus(image=chosen, languages=[language], local=_local_image_present(chosen))
    return list(statuses.values())


# ---------------------------------------------------------------------------
# Adapter: SCIP occurrences -> CALLS edges over existing nodes
# ---------------------------------------------------------------------------


@dataclass
class SemanticCalls:
    edges: list[Edge]
    external_calls: dict[str, list[str]]
    covered_files: set[str]
    provenance: dict[str, str]  # file_path -> "scip-go" | ...


def _short_external(symbol: str) -> str:
    """Compress a SCIP symbol to a short human name for external_calls.

    '… gomod go1.25 fmt/Errorf().' -> 'fmt.Errorf';
    '… `github.com/jackc/pgx/v5/pgxpool`/Pool#Exec().' -> 'pgxpool.Pool.Exec'.
    A SCIP symbol is '<scheme> <manager> <package> <version> <descriptors>'; only the
    descriptors (after the last space) matter here.
    """
    tail = symbol.rsplit(" ", 1)[-1]
    if tail.startswith("`") and "`" in tail[1:]:
        pkg, _, rest = tail[1:].partition("`")
        pkg_short = pkg.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
        rest = rest.lstrip("/")
    else:
        parts = tail.rsplit("/", 2)
        pkg_short = parts[-2] if len(parts) >= 2 else ""
        rest = parts[-1]
    name = f"{pkg_short}.{rest}" if pkg_short else rest
    name = name.replace("#", ".").replace("().", "").replace("()", "").rstrip(".")
    return name[:200]


def derive_semantic_calls(
    docs_with_prefix: list[tuple[str, list[ScipDocument]]],
    nodes: list[CodeNode],
) -> SemanticCalls:
    """Map SCIP occurrences onto existing nodes and derive CALLS edges.

    Caller = innermost function/method node containing the reference line; references
    outside any function (imports, top-level statements) are skipped — module-level
    calls stay with the heuristic ladder, which extracts them syntactically without
    mistaking import statements for calls.
    """
    known_files = {n.file_path for n in nodes}
    callables_by_file: dict[str, list[CodeNode]] = defaultdict(list)
    targets_by_file: dict[str, list[CodeNode]] = defaultdict(list)
    for n in nodes:
        if n.node_type.value in _CALLABLE_NODE_TYPES:
            callables_by_file[n.file_path].append(n)
        if n.node_type.value in _TARGET_NODE_TYPES:
            targets_by_file[n.file_path].append(n)

    def innermost(pool: dict[str, list[CodeNode]], file: str, line1: int) -> CodeNode | None:
        candidates = [n for n in pool.get(file, ()) if n.line_start <= line1 <= n.line_end]
        return max(candidates, key=lambda n: n.line_start) if candidates else None

    def _binds_as_call_target(symbol: str, kind: str, project_rel: str) -> bool:
        """May *symbol*, defined at a node of *kind*, be a CALLS target?

        '().'-descriptors are functions/methods in every SCIP indexer. Bare type
        references ('Foo#') are calls only in Python (constructor call syntax) — in
        Go/TS a type mention in a signature is not a call. Plain term symbols
        ('api.') bind only when the definition lands on one of our function nodes:
        that is exactly an arrow-function const (TS) — data-shaped terms never map
        to a function node, so constants can't produce noise. SCIP 'local N' symbols
        are DOCUMENT-scoped (the same string names different things in different
        files) and must never bind across documents.
        """
        if symbol.startswith("local "):
            return False
        if symbol.endswith("()."):
            return kind in ("function", "method")
        if symbol.endswith("#"):
            return project_rel.endswith(".py") and kind in ("class", "struct")
        return symbol.endswith(".") and kind in ("function", "method")

    # Pass 1: definitions -> symbol-to-node binding
    sym_to_node: dict[str, str] = {}
    normalized: list[tuple[str, ScipDocument]] = []
    for prefix, docs in docs_with_prefix:
        for doc in docs:
            rel = doc.relative_path
            if rel.startswith("..") or PurePosixPath(rel).is_absolute():
                continue  # go-build cache artifacts etc. — outside the project
            project_rel = str(PurePosixPath(prefix) / rel) if prefix else rel
            if project_rel not in known_files:
                continue
            normalized.append((project_rel, doc))
            for occ in doc.occurrences:
                if not occ.roles & 1 or occ.symbol in sym_to_node:
                    continue
                target = innermost(targets_by_file, project_rel, occ.start_line + 1)
                if target is not None and _binds_as_call_target(occ.symbol, target.node_type.value, project_rel):
                    sym_to_node[occ.symbol] = target.id

    # Pass 2: references -> edges / externals
    edges: dict[tuple[str, str], Edge] = {}
    external: dict[str, set[str]] = defaultdict(set)
    covered: set[str] = set()
    for project_rel, doc in normalized:
        covered.add(project_rel)
        for occ in doc.occurrences:
            if occ.roles & 1:
                continue
            caller = innermost(callables_by_file, project_rel, occ.start_line + 1)
            if caller is None:
                continue
            target_id = sym_to_node.get(occ.symbol)
            if target_id is not None:
                key = (caller.id, target_id)
                if key not in edges and target_id != caller.id:
                    edges[key] = Edge(
                        source=caller.id,
                        target=target_id,
                        edge_type=EdgeType.CALLS,
                        resolution=CallResolution.SEMANTIC.value,
                        callee_text=_short_external(occ.symbol),
                    )
            elif occ.symbol.endswith("().") and not occ.symbol.startswith("local "):
                external[caller.id].add(_short_external(occ.symbol))

    return SemanticCalls(
        edges=list(edges.values()),
        external_calls={k: sorted(v)[:100] for k, v in external.items()},
        covered_files=covered,
        provenance={},
    )


def semantic_calls_pass(
    root: Path,
    nodes: list[CodeNode],
    config: IndexerConfig,
) -> SemanticCalls | None:
    """Run every applicable SCIP indexer and adapt the results. None = nothing ran."""
    files_by_language: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        suffix = PurePosixPath(n.file_path).suffix
        if suffix == ".go":
            files_by_language["go"].append(n.file_path)
        elif suffix == ".py":
            files_by_language["python"].append(n.file_path)
        elif suffix in (".ts", ".tsx", ".js", ".jsx"):
            files_by_language["typescript"].append(n.file_path)

    docs_with_prefix: list[tuple[str, list[ScipDocument]]] = []
    provenance_by_file: dict[str, str] = {}
    ran_languages: dict[str, list[str]] = {}
    for language in ("go", "python", "typescript"):
        for scip_root in find_scip_roots(root, language, files_by_language):
            docs = run_scip_indexer(language, scip_root, config)
            if docs is None:
                continue
            prefix = "" if scip_root == root else str(scip_root.relative_to(root))
            docs_with_prefix.append((prefix, docs))
            ran_languages.setdefault(language, []).append(prefix or ".")
            for doc in docs:
                rel = doc.relative_path
                if rel.startswith(".."):
                    continue
                project_rel = str(PurePosixPath(prefix) / rel) if prefix else rel
                provenance_by_file[project_rel] = _PROVENANCE[language]

    if not docs_with_prefix:
        return None
    logger.info(
        "SCIP semantic pass ran: %s",
        ", ".join(f"{lang}({', '.join(roots)})" for lang, roots in ran_languages.items()),
    )
    result = derive_semantic_calls(docs_with_prefix, nodes)
    result.provenance = {f: p for f, p in provenance_by_file.items() if f in result.covered_files}
    return result
