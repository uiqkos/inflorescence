"""RAG indexer: chunk files -> embed -> store in Memgraph."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from inflorescence.code_indexer.models import CodeNode
from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository
from inflorescence.progress import ProgressReporter, resolve
from inflorescence.rag.chunker import Chunk, Chunker
from inflorescence.rag.config import RAGConfig
from inflorescence.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


def _chunk_id(chunk: Chunk) -> str:
    """Stable content-derived id — identical file content yields an identical id."""
    identity = "\0".join([
        chunk.node_id, chunk.file_path, str(chunk.start_line),
        str(chunk.end_line), chunk.chunk_kind, chunk.content,
    ])
    return f"chunk:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


class RAGIndexer:
    """Chunks source files, embeds them, and stores vectors in Memgraph."""

    def __init__(
        self,
        repo: GraphRepository,
        settings: Settings,
        chunker: Chunker | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings
        self._batch_size = max(1, settings.rag_index_batch_size)
        # Keep the chunker's size limit in lockstep with the graph scanner's, so a
        # file large enough to index still gets code chunks (not just summaries).
        self._chunker = chunker or Chunker(config=RAGConfig(max_file_size_bytes=settings.max_file_size_bytes))
        self._code_embedder = EmbeddingClient(
            settings=settings,
            model=settings.code_embedding_model,
            base_url=settings.code_embedding_base_url,
            api_key=settings.code_embedding_api_key,
        )
        self._summary_embedder = EmbeddingClient(
            settings=settings,
            model=settings.summary_embedding_model,
        )

    def preflight(self) -> None:
        """Verify BOTH embedding endpoints with one tiny call each; raises on auth failure.

        Call before mutating the graph so a keyless/misconfigured server aborts the whole
        index or update instead of silently writing degraded data. The code embedder can
        target a different provider/key than the summary embedder (Mistral codestral-embed);
        checking only the summary side used to let a first index build the graph and pay for
        summaries, then fail every chunk batch on the code endpoint (audit finding 6 / INV-2).
        The two are deduplicated so an identical endpoint is only probed once.
        """
        self._summary_embedder.preflight()
        s = self._settings
        # The code embedder falls back to the summary endpoint (llm_base_url/llm_api_key)
        # unless it was given its own base_url/api_key; when it shares the endpoint and model,
        # the summary probe already covered it, so skip the redundant (billed) call.
        code_shares_endpoint = (
            s.code_embedding_base_url is None
            and s.code_embedding_api_key is None
            and s.code_embedding_model == s.summary_embedding_model
        )
        if not code_shares_endpoint:
            self._code_embedder.preflight()

    async def index_code(
        self,
        project: str,
        root_path: Path,
        nodes: Sequence[CodeNode],
        stored_chunk_embeddings: dict[str, list[float]] | None = None,
        prune_stale: bool = False,
        reconcilable_files: set[str] | None = None,
        progress: ProgressReporter | None = None,
    ) -> int:
        """Chunk source files, embed only new/changed chunks, store CodeChunk nodes.

        Chunks whose content-derived id is already in *stored_chunk_embeddings* reuse that
        embedding (no API call). With *prune_stale*, chunks stored in the DB but absent
        from this run are deleted afterwards — skipped when any embedding failed, so an
        API outage can never shrink the stored chunk set. When *reconcilable_files* is
        given, pruning is scoped to those files: a file a read/parse failure dropped from
        the build keeps its stale chunks instead of being emptied (INV-1/INV-3, finding 4).
        Returns the number of chunks stored.
        """
        stored = stored_chunk_embeddings or {}
        bar = resolve(progress)
        # Files this run is responsible for. When every chunk stores without a failure
        # these get a coverage marker (see _mark_chunks_covered) so a later run can tell a
        # fully-chunked file from one with stale/partial chunks (audit finding 8b). A file
        # that legitimately produces zero chunks is still "covered".
        run_files = {n.file_path for n in nodes if n.file_path}
        # Files that turned unreadable mid-run are reported here; they keep their stale
        # chunks (excluded from pruning) and are NOT marked covered, so the next run retries
        # them instead of the transient failure emptying good data (INV-1/INV-3).
        unreadable: set[str] = set()
        all_chunks = self._chunker.chunk_repository(root_path, list(nodes), unreadable_out=unreadable)
        run_files -= unreadable
        if reconcilable_files is not None:
            reconcilable_files = reconcilable_files - unreadable
        if not all_chunks:
            logger.info("No chunks produced for project=%s", project)
            # A run can legitimately produce zero chunks — e.g. its only changed file was
            # emptied. That file's old chunks are stale and must still be pruned: stamping
            # coverage without pruning would leave garbage chunks no repair query can see
            # (finding 8b). Pruning here needs the authoritative reconcilable_files scope;
            # without it, "zero chunks" is indistinguishable from total failure, and
            # pruning would wipe the whole project (INV-1) — so it is skipped.
            if prune_stale and reconcilable_files is not None:
                await self._prune_stale_chunks(project, set(), reconcilable_files)
            await self._mark_chunks_covered(project, run_files)
            return 0

        prepared = [(chunk, _chunk_id(chunk)) for chunk in all_chunks]
        total = len(prepared)
        logger.info("Indexing %d code chunks in windows of %d for project=%s", total, self._batch_size, project)

        stored_count = 0
        embedded_count = 0
        failed_count = 0
        reused_count = 0
        # Embed + store one window at a time so peak embedding-vector memory is
        # bounded and each window commits to Memgraph before the next is built.
        for start in range(0, total, self._batch_size):
            window = prepared[start : start + self._batch_size]
            local_idx: list[int] = []
            texts: list[str] = []
            for j, (chunk, chunk_id) in enumerate(window):
                if chunk_id not in stored:
                    local_idx.append(j)
                    texts.append(chunk.content)

            fresh = self._code_embedder.embed(texts) if texts else []
            fresh_by_local = dict(zip(local_idx, fresh, strict=True))
            embedded_count += sum(1 for e in fresh if e is not None)
            failed_count += sum(1 for e in fresh if e is None)

            batch: list[dict[str, object]] = []
            for j, (chunk, chunk_id) in enumerate(window):
                if j in fresh_by_local:
                    embedding = fresh_by_local[j]
                else:
                    embedding = stored.get(chunk_id)
                    reused_count += 1
                if embedding is None:
                    continue
                batch.append({
                    "chunk_id": chunk_id,
                    "node_id": chunk.node_id,
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "chunk_kind": chunk.chunk_kind,
                    "embedding": embedding,
                })

            if batch:
                stored_count += await self._repo.store_code_chunks(project, batch)
            processed = min(start + self._batch_size, total)
            logger.info("Code chunks: %d/%d processed for project=%s", processed, total, project)
            bar.update("embed", processed, total)

        if failed_count:
            logger.error(
                "Chunk embedding had %d failure(s) for project=%s — failed chunks were not "
                "stored and stale chunks were left in place; the next index run retries them",
                failed_count, project,
            )
        else:
            if prune_stale:
                keep = {chunk_id for _, chunk_id in prepared}
                await self._prune_stale_chunks(project, keep, reconcilable_files)
            # No embedding failed: every chunk of every file this run handled is now stored,
            # so mark those files fully covered for their current content (audit finding 8b).
            await self._mark_chunks_covered(project, run_files)

        logger.info("Stored %d code chunks (%d embedded, %d reused, %d failed) for project=%s",
                    stored_count, embedded_count, reused_count, failed_count, project)
        return stored_count

    async def _prune_stale_chunks(
        self, project: str, keep: set[str], reconcilable_files: set[str] | None
    ) -> None:
        """Delete stored chunks absent from this run's *keep* set.

        With *reconcilable_files*, pruning is scoped to files this run authoritatively
        rebuilt or confirmed gone. A file dropped by a read/parse failure keeps its stale
        chunks — the node reconcile kept its nodes, so its chunks must survive too
        (finding 4).
        """
        if reconcilable_files is None:
            stale = await self._repo.get_chunk_ids(project) - keep
        else:
            chunk_index = await self._repo.get_chunk_index(project)
            stale = {
                cid for cid, fp in chunk_index.items()
                if cid not in keep and fp in reconcilable_files
            }
        if stale:
            await self._repo.delete_chunks_by_ids(project, sorted(stale))

    async def _mark_chunks_covered(self, project: str, files: set[str]) -> None:
        """Stamp the chunk-coverage marker on files whose chunks are fully stored.

        The marker is each file's own stored checksum, so a later run flags the file for
        re-chunking exactly when its content checksum no longer matches (audit finding 8b).
        Files whose checksum isn't recorded yet are skipped — they'll be treated as changed
        and rebuilt anyway.
        """
        if not files:
            return
        checksums = await self._repo.get_checksums(project)
        items = [
            {"file_path": fp, "checksum": checksums[fp]}
            for fp in sorted(files)
            if fp in checksums
        ]
        if items:
            await self._repo.mark_files_chunks_covered(project, items)

    async def index_summaries(
        self,
        project: str,
        nodes: Sequence[CodeNode],
        stored_summaries: dict[str, dict[str, object]] | None = None,
        dirty_ids: set[str] | None = None,
        progress: ProgressReporter | None = None,
    ) -> int:
        """Embed summaries and store them, reusing stored embeddings for non-dirty nodes.

        Returns the number of embeddings stored.
        """
        stored = stored_summaries or {}
        bar = resolve(progress)
        nodes_with_summaries = [n for n in nodes if n.summary]
        if not nodes_with_summaries:
            logger.info("No summaries to embed for project=%s", project)
            return 0

        total = len(nodes_with_summaries)
        stored_count = 0
        embedded_count = 0
        failed_count = 0
        # Window the embed + store so peak memory stays bounded and each window persists.
        for start in range(0, total, self._batch_size):
            window = nodes_with_summaries[start : start + self._batch_size]
            to_embed: list[CodeNode] = []
            items: list[dict[str, object]] = []
            for node in window:
                if dirty_ids is not None and node.id not in dirty_ids:
                    prev = stored.get(node.id) or {}
                    reused = prev.get("summary_embedding")
                    prev_hash = prev.get("summary_embedding_hash")
                    # Reuse a stored vector only when it is still *fresh* for the current
                    # summary: its recorded hash matches this node's input hash, or it is a
                    # legacy vector predating the marker (hash absent, assumed fresh). A
                    # vector whose hash no longer matches means the summary was regenerated
                    # but its embedding never caught up — re-embed it (audit finding 8a).
                    # The reused vector is re-stored with its *original* hash so a legacy
                    # embedding is never silently stamped fresh.
                    if reused is not None and (prev_hash is None or prev_hash == node.summary_input_hash):
                        items.append({"node_id": node.id, "embedding": reused, "embedding_hash": prev_hash})
                        continue
                to_embed.append(node)

            if to_embed:
                embeddings = self._summary_embedder.embed([n.summary for n in to_embed])
                for node, embedding in zip(to_embed, embeddings, strict=True):
                    if embedding is None:
                        failed_count += 1
                        continue
                    embedded_count += 1
                    items.append({
                        "node_id": node.id,
                        "embedding": embedding,
                        "embedding_hash": node.summary_input_hash,
                    })

            if items:
                stored_count += await self._repo.store_summary_embeddings(project, items)
            processed = min(start + self._batch_size, total)
            logger.info("Summary embeddings: %d/%d processed for project=%s", processed, total, project)
            bar.update("summ-embed", processed, total)

        if failed_count:
            logger.error(
                "Summary embedding had %d failure(s) for project=%s — affected nodes keep "
                "their previous embedding (if any) and are retried on the next index",
                failed_count, project,
            )
        logger.info(
            "Stored %d summary embeddings (%d re-embedded, %d failed) for project=%s",
            stored_count, embedded_count, failed_count, project,
        )
        return stored_count
