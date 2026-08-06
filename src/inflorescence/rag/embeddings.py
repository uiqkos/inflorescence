"""OpenRouter-compatible embedding client with dual-limit batching and retry."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from inflorescence.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class EmbeddingClient:
    """OpenAI-compatible embedding client routed through OpenRouter.

    Supports dual-limit batching (count *and* character ceiling) to stay
    under provider token limits.
    """

    def __init__(
        self,
        settings: Settings,
        model: str,
        batch_size: int = 50,
        batch_max_chars: int = 80_000,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.batch_max_chars = batch_max_chars
        # base_url/api_key let a code-specialized embedder (e.g. Mistral codestral-embed,
        # not on OpenRouter) target its own endpoint; both fall back to the LLM endpoint.
        self._client = OpenAI(
            api_key=api_key or settings.llm_api_key,
            base_url=base_url or settings.llm_base_url,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=2, max=60),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the embeddings API for a single batch of texts."""
        sanitized = [t if t.strip() else " " for t in texts]
        total_chars = sum(len(t) for t in sanitized)
        logger.info("Embedding batch: %d texts, %d total chars, model=%s", len(sanitized), total_chars, self.model)
        response = self._client.embeddings.create(model=self.model, input=sanitized)
        if not response.data:
            raise ValueError(f"No embedding data received for {len(sanitized)} texts ({total_chars} chars)")
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        logger.debug("Batch embedded: %d vectors of dim %d", len(embeddings), len(embeddings[0]) if embeddings else 0)
        return embeddings

    def _split_batches(self, texts: Sequence[str]) -> list[list[str]]:
        """Split *texts* into batches respecting both count and char limits."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for t in texts:
            t_len = len(t)
            if current and (len(current) >= self.batch_size or current_chars + t_len > self.batch_max_chars):
                batches.append(current)
                current = []
                current_chars = 0
            current.append(t)
            current_chars += t_len
        if current:
            batches.append(current)
        return batches

    def preflight(self) -> None:
        """Prove the endpoint accepts our credentials with one tiny embedding call.

        Raises the underlying API error (e.g. 401 on a missing/invalid key) instead of
        swallowing it — callers use this to abort an index/update BEFORE any graph
        mutation, so a misconfigured server cannot degrade stored data.
        """
        self._embed_batch(["ok"])

    def embed(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed a list of texts, batching as needed.

        Returns a list aligned with *texts*.  Failed batches produce ``None``
        entries so the caller can skip them while keeping alignment.
        """
        if not texts:
            return []
        text_list = list(texts)
        batches = self._split_batches(text_list)
        total_chars = sum(len(t) for t in text_list)
        logger.info(
            "Embedding %d texts (%d chars) in %d batches (model=%s, batch_size=%d, max_chars=%d)",
            len(text_list), total_chars, len(batches), self.model, self.batch_size, self.batch_max_chars,
        )
        all_embeddings: list[list[float] | None] = []
        failed_batches = 0
        for batch_num, batch in enumerate(batches, 1):
            logger.info(
                "Embedding batch %d/%d (%d texts, %d chars)...",
                batch_num, len(batches), len(batch), sum(len(t) for t in batch),
            )
            try:
                all_embeddings.extend(self._embed_batch(batch))
            except Exception:
                logger.exception("Embedding batch %d/%d failed, skipping %d texts", batch_num, len(batches), len(batch))
                all_embeddings.extend([None] * len(batch))
                failed_batches += 1
        logger.info(
            "Embedding complete: %d vectors (%d batches failed)",
            sum(1 for e in all_embeddings if e is not None), failed_batches,
        )
        return all_embeddings


def embed_code(texts: Sequence[str], settings: Settings) -> list[list[float] | None]:
    """Embed code chunks using the code embedding model from *settings*."""
    client = EmbeddingClient(
        settings=settings,
        model=settings.code_embedding_model,
        base_url=settings.code_embedding_base_url,
        api_key=settings.code_embedding_api_key,
    )
    return client.embed(texts)


def embed_summaries(texts: Sequence[str], settings: Settings) -> list[list[float] | None]:
    """Embed summaries using the summary embedding model from *settings*."""
    client = EmbeddingClient(settings=settings, model=settings.summary_embedding_model)
    return client.embed(texts)
