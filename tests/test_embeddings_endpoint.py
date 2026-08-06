from __future__ import annotations

import pytest

from inflorescence.config import Settings
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.rag.indexer import RAGIndexer


def _base(client: EmbeddingClient) -> str:
    return str(client._client.base_url).rstrip("/")


def test_embedding_client_targets_override_endpoint_else_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    settings = Settings(_env_file=None, llm_base_url="https://openrouter.ai/api/v1")

    # A code-specialized embedder points at its own provider (e.g. Mistral).
    code = EmbeddingClient(
        settings=settings, model="codestral-embed",
        base_url="https://api.mistral.ai/v1", api_key="mistral-key",
    )
    assert _base(code) == "https://api.mistral.ai/v1"
    assert code._client.api_key == "mistral-key"

    # No override -> falls back to the shared LLM endpoint + key.
    fallback = EmbeddingClient(settings=settings, model="openai/text-embedding-3-small")
    assert _base(fallback) == "https://openrouter.ai/api/v1"
    assert fallback._client.api_key == "or-key"


def test_rag_indexer_wires_code_endpoint_but_summary_stays_on_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    settings = Settings(
        _env_file=None,
        code_embedding_model="codestral-embed",
        code_embedding_base_url="https://api.mistral.ai/v1",
        code_embedding_api_key="mistral-key",
    )
    idx = RAGIndexer(repo=object(), settings=settings)  # type: ignore[arg-type]  # repo unused at init

    assert idx._code_embedder.model == "codestral-embed"
    assert _base(idx._code_embedder) == "https://api.mistral.ai/v1"
    assert idx._code_embedder._client.api_key == "mistral-key"

    # Summaries keep using the shared LLM endpoint — only code embeddings were redirected.
    assert _base(idx._summary_embedder) == "https://openrouter.ai/api/v1"


class _CountingEmbedder:
    def __init__(self) -> None:
        self.preflights = 0

    def preflight(self) -> None:
        self.preflights += 1


def test_preflight_probes_both_endpoints_when_the_code_embedder_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # INV-2/finding 6: a broken code-embedding key used to slip through because preflight only
    # probed the summary endpoint; a first index then built the graph and failed every chunk
    # batch. With a distinct code endpoint, BOTH must be probed.
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    settings = Settings(
        _env_file=None,
        code_embedding_model="codestral-embed",
        code_embedding_api_key="mistral-key",
    )
    idx = RAGIndexer(repo=object(), settings=settings)  # type: ignore[arg-type]
    summary, code = _CountingEmbedder(), _CountingEmbedder()
    idx._summary_embedder = summary  # type: ignore[assignment]
    idx._code_embedder = code  # type: ignore[assignment]

    idx.preflight()

    assert summary.preflights == 1
    assert code.preflights == 1


def test_preflight_probes_a_shared_endpoint_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # When code and summary share the same model/endpoint/key, the summary probe already
    # covered it — don't pay for a redundant second call.
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    settings = Settings(_env_file=None)  # both default to text-embedding-3-small on the LLM endpoint
    idx = RAGIndexer(repo=object(), settings=settings)  # type: ignore[arg-type]
    summary, code = _CountingEmbedder(), _CountingEmbedder()
    idx._summary_embedder = summary  # type: ignore[assignment]
    idx._code_embedder = code  # type: ignore[assignment]

    idx.preflight()

    assert summary.preflights == 1
    assert code.preflights == 0
