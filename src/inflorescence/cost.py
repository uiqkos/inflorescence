"""Pre-flight indexing cost estimation (USD).

Turns the no-cost scan preview (``estimated_tokens`` / ``estimated_nodes``) into a rough
dollar figure using the configured model prices, so a first-time index can be gated by a
``max_index_cost_usd`` cap before spending anything. The estimate is deliberately a slight
over-estimate (it is a guard, not a billing system).
"""

from __future__ import annotations

from inflorescence.config import Settings

# Per-node overheads for the estimate. A summary call sends the element's source plus a
# system prompt + signature/docstring (leaves) or child descriptors (containers), and the
# generated summary is short. These are intentionally generous.
_PROMPT_TOKENS_PER_NODE = 250
_OUTPUT_TOKENS_PER_NODE = 90
_SUMMARY_EMBED_TOKENS_PER_NODE = 90


class IndexCostExceededError(Exception):
    """Raised when a first-time index's estimated cost exceeds the configured cap."""

    def __init__(self, estimated_usd: float, limit_usd: float) -> None:
        self.estimated_usd = estimated_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"Estimated indexing cost ${estimated_usd:.4f} exceeds the limit ${limit_usd:.4f}. "
            f"Raise the limit (max_index_cost_usd / --max-cost), narrow scope with --exclude, "
            f"or preview with --dry-run."
        )


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def estimate_index_cost(preview: dict[str, object], settings: Settings) -> dict[str, float]:
    """Estimate the USD cost of indexing from a scan preview.

    Returns ``{"usd", "llm_input_tokens", "llm_output_tokens", "embedding_tokens"}``.
    """
    tokens = _as_int(preview.get("estimated_tokens"))  # source-derived proxy
    nodes = _as_int(preview.get("estimated_nodes"))

    if settings.use_llm_summaries:
        llm_input = tokens + nodes * _PROMPT_TOKENS_PER_NODE
        llm_output = nodes * _OUTPUT_TOKENS_PER_NODE
        embedding = tokens + nodes * _SUMMARY_EMBED_TOKENS_PER_NODE
    else:
        llm_input = llm_output = 0
        embedding = tokens  # code chunks only, no summaries

    usd = (
        llm_input * settings.llm_price_input_usd_per_mtok
        + llm_output * settings.llm_price_output_usd_per_mtok
        + embedding * settings.embedding_price_usd_per_mtok
    ) / 1_000_000

    return {
        "usd": round(usd, 4),
        "llm_input_tokens": float(llm_input),
        "llm_output_tokens": float(llm_output),
        "embedding_tokens": float(embedding),
    }
