"""Pre-flight indexing cost estimation (USD).

Turns the no-cost scan preview (``estimated_tokens`` / ``estimated_nodes``) into a rough
dollar figure using the configured model prices, so any run that would spend can be gated by
a ``max_index_cost_usd`` cap before spending anything — a first-time index over the whole
tree, an incremental re-index or a watcher update over just the files whose content moved.
The estimate is deliberately a slight over-estimate (it is a guard, not a billing system).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inflorescence.config import Settings

if TYPE_CHECKING:
    from inflorescence.code_indexer.summarizer import SummaryPlan

# Per-node overheads for the estimate. A summary call sends the element's source plus a
# system prompt + signature/docstring (leaves) or child descriptors (containers), and the
# generated summary is short. These are intentionally generous.
_PROMPT_TOKENS_PER_NODE = 250
_OUTPUT_TOKENS_PER_NODE = 90
_SUMMARY_EMBED_TOKENS_PER_NODE = 90
# The usual ~4 characters per token, for prompts measured in characters.
_CHARS_PER_TOKEN = 4
# How long a summary is assumed to be when a plan stands one in (see plan_summaries).
SUMMARY_PLACEHOLDER_CHARS = _OUTPUT_TOKENS_PER_NODE * _CHARS_PER_TOKEN


class IndexCostExceededError(Exception):
    """Raised when an index run's estimated cost exceeds the configured cap.

    Raised for all three gated runs: a first-time index (priced over the whole tree), an
    incremental re-index and a background watcher update (both priced over just the files
    whose content moved). ``changed_files``, when set, says how many files the quoted
    estimate covers, so the message cannot be misread as the price of the whole project;
    ``rechunk_files`` adds the unchanged files whose chunks the run will redo because the
    chunker changed (embedding cost only — see :func:`estimate_rechunk_cost`), and
    ``regenerated_summaries`` is set when the refusal came from the exact summary plan
    (:func:`estimate_summary_plan_cost`) rather than the file-level pre-flight.
    """

    def __init__(
        self,
        estimated_usd: float,
        limit_usd: float,
        *,
        changed_files: int | None = None,
        rechunk_files: int = 0,
        regenerated_summaries: int | None = None,
    ) -> None:
        self.estimated_usd = estimated_usd
        self.limit_usd = limit_usd
        self.changed_files = changed_files
        self.rechunk_files = rechunk_files
        self.regenerated_summaries = regenerated_summaries
        scope = f" over {changed_files} changed file(s)" if changed_files is not None else ""
        if rechunk_files:
            scope += (
                f"{' and' if scope else ' over'} {rechunk_files} unchanged file(s) whose chunks "
                "predate the current chunk format (re-chunked and embedded, no LLM)"
            )
        if regenerated_summaries is not None:
            scope += (
                f", including {regenerated_summaries} summar(ies) the run would regenerate "
                "(counted on the parsed graph, so entities an upgraded parser newly recognizes "
                "and the containers above them are in the price)"
            )
        look = (
            # The plan is counted after a parse; a preview prices files by size and would
            # quote a different number, so it is not the way to look at this one.
            "The count above is the run's own summarization pass, made without calling the LLM."
            if regenerated_summaries is not None
            else "Or look first with --dry-run / preview=true."
        )
        super().__init__(
            f"Estimated indexing cost ${estimated_usd:.4f}{scope} exceeds the limit "
            f"${limit_usd:.4f}. Raise the limit (max_index_cost_usd, --max-cost or the "
            f"max_cost_usd argument) or narrow scope with --exclude / exclude_patterns. {look}"
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


def price_largest_files(preview: dict[str, object], settings: Settings) -> dict[str, object]:
    """Put a dollar figure on each of a preview's ``largest_files`` — the same formula as the total.

    Each file's token/node proxies are replaced by its price, so a caller reads "this JSON
    table is $0.21 of the $0.62" and can ``.inflorescenceignore`` it, rather than doing token
    math. A preview without the list is returned unchanged.
    """
    largest = preview.get("largest_files")
    if not isinstance(largest, list):
        return preview
    priced = []
    for item in largest:
        proxies = {key: item.get(key, 0) for key in ("estimated_tokens", "estimated_nodes")}
        rest = {key: value for key, value in item.items() if key not in proxies}
        priced.append({**rest, "estimated_cost_usd": estimate_index_cost(proxies, settings)["usd"]})
    return {**preview, "largest_files": priced}


def estimate_rechunk_cost(preview: dict[str, object], settings: Settings) -> dict[str, float]:
    """Estimate re-chunking files whose content did not change: embeddings only, no LLM.

    A chunker upgrade (``rag.chunker.CHUNK_FORMAT``) re-cuts every file on the next run.
    Chunks identical to stored ones reuse their vectors; only what the new format adds is
    embedded — so pricing the files' whole text at the embedding rate over-estimates, which
    is the right direction for a guard. Summaries are untouched (their input hashes did not
    move), so there is no LLM term.
    """
    tokens = _as_int(preview.get("estimated_tokens"))
    return {
        "usd": round(tokens * settings.embedding_price_usd_per_mtok / 1_000_000, 4),
        "embedding_tokens": float(tokens),
    }


def estimate_summary_plan_cost(plan: SummaryPlan, settings: Settings) -> dict[str, float]:
    """Price a :class:`~inflorescence.code_indexer.summarizer.SummaryPlan`: the summaries a run
    would actually regenerate, with their prompts measured rather than estimated.

    Input is the measured prompt text; output and the new summaries' embeddings use the same
    per-summary allowances as :func:`estimate_index_cost`. Code chunks are not in it — they
    are priced per file by the callers.
    """
    llm_input = plan.prompt_chars / _CHARS_PER_TOKEN
    llm_output = plan.calls * _OUTPUT_TOKENS_PER_NODE
    embedding = len(plan.node_ids) * _SUMMARY_EMBED_TOKENS_PER_NODE
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
