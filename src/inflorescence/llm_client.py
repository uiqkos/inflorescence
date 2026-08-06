"""Async LLM client wrapping OpenAI SDK for OpenRouter API."""

import logging
from types import TracebackType

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from inflorescence.config import Settings

logger = logging.getLogger(__name__)

# Transient errors worth retrying — anything else should surface immediately.
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class LLMClient:
    """Thin async wrapper around OpenAI chat completions, pointed at OpenRouter."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = openai.AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=60),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Send a chat completion request and return the assistant's text.

        Retries up to 5 times with exponential backoff on transient API errors.
        """
        messages: list[openai.types.chat.ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self._settings.llm_model,
            messages=messages,
        )

        content = response.choices[0].message.content
        # OpenAI spec allows content to be None (e.g. tool-only replies);
        # for plain text generation that should never happen, but guard anyway.
        return content if content is not None else ""

    async def close(self) -> None:
        """Release underlying HTTP resources."""
        await self._client.close()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
