"""Cheaper Inference LLM implementation using the OpenAI-compatible Chat Completions API.

Cheaper Inference provides an OpenAI-compatible API at https://api.cheaperinference.com/v1.
Model ids are bare, for example ``gpt-5.4-mini`` (default), ``gpt-5.4`` or ``claude-sonnet-5``.
"""

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)


PLUGIN_NAME = "cheaperinference"

DEFAULT_BASE_URL = "https://api.cheaperinference.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"


class CheaperInferenceLLM(ChatCompletionsLLM):
    """Cheaper Inference LLM using the OpenAI-compatible Chat Completions API.

    Cheaper Inference is an OpenAI-compatible LLM gateway. This plugin reuses the
    OpenAI Chat Completions streaming and tool-calling implementation.

    Examples:

        from vision_agents.plugins import cheaperinference
        llm = cheaperinference.LLM()
    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
        max_tokens: int | None = None,
        tools_max_rounds: int = 3,
    ) -> None:
        """Initialize the CheaperInferenceLLM class.

        Args:
            model: The model id to use. Defaults to ``"gpt-5.4-mini"``.
            api_key: Optional API key. Defaults to ``CHEAPER_INFERENCE_API_KEY`` env var.
            base_url: Optional base URL. Defaults to ``CHEAPER_INFERENCE_BASE_URL``
                env var, or ``https://api.cheaperinference.com/v1``.
            client: Optional ``AsyncOpenAI`` client for dependency injection.
            max_tokens: Upper limit for the number of tokens the model can generate.
            tools_max_rounds: max calling rounds for multi-hop tool call.
        """
        if client is None:
            if api_key is None:
                api_key = os.environ.get("CHEAPER_INFERENCE_API_KEY")
            if base_url is None:
                base_url = os.environ.get(
                    "CHEAPER_INFERENCE_BASE_URL", DEFAULT_BASE_URL
                )
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
        self._max_tokens = max_tokens

    def _extra_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return kwargs
