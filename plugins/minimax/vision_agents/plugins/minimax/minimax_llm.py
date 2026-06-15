"""MiniMax LLM implementation using OpenAI-compatible Chat Completions API.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
Supported models: MiniMax-M3 (default), MiniMax-M2.7, MiniMax-M2.7-highspeed.
"""

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)


PLUGIN_NAME = "minimax"

DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M3"


class MiniMaxLLM(ChatCompletionsLLM):
    """MiniMax LLM using OpenAI-compatible Chat Completions API.

    MiniMax provides high-performance language models accessible through
    an OpenAI-compatible API. This plugin reuses the OpenAI Chat Completions
    streaming and tool-calling implementation, including ``<think>`` reasoning
    stripping for reasoning models such as MiniMax-M3.

    Examples:

        from vision_agents.plugins import minimax
        llm = minimax.LLM()
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
        """Initialize the MiniMaxLLM class.

        Args:
            model: The MiniMax model to use.
                Defaults to ``"MiniMax-M3"`` (latest flagship, 512K context).
                Supported models: ``"MiniMax-M3"``, ``"MiniMax-M2.7"``,
                ``"MiniMax-M2.7-highspeed"``.
            api_key: Optional API key. Defaults to ``MINIMAX_API_KEY`` env var.
            base_url: Optional base URL. Defaults to
                ``https://api.minimax.io/v1`` (overseas endpoint).
            client: Optional ``AsyncOpenAI`` client for dependency injection.
            max_tokens: This sets the upper limit for the number of tokens the
                model can generate in response.
            tools_max_rounds: max calling rounds for multi-hop tool call.
        """
        if client is None:
            if api_key is None:
                api_key = os.environ.get("MINIMAX_API_KEY")
            if base_url is None:
                base_url = os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
        self._max_tokens = max_tokens

    def _extra_request_kwargs(self) -> dict[str, Any]:
        # MiniMax rejects temperature 0.0, so default to 1.0.
        kwargs: dict[str, Any] = {"temperature": 1.0}
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        return kwargs
