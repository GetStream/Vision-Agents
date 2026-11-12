"""Neon AI Gateway LLM using the OpenAI-compatible Chat Completions endpoint.

Neon AI Gateway serves models from OpenAI, Anthropic, Meta, Alibaba and others
behind one branch-scoped Neon credential. Its unified ``/v1/chat/completions``
endpoint is OpenAI-compatible, so streaming, tool calling, and conversation history
are inherited from :class:`ChatCompletionsLLM`.

Docs: https://neon.com/docs/ai-gateway/chat-completions
"""

import logging
import os
from typing import Any

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)


PLUGIN_NAME = "neon"

DEFAULT_MODEL = "gpt-5-mini"


class NeonLLM(ChatCompletionsLLM):
    """Neon AI Gateway LLM using the OpenAI-compatible Chat Completions API.

    Examples:

        from vision_agents.plugins import neon
        llm = neon.LLM(model="claude-sonnet-4-6")
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
        """Initialize the NeonLLM class.

        Args:
            model: AI Gateway model ID, e.g. ``"gpt-5-mini"`` or ``"claude-sonnet-4-6"``.
                Defaults to ``"gpt-5-mini"``.
            api_key: Neon credential with the ``ai_gateway:invoke`` scope.
                Defaults to the ``NEON_AI_GATEWAY_TOKEN`` env var.
            base_url: Bare branch AI Gateway host, e.g.
                ``https://br-xxx-api.ai.c-2.us-east-2.aws.neon.tech``. ``/v1`` is
                appended. Defaults to the ``NEON_AI_GATEWAY_BASE_URL`` env var.
            client: Optional ``AsyncOpenAI`` client. Takes precedence over
                ``api_key`` / ``base_url``.
            max_tokens: Upper limit for the number of tokens the model can generate.
            tools_max_rounds: Max calling rounds for multi-hop tool calls.
        """
        if client is None:
            api_key = api_key or os.environ.get("NEON_AI_GATEWAY_TOKEN")
            base_url = base_url or os.environ.get("NEON_AI_GATEWAY_BASE_URL")
            if not api_key:
                raise ValueError(
                    "NEON_AI_GATEWAY_TOKEN env var or api_key parameter required for Neon LLM"
                )
            if not base_url:
                raise ValueError(
                    "NEON_AI_GATEWAY_BASE_URL env var or base_url parameter required for Neon LLM"
                )
            base_url = f"{base_url.rstrip('/')}/v1"
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
        self._max_tokens = max_tokens

    def _extra_request_kwargs(self) -> dict[str, Any]:
        if self._max_tokens is None:
            return {}
        return {"max_tokens": self._max_tokens}
