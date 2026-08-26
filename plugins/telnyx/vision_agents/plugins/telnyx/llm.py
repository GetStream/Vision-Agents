"""Telnyx Inference LLM using the OpenAI-compatible Chat Completions endpoint.

Telnyx serves ``/v2/ai/chat/completions`` with the same request and response
shape as OpenAI, so we point an ``AsyncOpenAI`` client at the Telnyx base URL
and authenticate with the standard bearer token. Streaming, tool calling, and
conversation history are all inherited from :class:`ChatCompletionsLLM`.

The catalogue of served models is fetched at runtime from ``/v2/ai/models``
and changes over time, so model ids are not validated locally.

Docs: https://developers.telnyx.com/api/inference/inference-embedding/post-chat-completions-public-chat-completions-post
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)

TELNYX_BASE_URL = "https://api.telnyx.com/v2/ai"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class TelnyxLLM(ChatCompletionsLLM):
    """Telnyx Inference Chat Completions LLM.

    Thin wrapper around :class:`ChatCompletionsLLM` that configures the OpenAI
    client for Telnyx's OpenAI-compatible inference endpoint.

    Examples:

        from vision_agents.plugins import telnyx
        llm = telnyx.LLM(model="openai/gpt-4o")
    """

    provider_name = "telnyx"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = TELNYX_BASE_URL,
        client: Optional[AsyncOpenAI] = None,
        tools_max_rounds: int = 3,
    ) -> None:
        """Initialize the Telnyx LLM.

        Args:
            model: The model id as served by Telnyx Inference, for example
                ``meta-llama/Llama-3.3-70B-Instruct`` or ``openai/gpt-4o``.
            api_key: Telnyx API key. Defaults to the ``TELNYX_API_KEY`` env var.
            base_url: API base URL. Defaults to ``https://api.telnyx.com/v2/ai``.
            client: Optional pre-configured ``AsyncOpenAI`` client. Takes
                precedence over ``api_key`` / ``base_url``.
            tools_max_rounds: Max calling rounds for multi-hop tool calls.
        """
        resolved_key = (
            api_key if api_key is not None else os.environ.get("TELNYX_API_KEY")
        )
        if client is None and not resolved_key:
            raise ValueError(
                "TELNYX_API_KEY env var or api_key parameter required for Telnyx LLM"
            )

        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
