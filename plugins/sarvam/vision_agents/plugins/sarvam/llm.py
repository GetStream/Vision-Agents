"""Sarvam AI LLM using the OpenAI-compatible Chat Completions endpoint.

Sarvam exposes ``/v1/chat/completions`` with the same shape as OpenAI, so we
point an ``AsyncOpenAI`` client at Sarvam's base URL and inject the
``api-subscription-key`` header. Streaming, tool calling, and conversation
history are all inherited from :class:`ChatCompletionsLLM`.

Docs: https://docs.sarvam.ai/api-reference-docs/chat/chat-completions
"""

import logging
import os
from typing import Optional

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
DEFAULT_MODEL = "sarvam-m"
SUPPORTED_MODELS = {"sarvam-m", "sarvam-30b", "sarvam-105b"}


class SarvamLLM(ChatCompletionsLLM):
    """Sarvam AI Chat Completions LLM.

    Thin wrapper around :class:`ChatCompletionsLLM` that configures the OpenAI
    client for Sarvam's OpenAI-compatible endpoint.

    Examples:

        from vision_agents.plugins import sarvam
        llm = sarvam.LLM(model="sarvam-30b")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = SARVAM_BASE_URL,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        """Initialize the Sarvam LLM.

        Args:
            model: The Sarvam model id. Defaults to ``sarvam-m``. Supported:
                ``sarvam-m``, ``sarvam-30b``, ``sarvam-105b``.
            api_key: Sarvam API key. Defaults to ``SARVAM_API_KEY`` env var.
            base_url: API base URL. Defaults to ``https://api.sarvam.ai/v1``.
            client: Optional pre-configured ``AsyncOpenAI`` client. Takes
                precedence over ``api_key`` / ``base_url``.
        """
        resolved_key = (
            api_key if api_key is not None else os.environ.get("SARVAM_API_KEY")
        )
        if client is None and not resolved_key:
            raise ValueError(
                "SARVAM_API_KEY env var or api_key parameter required for Sarvam LLM"
            )

        if client is None:
            client = AsyncOpenAI(
                api_key=resolved_key,
                base_url=base_url,
                default_headers={"api-subscription-key": resolved_key or ""},
            )

        super().__init__(model=model, client=client)
