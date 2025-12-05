"""OpenRouter LLM implementation using Chat Completions API.

OpenRouter supports many models from different providers. This implementation
extends the standard ChatCompletionsLLM and points it at OpenRouter's API.
"""

import os
from typing import Any, Optional

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM


class OpenRouterLLM(ChatCompletionsLLM):
    """OpenRouter LLM using the standard Chat Completions API.

    This is a thin wrapper around ChatCompletionsLLM that configures it
    to use OpenRouter's API endpoint. All the heavy lifting (streaming,
    tool calling, etc.) is handled by the parent class.

    Examples:

        from vision_agents.plugins import openrouter
        llm = openrouter.LLM(model="openai/gpt-4o")
        llm = openrouter.LLM(model="anthropic/claude-sonnet-4")
        llm = openrouter.LLM(model="google/gemini-2.5-flash")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openrouter/auto",
        client: Optional[AsyncOpenAI] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter LLM.

        Args:
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            model: Model to use (e.g., 'openai/gpt-4o', 'google/gemini-2.5-flash').
            client: Optional pre-configured AsyncOpenAI client.
            **kwargs: Additional arguments passed to parent class.
        """
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
