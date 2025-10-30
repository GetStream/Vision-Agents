"""OpenRouter LLM implementation using OpenAI-compatible API."""
import os
from typing import Any

from vision_agents.plugins.openai import LLM as OpenAILLM


class OpenRouterLLM(OpenAILLM):
    """OpenRouter LLM that extends OpenAI LLM with OpenRouter-specific configuration.

    It proxies the regular models by setting base url.
    It supports create response like the regular openAI API. It doesn't support conversation id, so that requires customization

    TODO:
    - Use manual conversation storage
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openrouter/andromeda-alpha",
        **kwargs: Any,
    ) -> None:
        """Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            model: Model to use. Defaults to openai/gpt-4o.
            **kwargs: Additional arguments passed to OpenAI LLM.
        """
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            **kwargs,
        )

    async def create_conversation(self):
        # Do nothing, dont call super
        pass

    def add_conversation_history(self, kwargs):
        # Use the manual storage
        # ensure the AI remembers the past conversation
        # TODO: there are additional formats to support here.
        new_messages = kwargs["input"]
        if not isinstance(new_messages, list):
            new_messages = [dict(content=new_messages, role="user", type="message")]
        if hasattr(self, '_conversation') and self._conversation:
            old_messages = [m.original for m in self._conversation.messages]
            kwargs["input"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

