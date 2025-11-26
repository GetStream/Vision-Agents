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

    async def create_response(self, *args: Any, **kwargs: Any):
        """Override create_response to handle OpenRouter-specific requirements.
        
        OpenRouter doesn't support the 'instructions' parameter like OpenAI does,
        so we convert it to a system message in the input.
        """
        # Set max_output_tokens default because we're getting rate limited by OpenRouter
        if "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = 1024
        
        # Convert instructions to system message before processing
        if self._instructions:
            current_input = kwargs.get("input", args[0] if args else "Hello")
            
            # Normalize input to list format
            if isinstance(current_input, str):
                input_messages = [{"content": current_input, "role": "user", "type": "message"}]
            elif isinstance(current_input, list):
                input_messages = current_input
            else:
                input_messages = [current_input]
            
            # Prepend system message with instructions
            system_message = {
                "content": self._instructions,
                "role": "system",
                "type": "message"
            }
            kwargs["input"] = [system_message] + input_messages
            
            # Temporarily clear instructions so parent doesn't add it
            original_instructions = self._instructions
            self._instructions = None  # type: ignore[assignment]
            try:
                result = await super().create_response(*args, **kwargs)
            finally:
                self._instructions = original_instructions
            return result
        
        return await super().create_response(*args, **kwargs)

    def add_conversation_history(self, kwargs):
        # Use the manual storage
        # ensure the AI remembers the past conversation
        # TODO: there are additional formats to support here.
        new_messages = kwargs["input"]
        if not isinstance(new_messages, list):
            new_messages = [dict(content=new_messages, role="user", type="message")]
        
        # Build the message list
        messages = []
        
        if hasattr(self, "_conversation") and self._conversation:
            # Extract serializable message content from conversation history
            for m in self._conversation.messages:
                if isinstance(m.original, dict):
                    messages.append(m.original)
                else:
                    # For non-dict originals, use the normalized content
                    messages.append({
                        "content": m.content,
                        "role": "user",
                        "type": "message"
                    })
        
        # Add new messages
        messages.extend(new_messages)
        kwargs["input"] = messages
        
        # Add messages to conversation
        if hasattr(self, "_conversation") and self._conversation:
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)
