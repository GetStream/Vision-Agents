"""OpenRouter LLM implementation using OpenAI-compatible API."""

import json
import os
from typing import Any, Dict, List, Optional

from openai.types.responses import Response as OpenAIResponse
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem
from vision_agents.plugins.openai import LLM as OpenAILLM


class OpenRouterLLM(OpenAILLM):
    """OpenRouter LLM that extends OpenAI LLM with OpenRouter-specific configuration.

    It proxies the regular models by setting base url.
    It supports create response like the regular openAI API. It doesn't support conversation id, so that requires customization

    TODO:
    - Use manual conversation storage
    """

    _instructions: str

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
        # # Set max_output_tokens default because we're getting rate limited by OpenRouter
        # kwargs.setdefault("max_output_tokens", 1024)

        # Convert instructions to system message
        if not self._instructions:
            return await super().create_response(*args, **kwargs)

        # Get and normalize input to list format
        current_input = kwargs.get("input", args[0] if args else "Hello")
        if not isinstance(current_input, list):
            current_input = [
                {"content": str(current_input), "role": "user", "type": "message"}
            ]

        # Prepend system message with instructions
        kwargs["input"] = [
            {"content": self._instructions, "role": "system", "type": "message"},
            *current_input,
        ]

        # Temporarily clear instructions so parent doesn't add it
        self._instructions, original = None, self._instructions  # type: ignore[assignment]
        try:
            return await super().create_response(*args, **kwargs)
        finally:
            self._instructions = original

    def add_conversation_history(self, kwargs):
        # Use the manual storage
        # ensure the AI remembers the past conversation
        # TODO: there are additional formats to support here.
        new_messages = kwargs["input"]
        if not isinstance(new_messages, list):
            new_messages = [dict(content=new_messages, role="user", type="message")]

        # Build the message list
        messages = []

        if self._conversation:
            # Extract serializable message content from conversation history
            for m in self._conversation.messages:
                if isinstance(m.original, dict):
                    messages.append(m.original)
                else:
                    # For non-dict originals, use the normalized content
                    messages.append(
                        {"content": m.content, "role": "user", "type": "message"}
                    )

        # Add new messages
        messages.extend(new_messages)
        kwargs["input"] = messages

        # Add messages to conversation
        if self._conversation:
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

    async def _handle_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        original_kwargs: Dict[str, Any],
    ) -> LLMResponseEvent[OpenAIResponse]:
        """Handle tool calls for OpenRouter without conversation IDs.

        Unlike OpenAI which uses server-side conversation state, OpenRouter requires
        full message history to be sent with each request. This includes the assistant's
        function_call items before the function_call_output results.

        Args:
            tool_calls: List of tool calls to execute.
            original_kwargs: Original kwargs from the request (contains full context).

        Returns:
            LLMResponseEvent with the final response.
        """
        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()

        # Get the full conversation context from the original request.
        # This already includes: system message (instructions) + history + user message
        conversation_context = list(original_kwargs.get("input", []))
        if not isinstance(conversation_context, list):
            conversation_context = [
                {"content": str(conversation_context), "role": "user", "type": "message"}
            ]

        for round_num in range(max_rounds):
            # Execute tools with deduplication
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,  # type: ignore[arg-type]
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            # Build assistant function_call items (model's decision to call functions)
            # These must precede the function_call_output in the conversation
            assistant_function_calls = []
            for tc, _res, _err in triples:
                cid = tc.get("id")
                if not cid:
                    continue
                assistant_function_calls.append(
                    {
                        "type": "function_call",
                        "call_id": cid,
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments_json", {})),
                    }
                )

            # Build tool output messages (results of function execution)
            tool_messages = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue
                output = err if err is not None else res
                output_str = self._sanitize_tool_output(output)
                tool_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": cid,
                        "output": output_str,
                    }
                )

            if not tool_messages:
                return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

            # Build follow-up input:
            # conversation context + assistant's function_calls + function_call_output results
            follow_up_input = conversation_context + assistant_function_calls + tool_messages

            follow_up_kwargs = {
                "model": original_kwargs.get("model", self.model),
                "input": follow_up_input,
                "stream": True,
            }

            # Include tools for potential follow-up calls
            tools_spec = self._get_tools_for_provider()
            if tools_spec:
                follow_up_kwargs["tools"] = self._convert_tools_to_provider_format(
                    tools_spec  # type: ignore[arg-type]
                )

            # Get follow-up response directly from client (bypass create_response wrapper)
            follow_up_response = await self.client.responses.create(**follow_up_kwargs)

            if isinstance(follow_up_response, OpenAIResponse):
                # Non-streaming response
                llm_response = LLMResponseEvent[OpenAIResponse](
                    follow_up_response, follow_up_response.output_text
                )

                # Check for more tool calls
                next_tool_calls = self._extract_tool_calls_from_response(follow_up_response)
                if next_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = next_tool_calls
                    # Update context to include the function calls and results for next round
                    conversation_context = follow_up_input
                    continue
                return llm_response

            elif hasattr(follow_up_response, "__aiter__"):  # async stream
                stream_response = follow_up_response
                pending_tool_calls: List[NormalizedToolCallItem] = []

                async for event in stream_response:
                    llm_response_optional = self._standardize_and_emit_event(event)
                    if llm_response_optional is not None:
                        llm_response = llm_response_optional

                    # Check for tool calls when response completes
                    if getattr(event, "type", "") == "response.completed":
                        calls = self._extract_tool_calls_from_response(event.response)
                        for c in calls:
                            key = (
                                c["id"],
                                c["name"],
                                json.dumps(c["arguments_json"], sort_keys=True),
                            )
                            if key not in seen:
                                pending_tool_calls.append(c)
                                seen.add(key)

                # If we have more tool calls and haven't exceeded max rounds, continue
                if pending_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = pending_tool_calls
                    # Update context to include the function calls and results for next round
                    conversation_context = follow_up_input
                    continue
                return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
            else:
                # Defensive fallback
                return LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

        # If we've exhausted all rounds, return the last response
        return llm_response or LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]
