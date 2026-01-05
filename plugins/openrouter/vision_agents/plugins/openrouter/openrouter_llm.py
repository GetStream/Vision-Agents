"""OpenRouter LLM implementation using Chat Completions API.

OpenRouter supports many models from different providers. This implementation
uses Chat Completions API for all models as it's the industry standard and
works consistently across all providers.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, cast

from openai import AsyncStream
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.plugins.openai import LLM as OpenAILLM

logger = logging.getLogger(__name__)

# Models that reliably support tool calling via Chat Completions API.
# Used as fallbacks when openrouter/auto routes to a model without tool support.
TOOL_SUPPORTING_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
]


class OpenRouterLLM(OpenAILLM):
    """OpenRouter LLM using Chat Completions API for all models.

    Extends OpenAI LLM with OpenRouter-specific handling:
    - Uses Chat Completions API for all models (consistent behavior)
    - Uses manual conversation history (no server-side conversation IDs)
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
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL.
            model: Model to use (e.g., 'openai/gpt-4o-mini', 'google/gemini-2.5-flash').
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
        # For tracking streaming tool calls in Chat Completions mode
        self._pending_tool_calls: Dict[int, Dict[str, Any]] = {}

    def _is_auto_model(self, model: Optional[str] = None) -> bool:
        """Check if the model is a meta/auto model that may not support tools."""
        model = model or self.model
        return model in ("openrouter/auto",)

    def _is_openai_model(self, model: Optional[str] = None) -> bool:
        """Check if the model is an OpenAI model.

        OpenAI models have stricter schema requirements for tool calling
        (all properties must be in `required` when using strict mode).
        """
        model = model or self.model
        return model.startswith("openai/")

    async def create_conversation(self):
        """No-op for OpenRouter (no server-side conversation IDs)."""
        pass

    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponseEvent:
        """Create a response using Chat Completions API.

        Always uses Chat Completions API for consistent behavior across all
        providers. OpenRouter's Responses API support is in beta, so we avoid it.
        """
        return await self._create_response_chat_completions(*args, **kwargs)

    # =========================================================================
    # Chat Completions API implementation
    # =========================================================================

    async def _create_response_chat_completions(
        self, *args: Any, **kwargs: Any
    ) -> LLMResponseEvent:
        """Create response using Chat Completions API."""
        from vision_agents.core.agents.conversation import Message

        # Get the user input
        user_input = kwargs.get("input", args[0] if args else "Hello")

        # Convert input to messages format (includes conversation history)
        messages = self._build_chat_messages(user_input)

        # Add tools if available
        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_chat_completions_format(tools_spec)

        response = await self._chat_completions_internal(
            messages=messages,
            tools=tools_param,
            model=kwargs.get("model", self.model),
            stream=kwargs.get("stream", True),
        )

        # Update conversation history with the exchange
        if self._conversation:
            # Add user message
            if isinstance(user_input, str):
                self._conversation.messages.append(
                    Message(
                        original={"role": "user", "content": user_input},
                        content=user_input,
                        role="user",
                    )
                )
            # Add assistant response
            if response.text:
                self._conversation.messages.append(
                    Message(
                        original={"role": "assistant", "content": response.text},
                        content=response.text,
                        role="assistant",
                    )
                )

        return response

    def _build_chat_messages(self, input_value: Any) -> List[Dict[str, Any]]:
        """Convert input to Chat Completions messages format."""
        messages: List[Dict[str, Any]] = []

        # Add instructions as system message
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        # Add conversation history
        if self._conversation:
            for m in self._conversation.messages:
                messages.append({"role": m.role or "user", "content": m.content})

        # Convert input to user message(s)
        if isinstance(input_value, str):
            messages.append({"role": "user", "content": input_value})
        elif isinstance(input_value, list):
            for item in input_value:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    item_type = item.get("type", "")

                    # Skip system messages if we already added instructions
                    if role == "system" and self._instructions:
                        continue

                    if item_type == "function_call_output":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.get("call_id", ""),
                                "content": item.get("output", ""),
                            }
                        )
                    else:
                        messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": "user", "content": str(item)})
        else:
            messages.append({"role": "user", "content": str(input_value)})

        return messages

    def _convert_tools_to_chat_completions_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema to Chat Completions API format.

        For non-OpenAI models: Adds strict mode to help models understand
        required parameters better.

        For OpenAI models: Omits strict mode because OpenAI requires ALL
        properties to be in `required` when strict is enabled, which breaks
        MCP tools that have optional parameters.
        """
        use_strict = not self._is_openai_model()
        result = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            func_spec: Dict[str, Any] = {
                "name": name,
                "description": description,
                "parameters": params,
            }

            # Add strict mode for non-OpenAI models to help them follow schemas
            if use_strict and params.get("required"):
                func_spec["strict"] = True
                params.setdefault("additionalProperties", False)

            result.append({"type": "function", "function": func_spec})
        return result

    async def _chat_completions_internal(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = True,
    ) -> LLMResponseEvent:
        """Internal Chat Completions implementation with tool handling."""
        effective_model = model or self.model
        request_kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": effective_model,
            "stream": stream,
        }
        if tools:
            request_kwargs["tools"] = tools
            # openrouter/auto may route to models that don't support tools.
            # Add fallbacks to ensure tool calls work.
            if self._is_auto_model(effective_model):
                logger.info(
                    "openrouter/auto with tools: adding fallbacks %s",
                    TOOL_SUPPORTING_MODELS,
                )
                request_kwargs["extra_body"] = {"models": TOOL_SUPPORTING_MODELS}

        response = await self.client.chat.completions.create(**request_kwargs)

        if stream:
            return await self._process_chat_stream(response, messages, tools, model)
        else:
            return await self._process_chat_response(
                cast(ChatCompletion, response), messages, tools, model
            )

    async def _process_chat_stream(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: Optional[str],
    ) -> LLMResponseEvent:
        """Process streaming Chat Completions response.

        Streaming strategy:
        - Emit chunks immediately for real-time TTS
        - If the response ends with tool_calls, we suppress the text (it was narration)
        - If the response ends normally (stop), the chunks were already emitted
        """
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        text_chunks: list[str] = []
        self._pending_tool_calls = {}
        accumulated_tool_calls: List[NormalizedToolCallItem] = []
        has_tool_call_delta = False
        i = 0

        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            # Track if we've seen any tool call deltas
            if choice.delta.tool_calls:
                has_tool_call_delta = True
                for tc in choice.delta.tool_calls:
                    self._accumulate_chat_tool_call(tc)

            if content:
                text_chunks.append(content)
                # Only emit if we haven't seen tool calls yet
                # (once tool calls start, text is likely narration like "Let me check...")
                if not has_tool_call_delta:
                    self.events.send(
                        LLMResponseChunkEvent(
                            plugin_name="openrouter",
                            content_index=None,
                            item_id=chunk.id,
                            output_index=0,
                            sequence_number=i,
                            delta=content,
                        )
                    )
                    i += 1

            if finish_reason == "tool_calls":
                accumulated_tool_calls = self._finalize_chat_tool_calls()
            elif finish_reason == "stop":
                total_text = "".join(text_chunks)
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name="openrouter",
                        original=chunk,
                        text=total_text,
                        item_id=chunk.id,
                    )
                )
                llm_response = LLMResponseEvent(original=chunk, text=total_text)

        # Handle tool calls - the text before tool calls was narration, discard it
        if accumulated_tool_calls:
            return await self._handle_chat_tool_calls(
                accumulated_tool_calls, messages, tools, model
            )

        return llm_response

    async def _process_chat_response(
        self,
        response: ChatCompletion,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: Optional[str],
    ) -> LLMResponseEvent:
        """Process non-streaming Chat Completions response."""
        text = response.choices[0].message.content or ""
        llm_response = LLMResponseEvent(original=response, text=text)

        # Check for tool calls
        tool_calls = self._extract_chat_tool_calls(response)
        if tool_calls:
            return await self._handle_chat_tool_calls(
                tool_calls, messages, tools, model
            )

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name="openrouter",
                original=response,
                text=text,
                item_id=response.id,
            )
        )
        return llm_response

    def _accumulate_chat_tool_call(self, tc_chunk: Any) -> None:
        """Accumulate tool call data from streaming chunks."""
        idx = tc_chunk.index
        if idx not in self._pending_tool_calls:
            self._pending_tool_calls[idx] = {
                "id": tc_chunk.id or "",
                "name": "",
                "arguments_parts": [],
            }

        pending = self._pending_tool_calls[idx]
        if tc_chunk.id:
            pending["id"] = tc_chunk.id
        if tc_chunk.function:
            if tc_chunk.function.name:
                pending["name"] = tc_chunk.function.name
            if tc_chunk.function.arguments:
                pending["arguments_parts"].append(tc_chunk.function.arguments)

    def _finalize_chat_tool_calls(self) -> List[NormalizedToolCallItem]:
        """Convert accumulated tool call chunks to normalized format."""
        tool_calls: List[NormalizedToolCallItem] = []
        for pending in self._pending_tool_calls.values():
            args_str = "".join(pending["arguments_parts"]).strip() or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call arguments: {args_str}")
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": pending["id"],
                "name": pending["name"],
                "arguments_json": args,
            }
            tool_calls.append(tool_call)
            logger.debug(f"Finalized tool call: {pending['name']} with args: {args}")

        self._pending_tool_calls = {}
        return tool_calls

    def _extract_chat_tool_calls(
        self, response: ChatCompletion
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from non-streaming Chat Completions response."""
        tool_calls: List[NormalizedToolCallItem] = []

        if not response.choices:
            return tool_calls

        message = response.choices[0].message
        if not message.tool_calls:
            return tool_calls

        for tc in message.tool_calls:
            if not isinstance(tc, ChatCompletionMessageToolCall) or not tc.function:
                continue

            args_str = tc.function.arguments or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": tc.id or "",
                "name": tc.function.name or "unknown",
                "arguments_json": args,
            }
            tool_calls.append(tool_call)

        return tool_calls

    async def _handle_chat_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        model: Optional[str],
    ) -> LLMResponseEvent:
        """Execute tool calls and get follow-up response (Chat Completions).

        Key behavior: We buffer ALL intermediate text and only emit the FINAL
        response (after all tool calls complete). This prevents the model from
        speaking "Now I'll search..." between each tool call.
        """
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)

        for tc in tool_calls:
            logger.debug(
                "Tool call requested: %s with args: %s",
                tc.get("name"),
                tc.get("arguments_json"),
            )

        for round_num in range(max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,  # type: ignore[arg-type]
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            # Build assistant message with tool_calls
            assistant_tool_calls = []
            tool_results = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue

                assistant_tool_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments_json", {})),
                        },
                    }
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": self._sanitize_tool_output(
                            err if err is not None else res
                        ),
                    }
                )

            if not tool_results:
                return llm_response

            # Add assistant message with tool_calls, then tool results
            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

            # Make follow-up request
            effective_model = model or self.model
            request_kwargs: Dict[str, Any] = {
                "messages": current_messages,
                "model": effective_model,
                "stream": True,
            }
            if tools:
                request_kwargs["tools"] = tools
                if self._is_auto_model(effective_model):
                    request_kwargs["extra_body"] = {"models": TOOL_SUPPORTING_MODELS}

            follow_up = await self.client.chat.completions.create(**request_kwargs)

            # Process follow-up streaming response
            text_chunks: list[str] = []
            self._pending_tool_calls = {}
            next_tool_calls: List[NormalizedToolCallItem] = []
            has_tool_call_delta = False
            seq = 0

            async for chunk in cast(AsyncStream[ChatCompletionChunk], follow_up):
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content
                finish_reason = choice.finish_reason

                if choice.delta.tool_calls:
                    has_tool_call_delta = True
                    for tc_delta in choice.delta.tool_calls:
                        self._accumulate_chat_tool_call(tc_delta)

                if content:
                    text_chunks.append(content)
                    # Stream text if no tool calls detected yet
                    if not has_tool_call_delta:
                        self.events.send(
                            LLMResponseChunkEvent(
                                plugin_name="openrouter",
                                content_index=None,
                                item_id=chunk.id,
                                output_index=0,
                                sequence_number=seq,
                                delta=content,
                            )
                        )
                        seq += 1

                if finish_reason == "tool_calls":
                    next_tool_calls = self._finalize_chat_tool_calls()
                elif finish_reason == "stop":
                    total_text = "".join(text_chunks)
                    self.events.send(
                        LLMResponseCompletedEvent(
                            plugin_name="openrouter",
                            original=chunk,
                            text=total_text,
                            item_id=chunk.id,
                        )
                    )
                    llm_response = LLMResponseEvent(original=chunk, text=total_text)

            # If more tool calls, continue the loop
            if next_tool_calls and round_num < max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            return llm_response

        return llm_response
