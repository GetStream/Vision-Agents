"""OpenRouter LLM implementation with auto-detection of API format.

OpenRouter supports many models from different providers. This implementation
automatically selects the appropriate API format based on the model:
- OpenAI models (openai/*): Use Responses API for best native support
- All other models: Use Chat Completions API (the industry standard)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, cast

from openai import AsyncStream
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.responses import Response as OpenAIResponse
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
    """OpenRouter LLM with automatic API format selection.

    Extends OpenAI LLM with OpenRouter-specific handling:
    - Auto-detects model provider and uses appropriate API format
    - OpenAI models use Responses API (native support)
    - Non-OpenAI models use Chat Completions API (universal standard)
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

    def _is_openai_model(self, model: Optional[str] = None) -> bool:
        """Check if the model is an OpenAI model (uses Responses API)."""
        model = model or self.model
        return model.startswith("openai/")

    def _is_auto_model(self, model: Optional[str] = None) -> bool:
        """Check if the model is a meta/auto model that may not support tools."""
        model = model or self.model
        return model in ("openrouter/auto",)

    async def create_conversation(self):
        """No-op for OpenRouter (no server-side conversation IDs)."""
        pass

    async def create_response(self, *args: Any, **kwargs: Any) -> LLMResponseEvent:
        """Create a response using the appropriate API based on model.

        For OpenAI models: Uses parent's Responses API (native, proven to work)
        For other models: Uses Chat Completions API
        """
        model = kwargs.get("model", self.model)

        if self._is_openai_model(model):
            return await super().create_response(*args, **kwargs)
        else:
            return await self._create_response_chat_completions(*args, **kwargs)

    # =========================================================================
    # Responses API path (for OpenAI models)
    # =========================================================================

    async def _create_response_responses_api(
        self, *args: Any, **kwargs: Any
    ) -> LLMResponseEvent[OpenAIResponse]:
        """Create response using Responses API (for OpenAI models).

        Handles everything directly instead of delegating to parent to ensure
        correct event emission and conversation handling for OpenRouter.
        """
        # Get and normalize input to list format
        user_input = kwargs.get("input", args[0] if args else "Hello")
        if not isinstance(user_input, list):
            user_input = [{"content": str(user_input), "role": "user", "type": "message"}]

        # Build messages: instructions + conversation history + new input
        messages: List[Dict[str, Any]] = []

        # Add system instructions
        if self._instructions:
            messages.append({"content": self._instructions, "role": "system", "type": "message"})

        # Add conversation history
        if self._conversation:
            for m in self._conversation.messages:
                if isinstance(m.original, dict):
                    messages.append(m.original)
                else:
                    messages.append({"content": m.content, "role": m.role, "type": "message"})

        # Add new user input
        messages.extend(user_input)

        # Build request
        request_kwargs: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "input": messages,
            "stream": True,
        }

        # Add tools if available
        tools_spec = self._get_tools_for_provider()
        if tools_spec:
            request_kwargs["tools"] = self._convert_tools_to_provider_format(tools_spec)  # type: ignore

        # Make the API call
        response = await self.client.responses.create(**request_kwargs)

        # Process streaming response
        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None
        pending_tool_calls: List[NormalizedToolCallItem] = []
        seen: set[tuple] = set()

        if isinstance(response, OpenAIResponse):
            # Non-streaming response
            llm_response = LLMResponseEvent[OpenAIResponse](response, response.output_text)
            pending_tool_calls = self._extract_tool_calls_from_response(response)
        else:
            # Streaming response
            async for event in response:
                result = self._standardize_and_emit_event(event)
                if result is not None:
                    llm_response = result

                # Collect tool calls
                if getattr(event, "type", "") == "response.completed":
                    for c in self._extract_tool_calls_from_response(event.response):
                        key = (c["id"], c["name"], json.dumps(c["arguments_json"], sort_keys=True))
                        if key not in seen:
                            pending_tool_calls.append(c)
                            seen.add(key)

        # Update conversation history with user input
        if self._conversation:
            normalized = self._normalize_message(user_input)
            for msg in normalized:
                self._conversation.messages.append(msg)

        # Handle tool calls if any
        if pending_tool_calls:
            return await self._handle_tool_calls(pending_tool_calls, request_kwargs)

        return llm_response or self._empty_response()

    def add_conversation_history(self, kwargs):
        """Add conversation history to the request input (for Responses API)."""
        new_messages = kwargs["input"]
        if not isinstance(new_messages, list):
            new_messages = [dict(content=new_messages, role="user", type="message")]

        if self._conversation:
            # Ensure old messages are properly formatted dicts
            old_messages = []
            for m in self._conversation.messages:
                if isinstance(m.original, dict):
                    old_messages.append(m.original)
                else:
                    # Fallback: create proper message format
                    old_messages.append({"content": m.content, "role": m.role, "type": "message"})
            kwargs["input"] = old_messages + new_messages
            # Add messages to conversation
            normalized_messages = self._normalize_message(new_messages)
            for msg in normalized_messages:
                self._conversation.messages.append(msg)

    def _empty_response(self) -> LLMResponseEvent[OpenAIResponse]:
        """Return an empty LLMResponseEvent."""
        return LLMResponseEvent[OpenAIResponse](None, "")  # type: ignore[arg-type]

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI Responses API format WITHOUT strict mode.

        OpenAI's strict mode requires ALL properties to be in 'required',
        but MCP tools often have optional parameters. We disable strict mode
        to allow optional parameters to work correctly.
        """
        out = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            # Don't set additionalProperties: false or strict: true
            # This allows MCP tools with optional parameters to work

            out.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": params,
                # NO "strict": True - this breaks MCP tools with optional params
            })
        return out

    async def _handle_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        original_kwargs: Dict[str, Any],
    ) -> LLMResponseEvent[OpenAIResponse]:
        """Handle tool calls for Responses API (OpenAI models)."""
        llm_response: Optional[LLMResponseEvent[OpenAIResponse]] = None
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()

        # Normalize conversation context - ensure all items are properly formatted dicts
        raw_context = original_kwargs.get("input", [])
        if not isinstance(raw_context, list):
            raw_context = [raw_context]

        conversation_context: List[Dict[str, Any]] = []
        for item in raw_context:
            if isinstance(item, dict):
                conversation_context.append(item)
            else:
                # String or other type - wrap in proper message format
                conversation_context.append(
                    {"content": str(item), "role": "user", "type": "message"}
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

            function_calls, tool_outputs = [], []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue
                function_calls.append({
                    "type": "function_call",
                    "call_id": cid,
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments_json", {})),
                })
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": self._sanitize_tool_output(err if err is not None else res),
                })

            if not tool_outputs:
                return llm_response or self._empty_response()

            follow_up_input = conversation_context + function_calls + tool_outputs
            follow_up_kwargs: Dict[str, Any] = {
                "model": original_kwargs.get("model", self.model),
                "input": follow_up_input,
                "stream": True,
            }

            tools_spec = self._get_tools_for_provider()
            if tools_spec:
                follow_up_kwargs["tools"] = self._convert_tools_to_provider_format(
                    tools_spec  # type: ignore[arg-type]
                )

            follow_up_response = await self.client.responses.create(**follow_up_kwargs)

            if isinstance(follow_up_response, OpenAIResponse):
                llm_response = LLMResponseEvent[OpenAIResponse](
                    follow_up_response, follow_up_response.output_text
                )
                next_tool_calls = self._extract_tool_calls_from_response(follow_up_response)
                if next_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = next_tool_calls
                    conversation_context = follow_up_input
                    continue
                return llm_response

            elif hasattr(follow_up_response, "__aiter__"):
                pending_tool_calls: List[NormalizedToolCallItem] = []
                async for event in follow_up_response:
                    result = self._standardize_and_emit_event(event)
                    if result is not None:
                        llm_response = result

                    if getattr(event, "type", "") == "response.completed":
                        for c in self._extract_tool_calls_from_response(event.response):
                            key = (c["id"], c["name"], json.dumps(c["arguments_json"], sort_keys=True))
                            if key not in seen:
                                pending_tool_calls.append(c)
                                seen.add(key)

                if pending_tool_calls and round_num < max_rounds - 1:
                    current_tool_calls = pending_tool_calls
                    conversation_context = follow_up_input
                    continue
                return llm_response or self._empty_response()
            else:
                return self._empty_response()

        return llm_response or self._empty_response()

    # =========================================================================
    # Chat Completions API path (for non-OpenAI models)
    # =========================================================================

    async def _create_response_chat_completions(
        self, *args: Any, **kwargs: Any
    ) -> LLMResponseEvent:
        """Create response using Chat Completions API (for non-OpenAI models)."""
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
                    Message(original={"role": "user", "content": user_input}, content=user_input, role="user")
                )
            # Add assistant response
            if response.text:
                self._conversation.messages.append(
                    Message(original={"role": "assistant", "content": response.text}, content=response.text, role="assistant")
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
                        messages.append({
                            "role": "tool",
                            "tool_call_id": item.get("call_id", ""),
                            "content": item.get("output", ""),
                        })
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
        """Convert ToolSchema to Chat Completions API format."""
        result = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            # Build the function spec
            func_spec: Dict[str, Any] = {
                "name": name,
                "description": description,
                "parameters": params,
            }

            # Add strict mode if the schema has required fields (helps models follow schema)
            if params.get("required"):
                func_spec["strict"] = True
                # Strict mode requires additionalProperties: false
                params.setdefault("additionalProperties", False)

            result.append({
                "type": "function",
                "function": func_spec,
            })
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
            return await self._handle_chat_tool_calls(tool_calls, messages, tools, model)

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

    def _extract_chat_tool_calls(self, response: ChatCompletion) -> List[NormalizedToolCallItem]:
        """Extract tool calls from non-streaming Chat Completions response."""
        tool_calls: List[NormalizedToolCallItem] = []

        if not response.choices:
            return tool_calls

        message = response.choices[0].message
        if not message.tool_calls:
            return tool_calls

        for tc in message.tool_calls:
            # Use getattr for safer access across different OpenAI SDK versions
            func = getattr(tc, "function", None)
            if not func:
                continue

            args_str = getattr(func, "arguments", "{}") or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": getattr(tc, "id", ""),
                "name": getattr(func, "name", "unknown"),
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

        # Debug: Log what tool calls the model is making
        for tc in tool_calls:
            logger.info(f"🔧 Model requesting tool: {tc.get('name')} with args: {tc.get('arguments_json')}")

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

                assistant_tool_calls.append({
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("arguments_json", {})),
                    },
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": self._sanitize_tool_output(err if err is not None else res),
                })

            if not tool_results:
                return llm_response

            # Add assistant message with tool_calls, then tool results
            current_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })
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
                    for tc in choice.delta.tool_calls:
                        self._accumulate_chat_tool_call(tc)

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
