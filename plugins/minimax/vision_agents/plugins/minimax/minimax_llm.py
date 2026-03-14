"""MiniMax LLM implementation using OpenAI-compatible Chat Completions API.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
Supported models: MiniMax-M2.5, MiniMax-M2.5-highspeed.
"""

import json
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional, cast

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    AsyncStream,
    RateLimitError,
)
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

from . import events

logger = logging.getLogger(__name__)

PLUGIN_NAME = "minimax"

DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxLLM(LLM):
    """MiniMax LLM using OpenAI-compatible Chat Completions API.

    MiniMax provides high-performance language models accessible through
    an OpenAI-compatible API. This plugin uses the Chat Completions endpoint.

    Examples:

        from vision_agents.plugins import minimax
        llm = minimax.LLM(model="MiniMax-M2.5")
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        api_key: str | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize the MiniMaxLLM class.

        Args:
            model: The MiniMax model to use. Supported: MiniMax-M2.5, MiniMax-M2.5-highspeed.
            api_key: Optional API key. Defaults to MINIMAX_API_KEY env var.
            base_url: Optional base URL. Defaults to https://api.minimax.io/v1.
            client: Optional AsyncOpenAI client for dependency injection.
        """
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self._pending_tool_calls: Dict[int, Dict[str, Any]] = {}

        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY")

        if base_url is None:
            base_url = os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL)

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> LLMResponseEvent:
        """Create a standardized LLM response.

        Args:
            text: The text to respond to.
            participant: Optional participant object.
        """
        messages: list[dict[str, Any]] = []

        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})

        messages.append({"role": "user", "content": text})

        return await self.create_response(messages=messages)

    async def create_response(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        input: Any | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Create a response using MiniMax Chat Completions API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            input: Alternative to messages - will be converted to messages format.
            stream: Whether to stream the response.
            **kwargs: Additional arguments passed to the API.
        """
        if messages is None:
            if input is not None:
                messages = self._input_to_messages(input)
            else:
                messages = self._build_model_request()

        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        return await self._create_response_internal(
            messages=messages,
            tools=tools_param,
            stream=stream,
            **kwargs,
        )

    async def _create_response_internal(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Internal method to create response with tool handling loop."""
        request_kwargs: Dict[str, Any] = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "stream": stream,
            "temperature": kwargs.get("temperature", 1.0),
        }
        if tools:
            request_kwargs["tools"] = tools

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=request_kwargs["model"],
                streaming=stream,
            )
        )

        request_start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except (APIError, APIConnectionError, APIStatusError, RateLimitError) as e:
            logger.exception(f'Failed to get a response from MiniMax model "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data={
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
            return LLMResponseEvent(original=None, text="")

        if stream:
            return await self._process_streaming_response(
                response, messages, tools, kwargs, request_start_time
            )
        else:
            return await self._process_non_streaming_response(
                response, messages, tools, kwargs, request_start_time
            )

    async def _process_streaming_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        kwargs: Dict[str, Any],
        request_start_time: float,
    ) -> LLMResponseEvent:
        """Process a streaming response, handling tool calls if present."""
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        text_chunks: list[str] = []
        total_text = ""
        self._pending_tool_calls = {}
        accumulated_tool_calls: List[NormalizedToolCallItem] = []
        i = 0
        first_token_time: float | None = None

        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            if choice.delta.tool_calls:
                for tc in choice.delta.tool_calls:
                    self._accumulate_tool_call_chunk(tc)

            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                is_first = len(text_chunks) == 0
                ttft_ms = None
                if is_first:
                    ttft_ms = (first_token_time - request_start_time) * 1000

                text_chunks.append(content)
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        content_index=None,
                        item_id=chunk.id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                        is_first_chunk=is_first,
                        time_to_first_token_ms=ttft_ms,
                    )
                )

            if finish_reason:
                if finish_reason == "tool_calls":
                    accumulated_tool_calls = self._finalize_pending_tool_calls()

                total_text = "".join(text_chunks)
                latency_ms = (time.perf_counter() - request_start_time) * 1000
                ttft_ms_final = None
                if first_token_time is not None:
                    ttft_ms_final = (first_token_time - request_start_time) * 1000

                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk.id,
                        latency_ms=latency_ms,
                        time_to_first_token_ms=ttft_ms_final,
                        model=self.model,
                    )
                )

            llm_response = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        if accumulated_tool_calls:
            return await self._handle_tool_calls(
                accumulated_tool_calls, messages, tools, kwargs
            )

        return llm_response

    async def _process_non_streaming_response(
        self,
        response: ChatCompletion,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        kwargs: Dict[str, Any],
        request_start_time: float,
    ) -> LLMResponseEvent:
        """Process a non-streaming response, handling tool calls if present."""
        latency_ms = (time.perf_counter() - request_start_time) * 1000
        if not response.choices:
            return LLMResponseEvent(original=response, text="")
        text = response.choices[0].message.content or ""
        llm_response = LLMResponseEvent(original=response, text=text)

        tool_calls = self._extract_tool_calls_from_response(response)
        if tool_calls:
            return await self._handle_tool_calls(tool_calls, messages, tools, kwargs)

        input_tokens: int | None = None
        output_tokens: int | None = None
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=response,
                text=text,
                item_id=response.id,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0)
                if input_tokens or output_tokens
                else None,
                model=self.model,
            )
        )
        return llm_response

    def _build_model_request(self) -> list[dict[str, Any]]:
        """Build messages list from instructions and conversation history."""
        messages: list[dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})
        return messages

    def _input_to_messages(self, input_value: Any) -> list[dict[str, Any]]:
        """Convert input parameter to messages format."""
        messages: list[dict[str, Any]] = []

        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        if isinstance(input_value, str):
            messages.append({"role": "user", "content": input_value})
        elif isinstance(input_value, list):
            for item in input_value:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if item.get("type") == "function_call_output":
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

    def _accumulate_tool_call_chunk(self, tc_chunk: Any) -> None:
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

    def _finalize_pending_tool_calls(self) -> List[NormalizedToolCallItem]:
        """Convert accumulated tool call chunks into normalized tool calls."""
        tool_calls: List[NormalizedToolCallItem] = []
        for pending in self._pending_tool_calls.values():
            args_str = "".join(pending["arguments_parts"]).strip() or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": pending["id"],
                "name": pending["name"],
                "arguments_json": args,
            }
            tool_calls.append(tool_call)

        self._pending_tool_calls = {}
        return tool_calls

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        """Convert ToolSchema objects to Chat Completions API format."""
        result = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params,
                    },
                }
            )
        return result

    def _extract_tool_calls_from_response(
        self, response: Any
    ) -> List[NormalizedToolCallItem]:
        """Extract tool calls from a non-streaming Chat Completions response."""
        tool_calls: List[NormalizedToolCallItem] = []

        if not response.choices:
            return tool_calls

        message = response.choices[0].message
        if not message.tool_calls:
            return tool_calls

        for tc in message.tool_calls:
            args_str = tc.function.arguments or "{}"
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}

            tool_call: NormalizedToolCallItem = {
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "arguments_json": args,
            }
            tool_calls.append(tool_call)

        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Execute tool calls and get follow-up response."""
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)

        for round_num in range(max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

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

            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

            request_kwargs: Dict[str, Any] = {
                "messages": current_messages,
                "model": kwargs.get("model", self.model),
                "stream": True,
                "temperature": kwargs.get("temperature", 1.0),
            }
            if tools:
                request_kwargs["tools"] = tools

            try:
                follow_up = await self._client.chat.completions.create(**request_kwargs)
            except (APIError, APIConnectionError, APIStatusError, RateLimitError) as e:
                logger.exception("Failed to get follow-up response from MiniMax")
                self.events.send(
                    events.LLMErrorEvent(
                        plugin_name=PLUGIN_NAME,
                        error_message=str(e),
                        event_data={
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )
                return llm_response

            text_chunks: list[str] = []
            self._pending_tool_calls = {}
            next_tool_calls: List[NormalizedToolCallItem] = []
            i = 0

            async for chunk in cast(AsyncStream[ChatCompletionChunk], follow_up):
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content
                finish_reason = choice.finish_reason

                if choice.delta.tool_calls:
                    for tc in choice.delta.tool_calls:
                        self._accumulate_tool_call_chunk(tc)

                if content:
                    text_chunks.append(content)
                    self.events.send(
                        LLMResponseChunkEvent(
                            plugin_name=PLUGIN_NAME,
                            content_index=None,
                            item_id=chunk.id,
                            output_index=0,
                            sequence_number=i,
                            delta=content,
                        )
                    )

                if finish_reason:
                    if finish_reason == "tool_calls":
                        next_tool_calls = self._finalize_pending_tool_calls()

                    total_text = "".join(text_chunks)
                    self.events.send(
                        LLMResponseCompletedEvent(
                            plugin_name=PLUGIN_NAME,
                            original=chunk,
                            text=total_text,
                            item_id=chunk.id,
                        )
                    )
                    llm_response = LLMResponseEvent(original=chunk, text=total_text)

                i += 1

            if next_tool_calls and round_num < max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            return llm_response

        return llm_response
