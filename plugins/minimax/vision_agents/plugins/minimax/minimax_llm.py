"""MiniMax LLM implementation using OpenAI-compatible Chat Completions API.

MiniMax provides an OpenAI-compatible API at https://api.minimax.io/v1.
Supported models: MiniMax-M3 (default), MiniMax-M2.7, MiniMax-M2.7-highspeed.
"""

import json
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
)
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema

logger = logging.getLogger(__name__)


PLUGIN_NAME = "minimax"

DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M3"


class MiniMaxLLM(LLM):
    """MiniMax LLM using OpenAI-compatible Chat Completions API.

    MiniMax provides high-performance language models accessible through
    an OpenAI-compatible API. This plugin uses the Chat Completions endpoint.

    Examples:

        from vision_agents.plugins import minimax
        llm = minimax.LLM()
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
        """Initialize the MiniMaxLLM class.

        Args:
            model: The MiniMax model to use.
                Defaults to ``"MiniMax-M3"`` (latest flagship, 512K context).
                Supported models: ``"MiniMax-M3"``, ``"MiniMax-M2.7"``,
                ``"MiniMax-M2.7-highspeed"``.
            api_key: Optional API key. Defaults to ``MINIMAX_API_KEY`` env var.
            base_url: Optional base URL. Defaults to
                ``https://api.minimax.io/v1`` (overseas endpoint).
            client: Optional ``AsyncOpenAI`` client for dependency injection.
            max_tokens: This sets the upper limit for the number of tokens the
                model can generate in response.
            tools_max_rounds: max calling rounds for multi-hop tool call.
        """
        super().__init__()
        self.model = model
        self._max_tokens = max_tokens
        self._tools_max_rounds = max(tools_max_rounds, 1)
        # For tracking streaming tool calls in Chat Completions mode
        self._pending_tool_calls: Dict[int, Dict[str, Any]] = {}

        if client is not None:
            self._client = client
        else:
            if api_key is None:
                api_key = os.environ.get("MINIMAX_API_KEY")
            if base_url is None:
                base_url = os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL)
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        simple_response is a standardized way to create an LLM response.

        This method is also called every time a new STT transcript is received.

        Args:
            text: The text to respond to.
            participant: the Participant object, optional. If not provided, the
                message will be sent with the "user" role.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            yield LLMResponseFinal(original=None, text="")
            return

        # The simple_response is called directly without providing the participant -
        # assuming it's an initial prompt.
        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = await self._build_model_request()
        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        async for item in self._create_response_internal(
            messages=messages, tools=tools_param
        ):
            yield item

    async def close(self) -> None:
        await self._client.close()

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        # Add Agent's instructions as system prompt.
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})

        # Add all messages from the conversation to the prompt
        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append({"role": message.role, "content": message.content})
        return messages

    def _convert_tools_to_provider_format(
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

            func_spec: Dict[str, Any] = {
                "name": name,
                "description": description,
                "parameters": params,
            }
            result.append({"type": "function", "function": func_spec})
        return result

    async def _create_response_internal(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Internal method to create response with tool handling loop."""
        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "stream": True,
            "temperature": 1.0,
        }
        if self._max_tokens is not None:
            request_kwargs["max_tokens"] = self._max_tokens
        if tools:
            request_kwargs["tools"] = tools

        request_start_time = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as e:
            logger.exception(
                f'Failed to get a response from MiniMax LLM "{self.model}"'
            )
            self.on_llm_error(error=e)
            yield LLMResponseFinal(original=None, text="")
            return

        async for item in self._process_streaming_response(
            response, messages, tools, request_start_time
        ):
            yield item

    async def _process_streaming_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        request_start_time: float,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Process a streaming response, handling tool calls if present."""
        text_chunks: list[str] = []
        self._pending_tool_calls = {}
        accumulated_tool_calls: list[NormalizedToolCallItem] = []
        sequence_number = 0
        first_token_time: Optional[float] = None
        last_chunk: ChatCompletionChunk | None = None
        has_tool_call_delta_seen = False

        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            last_chunk = chunk
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason
            if choice.delta.tool_calls:
                has_tool_call_delta_seen = True
                for tc in choice.delta.tool_calls:
                    self._accumulate_tool_call_chunk(tc)

            if content:
                is_first = first_token_time is None
                ttft_ms = None
                if is_first:
                    first_token_time = time.perf_counter()
                    ttft_ms = (first_token_time - request_start_time) * 1000

                text_chunks.append(content)
                if not has_tool_call_delta_seen:
                    yield LLMResponseDelta(
                        content_index=None,
                        item_id=chunk.id,
                        output_index=0,
                        sequence_number=sequence_number,
                        delta=content,
                        is_first_chunk=is_first,
                        time_to_first_token_ms=ttft_ms,
                    )
                    sequence_number += 1

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )

                if finish_reason == "tool_calls" and self._pending_tool_calls:
                    accumulated_tool_calls = self._finalize_pending_tool_calls()

        total_text = "".join(text_chunks)

        # Handle tool calls if any were accumulated
        if accumulated_tool_calls:
            async for item in self._handle_tool_calls(
                accumulated_tool_calls,
                messages,
                tools,
                request_start_time,
                first_token_time,
                sequence_number,
                initial_text=total_text,
            ):
                yield item
            return

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start_time) * 1000

        item_id = last_chunk.id if last_chunk is not None else None
        yield LLMResponseFinal(
            original=last_chunk,
            text=total_text,
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            model=self.model,
        )

    def _accumulate_tool_call_chunk(self, tc_chunk: ChoiceDeltaToolCall) -> None:
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

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        request_start_time: float,
        first_token_time: Optional[float],
        sequence_number: int,
        initial_text: str = "",
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Execute tool calls and get follow-up response."""
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)
        last_chunk: ChatCompletionChunk | None = None
        all_text_parts: list[str] = [initial_text] if initial_text else []

        for tc in tool_calls:
            logger.debug(
                "Tool call requested: %s with args: %s",
                tc.get("name"),
                tc.get("arguments_json"),
            )

        for round_num in range(self._tools_max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
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
                break

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
            request_kwargs: dict[str, Any] = {
                "messages": current_messages,
                "model": self.model,
                "stream": True,
                "temperature": 1.0,
            }
            if self._max_tokens is not None:
                request_kwargs["max_tokens"] = self._max_tokens
            if tools:
                request_kwargs["tools"] = tools

            try:
                follow_up = await self._client.chat.completions.create(**request_kwargs)
            except Exception as e:
                logger.exception("Failed to get follow-up response from MiniMax")
                self.on_llm_error(error=e)
                break

            text_chunks: list[str] = []
            self._pending_tool_calls = {}
            next_tool_calls: list[NormalizedToolCallItem] = []
            has_tool_call_delta_seen = False

            async for chunk in cast(AsyncStream[ChatCompletionChunk], follow_up):
                last_chunk = chunk
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content
                finish_reason = choice.finish_reason

                if choice.delta.tool_calls:
                    has_tool_call_delta_seen = True
                    for tc_delta in choice.delta.tool_calls:
                        self._accumulate_tool_call_chunk(tc_delta)

                if content:
                    is_first = first_token_time is None
                    ttft_ms = None
                    if is_first:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - request_start_time) * 1000

                    text_chunks.append(content)
                    if not has_tool_call_delta_seen:
                        yield LLMResponseDelta(
                            content_index=None,
                            item_id=chunk.id,
                            output_index=0,
                            sequence_number=sequence_number,
                            delta=content,
                            is_first_chunk=is_first,
                            time_to_first_token_ms=ttft_ms,
                        )
                        sequence_number += 1

                if finish_reason == "tool_calls" and self._pending_tool_calls:
                    next_tool_calls = self._finalize_pending_tool_calls()

            all_text_parts.extend(text_chunks)

            if next_tool_calls and round_num < self._tools_max_rounds - 1:
                current_tool_calls = next_tool_calls
                continue

            break

        latency_ms = (time.perf_counter() - request_start_time) * 1000
        ttft_ms = None
        if first_token_time is not None:
            ttft_ms = (first_token_time - request_start_time) * 1000

        item_id = last_chunk.id if last_chunk is not None else None
        yield LLMResponseFinal(
            original=last_chunk,
            text="".join(all_text_parts),
            item_id=item_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            model=self.model,
        )
