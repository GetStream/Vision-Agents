"""Shared tool-calling and response processing for HuggingFace Inference API plugins.

Used by both ``HuggingFaceLLM`` and ``HuggingFaceVLM`` to avoid duplicating
the streaming response processing and multi-round tool execution loop.
"""

import json
import logging
import time
from typing import Any, Optional

from huggingface_hub import AsyncInferenceClient, InferenceTimeoutError
from huggingface_hub.errors import HfHubHTTPError
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem

from . import events
from ._tool_call_loop import (
    convert_tools_to_chat_completions_format,
    run_tool_call_loop,
)

logger = logging.getLogger(__name__)

convert_tools_to_hf_format = convert_tools_to_chat_completions_format


def accumulate_tool_call_chunk(
    pending: dict[int, dict[str, Any]], tc_chunk: Any
) -> None:
    """Accumulate tool-call data from a single streaming delta chunk."""
    idx = tc_chunk.index
    if idx not in pending:
        pending[idx] = {"id": tc_chunk.id or "", "name": "", "arguments_parts": []}
    entry = pending[idx]
    if tc_chunk.id:
        entry["id"] = tc_chunk.id
    if tc_chunk.function:
        if tc_chunk.function.name:
            entry["name"] = tc_chunk.function.name
        if tc_chunk.function.arguments:
            entry["arguments_parts"].append(tc_chunk.function.arguments)


def finalize_pending_tool_calls(
    pending: dict[int, dict[str, Any]],
) -> list[NormalizedToolCallItem]:
    """Convert accumulated streaming chunks into normalized tool calls."""
    tool_calls: list[NormalizedToolCallItem] = []
    for entry in pending.values():
        args_str = "".join(entry["arguments_parts"]).strip() or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            {
                "type": "tool_call",
                "id": entry["id"],
                "name": entry["name"],
                "arguments_json": args,
            }
        )
    return tool_calls


def extract_tool_calls_from_hf_response(
    response: Any,
) -> list[NormalizedToolCallItem]:
    """Extract tool calls from a non-streaming Chat Completions response."""
    tool_calls: list[NormalizedToolCallItem] = []
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
        tool_calls.append(
            {
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "arguments_json": args,
            }
        )
    return tool_calls


async def create_hf_response(
    llm: LLM,
    client: AsyncInferenceClient,
    model_id: str,
    plugin_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    stream: bool = True,
    **kwargs: Any,
) -> LLMResponseEvent:
    """Send a request to the HF Inference API, process the response, and handle tool calls.

    This is the single entry-point shared by ``HuggingFaceLLM`` and
    ``HuggingFaceVLM``.  It emits all standard LLM lifecycle events
    (request-started, chunks, response-completed) and transparently runs a
    multi-round tool-execution loop when the model returns tool calls.
    """
    request_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model_id,
        "stream": stream,
    }
    if tools:
        request_kwargs["tools"] = tools

    llm.events.send(
        LLMRequestStartedEvent(
            plugin_name=plugin_name,
            model=model_id,
            streaming=stream,
        )
    )

    request_start = time.perf_counter()

    try:
        response = await client.chat.completions.create(**request_kwargs)
    except (HfHubHTTPError, InferenceTimeoutError, OSError) as e:
        logger.exception(f'Failed to get a response from the LLM "{model_id}"')
        llm.events.send(
            events.LLMErrorEvent(
                plugin_name=plugin_name,
                error_message=str(e),
                event_data=e,
            )
        )
        return LLMResponseEvent(original=None, text="", exception=e)

    if stream:
        return await _process_streaming(
            llm,
            client,
            model_id,
            plugin_name,
            response,
            messages,
            tools,
            kwargs,
            request_start,
        )
    return await _process_non_streaming(
        llm,
        client,
        model_id,
        plugin_name,
        response,
        messages,
        tools,
        kwargs,
        request_start,
    )


async def _process_streaming(
    llm: LLM,
    client: AsyncInferenceClient,
    model_id: str,
    plugin_name: str,
    response: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
    request_start: float,
) -> LLMResponseEvent:
    llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
    text_chunks: list[str] = []
    total_text = ""
    pending: dict[int, dict[str, Any]] = {}
    accumulated_tool_calls: list[NormalizedToolCallItem] = []
    i = 0
    chunk_id = ""
    first_token_time: Optional[float] = None

    async for chunk in response:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        content = choice.delta.content if choice.delta else None
        finish_reason = choice.finish_reason
        chunk_id = chunk.id if chunk.id else chunk_id

        if choice.delta and choice.delta.tool_calls:
            for tc_chunk in choice.delta.tool_calls:
                accumulate_tool_call_chunk(pending, tc_chunk)

        if content:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            is_first = len(text_chunks) == 0
            ttft_ms = None
            if is_first:
                ttft_ms = (first_token_time - request_start) * 1000

            text_chunks.append(content)
            llm.events.send(
                LLMResponseChunkEvent(
                    plugin_name=plugin_name,
                    content_index=None,
                    item_id=chunk_id,
                    output_index=0,
                    sequence_number=i,
                    delta=content,
                    is_first_chunk=is_first,
                    time_to_first_token_ms=ttft_ms,
                )
            )

        if finish_reason:
            if finish_reason in ("length", "content"):
                logger.warning(
                    f'The model finished the response due to reason "{finish_reason}"'
                )

            if finish_reason == "tool_calls":
                accumulated_tool_calls = finalize_pending_tool_calls(pending)

            total_text = "".join(text_chunks)

            if finish_reason != "tool_calls":
                latency_ms = (time.perf_counter() - request_start) * 1000
                ttft_ms_final = None
                if first_token_time is not None:
                    ttft_ms_final = (first_token_time - request_start) * 1000

                llm.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=plugin_name,
                        original=chunk,
                        text=total_text,
                        item_id=chunk_id,
                        latency_ms=latency_ms,
                        time_to_first_token_ms=ttft_ms_final,
                        model=model_id,
                    )
                )

        llm_response = LLMResponseEvent(original=chunk, text=total_text)
        i += 1

    if accumulated_tool_calls:
        return await _handle_tool_calls(
            llm,
            client,
            model_id,
            plugin_name,
            accumulated_tool_calls,
            messages,
            tools,
            kwargs,
        )

    return llm_response


async def _process_non_streaming(
    llm: LLM,
    client: AsyncInferenceClient,
    model_id: str,
    plugin_name: str,
    response: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
    request_start: float,
) -> LLMResponseEvent:
    latency_ms = (time.perf_counter() - request_start) * 1000
    text = response.choices[0].message.content or ""

    tool_calls = extract_tool_calls_from_hf_response(response)
    if tool_calls:
        return await _handle_tool_calls(
            llm,
            client,
            model_id,
            plugin_name,
            tool_calls,
            messages,
            tools,
            kwargs,
        )

    llm.events.send(
        LLMResponseCompletedEvent(
            plugin_name=plugin_name,
            original=response,
            text=text,
            item_id=response.id,
            latency_ms=latency_ms,
            model=model_id,
        )
    )
    return LLMResponseEvent(original=response, text=text)


async def _handle_tool_calls(
    llm: LLM,
    client: AsyncInferenceClient,
    model_id: str,
    plugin_name: str,
    tool_calls: list[NormalizedToolCallItem],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
) -> LLMResponseEvent:
    """Execute tool calls and stream follow-up responses (up to 3 rounds)."""

    async def _generate_followup(
        current_messages: list[dict[str, Any]],
    ) -> tuple[LLMResponseEvent, list[NormalizedToolCallItem]]:
        request_kwargs: dict[str, Any] = {
            "messages": current_messages,
            "model": kwargs.get("model", model_id),
            "stream": True,
        }
        if tools:
            request_kwargs["tools"] = tools

        try:
            follow_up = await client.chat.completions.create(**request_kwargs)
        except (HfHubHTTPError, InferenceTimeoutError, OSError) as e:
            logger.exception("Failed to get follow-up response after tool execution")
            llm.events.send(
                events.LLMErrorEvent(
                    plugin_name=plugin_name,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text=""), []

        text_chunks: list[str] = []
        pending: dict[int, dict[str, Any]] = {}
        next_tool_calls: list[NormalizedToolCallItem] = []
        i = 0
        chunk_id = ""
        llm_response = LLMResponseEvent(original=None, text="")

        async for chunk in follow_up:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content if choice.delta else None
            finish_reason = choice.finish_reason
            chunk_id = chunk.id if chunk.id else chunk_id

            if choice.delta and choice.delta.tool_calls:
                for tc_chunk in choice.delta.tool_calls:
                    accumulate_tool_call_chunk(pending, tc_chunk)

            if content:
                text_chunks.append(content)
                llm.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=plugin_name,
                        content_index=None,
                        item_id=chunk_id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                    )
                )

            if finish_reason:
                if finish_reason == "tool_calls":
                    next_tool_calls = finalize_pending_tool_calls(pending)

                total_text = "".join(text_chunks)

                if finish_reason != "tool_calls":
                    llm.events.send(
                        LLMResponseCompletedEvent(
                            plugin_name=plugin_name,
                            original=chunk,
                            text=total_text,
                            item_id=chunk_id,
                        )
                    )
                llm_response = LLMResponseEvent(original=chunk, text=total_text)

            i += 1

        return llm_response, next_tool_calls

    return await run_tool_call_loop(llm, tool_calls, messages, _generate_followup)
