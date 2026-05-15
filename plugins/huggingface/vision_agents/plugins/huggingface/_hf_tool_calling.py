"""Shared tool-calling and response processing for HuggingFace Inference API plugins.

Used by both ``HuggingFaceLLM`` and ``HuggingFaceVLM`` to avoid duplicating
the streaming response processing and multi-round tool execution loop.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Optional

from aiohttp import ClientResponseError
from huggingface_hub import AsyncInferenceClient, InferenceTimeoutError
from huggingface_hub.errors import HfHubHTTPError
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem

from ._tool_call_loop import convert_tools_to_chat_completions_format

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
) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
    """Send a request to the HF Inference API, process the response, and handle tool calls.

    Yields ``LLMResponseDelta`` for each streaming text chunk and a single
    trailing ``LLMResponseFinal``, transparently running a multi-round
    tool-execution loop when the model returns tool calls.
    """
    request_kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model_id,
        "stream": stream,
    }
    if tools:
        request_kwargs["tools"] = tools

    request_start = time.perf_counter()

    try:
        response = await client.chat.completions.create(**request_kwargs)
    except ClientResponseError as e:
        # HuggingFace client sets error message from the aiohttp error
        # as "response_error_payload".
        error_payload = str(getattr(e, "response_error_payload", None) or "")
        logger.exception(
            f'Failed to get a response from the LLM "{model_id}"; payload: {error_payload}'
        )
        llm.metrics.on_llm_error(
            provider=llm.provider_name,
            error_type=type(e).__name__,
        )
        yield LLMResponseFinal(original=None, text="")
        return
    except Exception as e:
        logger.exception(f'Failed to get a response from the LLM "{model_id}"')
        llm.metrics.on_llm_error(
            provider=llm.provider_name,
            error_type=type(e).__name__,
        )
        yield LLMResponseFinal(original=None, text="")
        return

    if stream:
        async for item in _process_streaming(
            llm,
            client,
            model_id,
            plugin_name,
            response,
            messages,
            tools,
            kwargs,
            request_start,
        ):
            yield item
        return

    async for item in _process_non_streaming(
        llm,
        client,
        model_id,
        plugin_name,
        response,
        messages,
        tools,
        kwargs,
        request_start,
    ):
        yield item


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
) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
    text_chunks: list[str] = []
    pending: dict[int, dict[str, Any]] = {}
    accumulated_tool_calls: list[NormalizedToolCallItem] = []
    i = 0
    chunk_id = ""
    first_token_time: Optional[float] = None
    last_chunk: Any = None

    async for chunk in response:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        content = choice.delta.content if choice.delta else None
        finish_reason = choice.finish_reason
        chunk_id = chunk.id if chunk.id else chunk_id
        last_chunk = chunk

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
            yield LLMResponseDelta(
                content_index=None,
                item_id=chunk_id,
                output_index=0,
                sequence_number=i,
                delta=content,
                is_first_chunk=is_first,
                time_to_first_token_ms=ttft_ms,
            )

        if finish_reason:
            if finish_reason in ("length", "content"):
                logger.warning(
                    f'The model finished the response due to reason "{finish_reason}"'
                )

            if finish_reason == "tool_calls":
                accumulated_tool_calls = finalize_pending_tool_calls(pending)

        i += 1

    if accumulated_tool_calls:
        async for item in _run_tool_loop(
            llm,
            client,
            model_id,
            plugin_name,
            accumulated_tool_calls,
            messages,
            tools,
            kwargs,
        ):
            yield item
        return

    total_text = "".join(text_chunks)
    latency_ms = (time.perf_counter() - request_start) * 1000
    ttft_ms_final = (
        (first_token_time - request_start) * 1000 if first_token_time else None
    )
    yield LLMResponseFinal(
        original=last_chunk,
        text=total_text,
        item_id=chunk_id,
        latency_ms=latency_ms,
        time_to_first_token_ms=ttft_ms_final,
        model=model_id,
    )


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
) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
    latency_ms = (time.perf_counter() - request_start) * 1000
    text = response.choices[0].message.content or ""

    tool_calls = extract_tool_calls_from_hf_response(response)
    if tool_calls:
        async for item in _run_tool_loop(
            llm,
            client,
            model_id,
            plugin_name,
            tool_calls,
            messages,
            tools,
            kwargs,
        ):
            yield item
        return

    if text:
        yield LLMResponseDelta(
            content_index=None,
            item_id=response.id,
            output_index=0,
            sequence_number=0,
            delta=text,
            is_first_chunk=True,
            time_to_first_token_ms=None,
        )
    yield LLMResponseFinal(
        original=response,
        text=text,
        item_id=response.id,
        latency_ms=latency_ms,
        model=model_id,
    )


async def _run_tool_loop(
    llm: LLM,
    client: AsyncInferenceClient,
    model_id: str,
    plugin_name: str,
    initial_tool_calls: list[NormalizedToolCallItem],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
    *,
    max_rounds: int = 3,
) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
    """Execute tool calls and stream follow-up responses (up to ``max_rounds``)."""
    current_messages = list(messages)
    current_tool_calls = initial_tool_calls
    seen: set[tuple[str | None, str, str]] = set()

    for round_num in range(max_rounds):
        logger.info(
            "Tool call round %d: executing %d call(s) — %s",
            round_num + 1,
            len(current_tool_calls),
            ", ".join(tc.get("name", "?") for tc in current_tool_calls),
        )

        triples, seen = await llm._dedup_and_execute(
            current_tool_calls, max_concurrency=8, timeout_s=30, seen=seen
        )
        if not triples:
            yield LLMResponseFinal(original=None, text="")
            return

        assistant_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for call_index, (tc, res, err) in enumerate(triples):
            cid = tc.get("id") or f"tool_call_{round_num}_{call_index}"
            name = tc["name"]
            args = tc.get("arguments_json", {})
            if err is not None:
                logger.warning("  [tool] %s(%s) failed: %s", name, args, err)
            else:
                logger.info("  [tool] %s(%s) → %s", name, args, res)
            assistant_tool_calls.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            )
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": cid,
                    "content": llm._sanitize_tool_output(
                        err if err is not None else res
                    ),
                }
            )

        current_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            }
        )
        current_messages.extend(tool_results)

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
            llm.metrics.on_llm_error(
                provider=llm.provider_name,
                error_type=type(e).__name__,
            )
            yield LLMResponseFinal(original=None, text="")
            return

        followup_start = time.perf_counter()
        text_chunks: list[str] = []
        pending: dict[int, dict[str, Any]] = {}
        next_tool_calls: list[NormalizedToolCallItem] = []
        i = 0
        chunk_id = ""
        first_token_time: Optional[float] = None
        last_chunk: Any = None

        async for chunk in follow_up:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content if choice.delta else None
            finish_reason = choice.finish_reason
            chunk_id = chunk.id if chunk.id else chunk_id
            last_chunk = chunk

            if choice.delta and choice.delta.tool_calls:
                for tc_chunk in choice.delta.tool_calls:
                    accumulate_tool_call_chunk(pending, tc_chunk)

            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                is_first = len(text_chunks) == 0
                ttft_ms = None
                if is_first:
                    ttft_ms = (first_token_time - followup_start) * 1000

                text_chunks.append(content)
                yield LLMResponseDelta(
                    content_index=None,
                    item_id=chunk_id,
                    output_index=0,
                    sequence_number=i,
                    delta=content,
                    is_first_chunk=is_first,
                    time_to_first_token_ms=ttft_ms,
                )

            if finish_reason == "tool_calls":
                next_tool_calls = finalize_pending_tool_calls(pending)

            i += 1

        if next_tool_calls and round_num < max_rounds - 1:
            current_tool_calls = next_tool_calls
            continue

        total_text = "".join(text_chunks)
        latency_ms = (time.perf_counter() - followup_start) * 1000
        ttft_ms_final = (
            (first_token_time - followup_start) * 1000 if first_token_time else None
        )
        yield LLMResponseFinal(
            original=last_chunk,
            text=total_text,
            item_id=chunk_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms_final,
            model=model_id,
        )
        return

    yield LLMResponseFinal(original=None, text="")
