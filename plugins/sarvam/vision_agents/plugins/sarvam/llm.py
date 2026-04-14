"""Sarvam AI LLM using the OpenAI-compatible Chat Completions endpoint.

Sarvam exposes ``/v1/chat/completions`` with the same shape as OpenAI, so we
point an ``AsyncOpenAI`` client at Sarvam's base URL and inject the
``api-subscription-key`` header. Streaming, tool calling, and conversation
history are all inherited from :class:`ChatCompletionsLLM`.

Sarvam-m supports "hybrid thinking" which emits ``<think>…</think>`` blocks
before the actual answer. This plugin strips those blocks from the streamed
output so they don't reach TTS.

Docs: https://docs.sarvam.ai/api-reference-docs/chat/chat-completions
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, cast

from openai import AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem
from vision_agents.plugins.openai import ChatCompletionsLLM

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
DEFAULT_MODEL = "sarvam-m"
SUPPORTED_MODELS = {"sarvam-m", "sarvam-30b", "sarvam-105b"}

PLUGIN_NAME = "sarvam"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class _ThinkTagFilter:
    """Streaming filter that strips ``<think>…</think>`` blocks.

    Feed each streamed delta via :meth:`feed` and use the return value
    (possibly empty) as the filtered delta to emit.
    """

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, delta: str) -> str:
        """Process *delta* and return the portion that should be emitted."""
        self._buf += delta
        out_parts: list[str] = []

        while self._buf:
            if self._inside:
                end = self._buf.find("</think>")
                if end == -1:
                    # Still inside — keep a partial ``</think>`` prefix so we
                    # can detect the closing tag when it spans multiple chunks.
                    lt = self._buf.rfind("<")
                    if lt != -1 and "</think>".startswith(self._buf[lt:]):
                        self._buf = self._buf[lt:]
                    else:
                        self._buf = ""
                    break
                # Skip past closing tag
                self._buf = self._buf[end + len("</think>") :]
                self._inside = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    # No opening tag — check for a possible partial tag at the
                    # end (e.g. "<thi") and keep it buffered.
                    lt = self._buf.rfind("<")
                    if lt != -1 and "<think>".startswith(self._buf[lt:]):
                        out_parts.append(self._buf[:lt])
                        self._buf = self._buf[lt:]
                    else:
                        out_parts.append(self._buf)
                        self._buf = ""
                    break
                # Emit text before the tag, consume the tag
                out_parts.append(self._buf[:start])
                self._buf = self._buf[start + len("<think>") :]
                self._inside = True

        return "".join(out_parts)

    def flush(self, text: str) -> str:
        """Strip think tags from the final accumulated text."""
        return _THINK_RE.sub("", text).strip()


class SarvamLLM(ChatCompletionsLLM):
    """Sarvam AI Chat Completions LLM.

    Thin wrapper around :class:`ChatCompletionsLLM` that configures the OpenAI
    client for Sarvam's OpenAI-compatible endpoint and strips ``<think>``
    blocks from streamed output so TTS doesn't speak the reasoning text.

    Examples:

        from vision_agents.plugins import sarvam
        llm = sarvam.LLM(model="sarvam-30b")
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = SARVAM_BASE_URL,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        """Initialize the Sarvam LLM.

        Args:
            model: The Sarvam model id. Defaults to ``sarvam-m``. Supported:
                ``sarvam-m``, ``sarvam-30b``, ``sarvam-105b``.
            api_key: Sarvam API key. Defaults to ``SARVAM_API_KEY`` env var.
            base_url: API base URL. Defaults to ``https://api.sarvam.ai/v1``.
            client: Optional pre-configured ``AsyncOpenAI`` client. Takes
                precedence over ``api_key`` / ``base_url``.
        """
        resolved_key = (
            api_key if api_key is not None else os.environ.get("SARVAM_API_KEY")
        )
        if client is None and not resolved_key:
            raise ValueError(
                "SARVAM_API_KEY env var or api_key parameter required for Sarvam LLM"
            )

        if client is None:
            client = AsyncOpenAI(
                api_key=resolved_key,
                base_url=base_url,
                default_headers={"api-subscription-key": resolved_key or ""},
            )

        super().__init__(model=model, client=client)

    async def _process_streaming_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
        request_start_time: float,
    ) -> LLMResponseEvent:
        """Process streaming response, stripping ``<think>`` blocks."""
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        text_chunks: list[str] = []
        total_text = ""
        self._pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        accumulated_tool_calls: List[NormalizedToolCallItem] = []
        seq = 0
        first_token_time: Optional[float] = None
        think_filter = _ThinkTagFilter()

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

                text_chunks.append(content)

                filtered = think_filter.feed(content)
                if filtered:
                    is_first = seq == 0
                    ttft_ms = None
                    if is_first and first_token_time is not None:
                        ttft_ms = (first_token_time - request_start_time) * 1000
                    self.events.send(
                        LLMResponseChunkEvent(
                            plugin_name=PLUGIN_NAME,
                            content_index=None,
                            item_id=chunk.id,
                            output_index=0,
                            sequence_number=seq,
                            delta=filtered,
                            is_first_chunk=is_first,
                            time_to_first_token_ms=ttft_ms,
                        )
                    )
                    seq += 1

            if finish_reason:
                if finish_reason == "tool_calls":
                    accumulated_tool_calls = self._finalize_pending_tool_calls()

                total_text = think_filter.flush("".join(text_chunks))
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

        if accumulated_tool_calls:
            return await self._handle_tool_calls(
                accumulated_tool_calls, messages, tools, kwargs
            )

        return llm_response
