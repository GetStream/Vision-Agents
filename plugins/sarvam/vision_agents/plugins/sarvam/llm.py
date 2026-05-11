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
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.plugins.openai import ChatCompletionsLLM

logger = logging.getLogger(__name__)

SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
DEFAULT_MODEL = "sarvam-m"
SUPPORTED_MODELS = {"sarvam-m", "sarvam-30b", "sarvam-105b"}

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

    provider_name = "sarvam"

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
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        kwargs: dict[str, Any],
        request_start_time: float,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Strip ``<think>`` blocks from streamed chunks before delegating."""
        wrapped = self._filter_think_chunks(response)
        async for item in super()._process_streaming_response(
            wrapped, messages, tools, kwargs, request_start_time
        ):
            yield item

    async def _filter_think_chunks(
        self, response: AsyncStream[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        f = _ThinkTagFilter()
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk.choices[0].delta.content = f.feed(chunk.choices[0].delta.content)
            yield chunk
