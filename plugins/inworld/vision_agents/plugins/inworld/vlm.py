"""Inworld AI VLM router via the OpenAI-compatible chat completions endpoint.

Mirrors :class:`vision_agents.plugins.inworld.LLM` but for vision-capable
upstream models. Sends buffered video frames as ``image_url`` content parts
through Inworld's ``/v1/chat/completions`` proxy. Use this when the upstream
provider you're routing to supports vision (e.g. ``openai/gpt-4o``,
``google-ai-studio/gemini-2.5-flash``, ``anthropic/claude-sonnet-4-5``).
"""

import os
from typing import Any, Optional

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsVLM

from .llm import (
    INWORLD_BASE_URL,
    PLUGIN_NAME,
    _validate_ttft_timeout,
    build_extra_body,
    inject_compression,
)


class VLM(ChatCompletionsVLM):
    """Inworld VLM router (text + video chat completions).

    Same routing kwargs as :class:`LLM`. Vision works through the standard
    OpenAI ``image_url`` content shape, which Inworld passes through to the
    selected upstream provider.

    Examples:

        from vision_agents.plugins import inworld

        vlm = inworld.VLM(
            model="openai/gpt-4o",
            sort_by=["latency"],
            fps=1,
            frame_buffer_seconds=10,
        )
    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        model: str = "auto",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        frame_width: int = 800,
        frame_height: int = 600,
        max_workers: int = 4,
        client: Optional[AsyncOpenAI] = None,
        *,
        fallback_models: Optional[list[str]] = None,
        ignore_models: Optional[list[str]] = None,
        sort_by: Optional[list[str]] = None,
        ttft_timeout: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        web_search: bool = False,
        web_search_options: Optional[dict[str, Any]] = None,
        compression_aggressiveness: Optional[float] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the Inworld VLM router client.

        See :class:`LLM` for documentation of the routing kwargs. Frame
        kwargs (``fps``, ``frame_buffer_seconds``, ``frame_width``,
        ``frame_height``, ``max_workers``) match
        :class:`vision_agents.plugins.openai.ChatCompletionsVLM`.
        """
        _validate_ttft_timeout(ttft_timeout)

        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("INWORLD_API_KEY"),
            base_url=base_url or INWORLD_BASE_URL,
            fps=fps,
            frame_buffer_seconds=frame_buffer_seconds,
            frame_width=frame_width,
            frame_height=frame_height,
            max_workers=max_workers,
            client=client,
        )
        self._compression_aggressiveness = compression_aggressiveness
        self._extra_body = build_extra_body(
            fallback_models=fallback_models,
            ignore_models=ignore_models,
            sort_by=sort_by,
            ttft_timeout=ttft_timeout,
            metadata=metadata,
            web_search=web_search,
            web_search_options=web_search_options,
            override=extra_body,
        )

    def _extra_request_kwargs(self) -> dict[str, Any]:
        return {"extra_body": self._extra_body} if self._extra_body else {}

    async def _build_model_request(self) -> list[dict]:
        messages = await super()._build_model_request()
        return inject_compression(messages, self._compression_aggressiveness)
