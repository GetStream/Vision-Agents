"""Abstract base class for local vision-language model plugins.

Provides ``LocalVLM`` — the shared orchestration for video-capable local
inference backends (Transformers, MLX). Subclasses implement model loading
and the raw generate-from-frames call.
"""

import abc
import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Any, Generic, Optional, TypeVar, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseCompletedEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.warmup import Warmable

from . import events
from ._local_inference import (
    build_messages,
    extract_tool_calls_from_text,
    run_local_tool_call_loop,
)
from ._tool_call_loop import convert_tools_to_chat_completions_format

logger = logging.getLogger(__name__)

R = TypeVar("R")


class LocalVLM(VideoLLM, Warmable[R], Generic[R]):
    """Abstract base for local VLM inference (Transformers, MLX, etc.).

    Subclasses implement model loading and a single
    ``_generate_with_frames`` method. This base provides the shared
    orchestration: warmup, video track management, ``simple_response``,
    ``create_response`` (with tool-call handling), and lifecycle management.
    """

    _plugin_name: str

    def __init__(
        self,
        model: str,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        max_frames: int = 4,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 3,
    ):
        super().__init__()
        self.model_id = model
        self._fps = fps
        self._max_frames = max_frames
        self._max_new_tokens = max_new_tokens
        self._max_tool_rounds = max_tool_rounds
        self._resources: Optional[R] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self.events.register_events_from_module(events)

    @abc.abstractmethod
    def _load_model_sync(self) -> R:
        """Load and return model resources (called in a thread)."""

    @abc.abstractmethod
    def _generate_with_frames(
        self,
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        tools_param: Optional[list[dict[str, Any]]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Run VLM generation synchronously (called via ``asyncio.to_thread``)."""

    async def on_warmup(self) -> R:
        logger.info("Loading model: %s", self.model_id)
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info("Model loaded: %s", self.model_id)
        return resources

    def on_warmed_up(self, resource: R) -> None:
        self._resources = resource

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None

        logger.info('Subscribing plugin "%s" to VideoForwarder', self._plugin_name)

        if shared_forwarder:
            self._video_forwarder = shared_forwarder
        else:
            self._video_forwarder = VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=self._fps,
                name=f"{self._plugin_name}_forwarder",
            )
            self._video_forwarder.start()

        self._video_forwarder.add_frame_handler(
            self._frame_buffer.append, fps=self._fps
        )

    async def stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._frame_buffer.append)
            self._video_forwarder = None
            logger.info("Stopped video forwarding to %s", self._plugin_name)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Any] = None,
    ) -> LLMResponseEvent:
        if self._conversation is None:
            logger.warning(
                "Conversation not initialized. Call set_conversation() first."
            )
            return LLMResponseEvent(original=None, text="")

        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        frames_snapshot = list(self._frame_buffer)
        image_count = min(len(frames_snapshot), self._max_frames)

        messages = self._build_messages()
        image_content: list[dict[str, Any]] = [
            {"type": "image"} for _ in range(image_count)
        ]
        image_content.append({"type": "text", "text": text or "Describe what you see."})

        if messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": image_content}
        else:
            messages.append({"role": "user", "content": image_content})

        inference_id = str(uuid.uuid4())

        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=self._plugin_name,
                inference_id=inference_id,
                model=self.model_id,
                frames_count=len(frames_snapshot),
            )
        )

        request_start = time.perf_counter()
        response = await self.create_response(messages=messages, frames=frames_snapshot)
        latency_ms = (time.perf_counter() - request_start) * 1000

        if response.exception is not None:
            self.events.send(
                VLMErrorEvent(
                    plugin_name=self._plugin_name,
                    inference_id=inference_id,
                    error=response.exception,
                    context="generation",
                )
            )
        else:
            self.events.send(
                VLMInferenceCompletedEvent(
                    plugin_name=self._plugin_name,
                    inference_id=inference_id,
                    model=self.model_id,
                    text=response.text,
                    latency_ms=latency_ms,
                    frames_processed=len(frames_snapshot),
                )
            )

        return response

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        frames: Optional[list[av.VideoFrame]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Generate a response from messages and optional video frames."""
        is_tool_followup = kwargs.pop("_tool_followup", False)

        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if messages is None:
            messages = self._build_messages()
        if frames is None:
            frames = list(self._frame_buffer)

        tools_param: Optional[list[dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_chat_completions_format(tools_spec)

        max_tokens = max_new_tokens or self._max_new_tokens

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=self._plugin_name,
                model=self.model_id,
                streaming=False,
            )
        )

        request_start = time.perf_counter()

        try:
            result_text = await asyncio.to_thread(
                self._generate_with_frames,
                messages,
                frames,
                tools_param,
                max_tokens,
                temperature,
            )
        except (RuntimeError, ValueError) as e:
            logger.exception("VLM generation failed")
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=self._plugin_name,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="", exception=e)

        if tools_param and result_text:
            tool_calls = extract_tool_calls_from_text(result_text)
            if tool_calls:
                if is_tool_followup:
                    return LLMResponseEvent(original=None, text=result_text)
                return await self._handle_tool_calls(
                    tool_calls, messages, frames, kwargs
                )

        latency_ms = (time.perf_counter() - request_start) * 1000
        response_id = str(uuid.uuid4())

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=self._plugin_name,
                original=None,
                text=result_text,
                item_id=response_id,
                latency_ms=latency_ms,
                model=self.model_id,
            )
        )

        return LLMResponseEvent(original=None, text=result_text)

    def _build_messages(self) -> list[dict[str, Any]]:
        return build_messages(self._instructions, self._conversation)

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        return convert_tools_to_chat_completions_format(tools)

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        kwargs: dict[str, Any],
    ) -> LLMResponseEvent:
        async def _followup(msgs: list[dict[str, Any]]) -> LLMResponseEvent:
            return await self.create_response(
                messages=msgs, frames=frames, _tool_followup=True, **kwargs
            )

        return await run_local_tool_call_loop(
            self, tool_calls, messages, _followup, max_tool_rounds=self._max_tool_rounds
        )

    def unload(self) -> None:
        logger.info("Unloading model: %s", self.model_id)
        self._resources = None

    @property
    def is_loaded(self) -> bool:
        return self._resources is not None
