import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.client import AsyncClient, Client
from google.genai.errors import APIError
from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponse,
    MediaResolution,
    ThinkingConfig,
    ThinkingLevel,
)
from vision_agents.core.edge.types import Participant
from vision_agents.core.instructions import Instructions
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal, VideoLLM
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

logger = logging.getLogger(__name__)

PLUGIN_NAME = "gemini_vlm"
DEFAULT_MODEL = "gemini-3-flash-preview"


class GeminiVLM(VideoLLM):
    """
    Gemini VLM integration for vision language models.

    This plugin sends buffered video frames along with text prompts to Gemini models
    that support multimodal inputs.

    Examples:
        from vision_agents.plugins import gemini

        vlm = gemini.VLM(model="gemini-3-flash-preview")
        response = await vlm.simple_response("Describe what you see.")
    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: AsyncClient | None = None,
        thinking_level: ThinkingLevel | None = None,
        media_resolution: MediaResolution | None = None,
        config: GenerateContentConfig | None = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        frame_width: int = 800,
        frame_height: int = 600,
        max_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize the GeminiVLM class.

        Args:
            model: Gemini model ID to use.
            api_key: Optional API key. Defaults to GOOGLE_API_KEY or GEMINI_API_KEY.
            client: Optional AsyncClient instance.
            thinking_level: Optional thinking level for Gemini 3.
            media_resolution: Optional media resolution for multimodal processing.
            config: Optional GenerateContentConfig to use as base.
            fps: Number of video frames per second to handle.
            frame_buffer_seconds: Number of seconds to buffer for model input.
            frame_width: Width of video frames sent to the model.
            frame_height: Height of video frames sent to the model.
            max_workers: Max worker threads for frame conversion.
            **kwargs: Additional args for GenerateContentConfig if config is not provided.
        """
        super().__init__()
        self.model = model
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution

        if config is not None:
            self._base_config: Optional[GenerateContentConfig] = config
        elif kwargs:
            self._base_config = GenerateContentConfig(**kwargs)
        else:
            self._base_config = None

        self.chat: AsyncChat | None = None
        self._config = self._build_config()

        if client is not None:
            self.client = client
        else:
            self.client = Client(api_key=api_key).aio

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _build_config(self) -> GenerateContentConfig:
        """Build GenerateContentConfig from current instance settings."""
        config = (
            self._base_config
            if self._base_config is not None
            else GenerateContentConfig()
        )

        if self._instructions:
            config.system_instruction = self._instructions

        if self.thinking_level:
            config.thinking_config = ThinkingConfig(thinking_level=self.thinking_level)

        if self.media_resolution:
            config.media_resolution = self.media_resolution

        return config

    def set_instructions(self, instructions: Instructions | str) -> None:
        super().set_instructions(instructions)
        self._config = self._build_config()

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        Create a response from text input with video context.

        Args:
            text: The prompt to respond to.
            participant: Optional participant object for message attribution.
        """
        if participant is None and self._conversation is not None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        if self.chat is None:
            self.chat = self.client.chats.create(model=self.model, config=self._config)

        frames_count = len(self._frame_buffer)

        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        text_parts: list[str] = []
        final_chunk: Optional[GenerateContentResponse] = None
        item_id = str(uuid.uuid4())
        sequence_number = 0

        try:
            parts = await self._build_message_parts(text)
            iterator = await self.chat.send_message_stream(
                message=parts, config=self._config
            )

            async for chunk in iterator:
                final_chunk = chunk
                chunk_text = self._extract_text_from_chunk(chunk)
                if chunk_text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    is_first_chunk = len(text_parts) == 0
                    delta_ttft_ms = (
                        (first_token_time - request_start_time) * 1000
                        if is_first_chunk
                        else None
                    )
                    text_parts.append(chunk_text)
                    yield LLMResponseDelta(
                        content_index=sequence_number,
                        item_id=item_id,
                        delta=chunk_text,
                        sequence_number=sequence_number,
                        is_first_chunk=is_first_chunk,
                        time_to_first_token_ms=delta_ttft_ms,
                    )
                    sequence_number += 1

            total_text = "".join(text_parts)
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            ttft_ms: Optional[float] = None
            if first_token_time is not None:
                ttft_ms = (first_token_time - request_start_time) * 1000

            input_tokens, output_tokens = self._extract_usage_tokens(final_chunk)

            self.metrics.on_vlm_inference(
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                frames_processed=frames_count,
            )

            yield LLMResponseFinal(
                original=final_chunk,
                text=total_text,
                item_id=item_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=(input_tokens or 0) + (output_tokens or 0)
                if input_tokens or output_tokens
                else None,
                model=self.model,
            )
        except APIError:
            logger.exception(f'Failed to get a response from the model "{self.model}"')
            raise

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start buffering video frames.

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: Shared VideoForwarder instance if present.
        """
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

        logger.info(f'🎥Subscribing plugin "{PLUGIN_NAME}" to VideoForwarder')
        if shared_forwarder:
            self._video_forwarder = shared_forwarder
        else:
            self._video_forwarder = VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=self._fps,
                name=f"{PLUGIN_NAME}_forwarder",
            )
            self._video_forwarder.start()

        self._video_forwarder.add_frame_handler(
            self._frame_buffer.append, fps=self._fps
        )

    async def stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._frame_buffer.append)
            self._video_forwarder = None
            logger.info(
                f"🛑 Stopped video forwarding to {PLUGIN_NAME} (participant left)"
            )

    async def _get_frames_bytes(self) -> list[bytes]:
        """Convert buffered video frames to JPEG bytes."""
        frames = list(self._frame_buffer)
        if not frames:
            return []

        loop = asyncio.get_running_loop()
        coroutines = [
            loop.run_in_executor(
                self._executor,
                frame_to_jpeg_bytes,
                frame,
                self._frame_width,
                self._frame_height,
                85,
            )
            for frame in frames
        ]
        return await asyncio.gather(*coroutines)

    async def _build_message_parts(self, text: str) -> list[Any]:
        """Build message parts with text and image frames."""
        parts: list[Any] = []
        if text:
            parts.append(text)

        frames = await self._get_frames_bytes()
        for frame_bytes in frames:
            parts.append(
                types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
            )

        if not parts:
            parts.append("")

        return parts

    @staticmethod
    def _extract_text_from_chunk(chunk: GenerateContentResponse) -> str:
        """Extract text from response chunk without SDK warnings."""
        texts = []
        if chunk.candidates:
            for candidate in chunk.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            texts.append(part.text)
        return "".join(texts)

    @staticmethod
    def _extract_usage_tokens(
        response: Optional[GenerateContentResponse],
    ) -> tuple[Optional[int], Optional[int]]:
        """Extract token usage from response if available."""
        if response is None:
            return None, None

        try:
            usage = response.usage_metadata
        except AttributeError:
            return None, None

        if not usage:
            return None, None

        input_tokens: Optional[int]
        output_tokens: Optional[int]
        try:
            input_tokens = usage.prompt_token_count
        except AttributeError:
            input_tokens = None
        try:
            output_tokens = usage.candidates_token_count
        except AttributeError:
            output_tokens = None

        return input_tokens, output_tokens

    def add_frame(self, frame: av.VideoFrame) -> None:
        """Append a video frame to the internal buffer.

        Args:
            frame: The video frame to add.
        """
        self._frame_buffer.append(frame)

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_watching_video_track()
        self._executor.shutdown(wait=False)
