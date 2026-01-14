import base64
import logging
import time
import uuid
from collections import deque
from typing import Iterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from huggingface_hub import AsyncInferenceClient
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    VLMInferenceStartEvent,
    VLMInferenceCompletedEvent,
    VLMErrorEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

from . import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "huggingface_vlm"


class HuggingFaceVLM(VideoLLM):
    """
    HuggingFace Inference integration for vision language models.

    This plugin allows developers to interact with vision models via HuggingFace's
    Inference Providers API. Supports models that accept both text and images.

    Features:
        - Video understanding: Automatically buffers and forwards video frames
        - Streaming responses with real-time chunk events
        - Configurable frame rate and buffer duration

    Examples:

        from vision_agents.plugins import huggingface
        vlm = huggingface.VLM(model="Qwen/Qwen2-VL-7B-Instruct")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        client: Optional[AsyncInferenceClient] = None,
    ):
        """
        Initialize the HuggingFaceVLM class.

        Args:
            model: The HuggingFace model ID to use.
            api_key: HuggingFace API token. Defaults to HF_TOKEN environment variable.
            provider: Inference provider (e.g., "hf-inference"). Auto-selects if omitted.
            fps: Number of video frames per second to handle.
            frame_buffer_seconds: Number of seconds to buffer for the model's input.
            client: Optional AsyncInferenceClient instance for dependency injection.
        """
        super().__init__()
        self.model = model
        self.provider = provider
        self.events.register_events_from_module(events)

        if client is not None:
            self._client = client
        else:
            self._client = AsyncInferenceClient(
                token=api_key,
                model=model,
            )

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self._frame_width = 800
        self._frame_height = 600

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        Create an LLM response from text input with video context.

        This method is called when a new STT transcript is received.

        Args:
            text: The text to respond to.
            processors: List of processors with video/voice AI state.
            participant: The participant object. If not provided, uses "user" role.
        """
        if self._conversation is None:
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - '
                "the conversation has not been initialized yet."
            )
            return LLMResponseEvent(original=None, text="")

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = await self._build_model_request()

        # Count frames being processed
        frames_count = len(self._frame_buffer)
        inference_id = str(uuid.uuid4())

        # Emit VLM start event
        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model,
                frames_count=frames_count,
            )
        )

        # Emit request started event
        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=self.model,
                streaming=True,
            )
        )

        # Track timing
        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None

        try:
            response = await self._client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=True,
            )
        except Exception as e:
            logger.exception(f'Failed to get a response from the model "{self.model}"')
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=e,
                    context="api_request",
                )
            )
            return LLMResponseEvent(original=None, text="")

        i = 0
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        text_chunks: list[str] = []
        total_text = ""
        chunk_id = ""

        async for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content if choice.delta else None
            finish_reason = choice.finish_reason
            chunk_id = chunk.id if chunk.id else chunk_id

            if content:
                # Track time to first token
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
                total_text = "".join(text_chunks)
                latency_ms = (time.perf_counter() - request_start_time) * 1000
                ttft_ms_final = None
                if first_token_time is not None:
                    ttft_ms_final = (first_token_time - request_start_time) * 1000

                # Emit VLM-specific completion event with metrics
                self.events.send(
                    VLMInferenceCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        inference_id=inference_id,
                        model=self.model,
                        text=total_text,
                        latency_ms=latency_ms,
                        frames_processed=frames_count,
                    )
                )

                # Also emit LLM completion for compatibility
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk_id,
                        latency_ms=latency_ms,
                        time_to_first_token_ms=ttft_ms_final,
                        model=self.model,
                    )
                )

            llm_response = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        return llm_response

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start buffering video frames.

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: A shared VideoForwarder instance if present.
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

    def _get_frames_bytes(self) -> Iterator[bytes]:
        """Iterate over all buffered video frames."""
        for frame in self._frame_buffer:
            yield frame_to_jpeg_bytes(
                frame=frame,
                target_width=self._frame_width,
                target_height=self._frame_height,
                quality=85,
            )

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        if self._instructions:
            messages.append(
                {
                    "role": "system",
                    "content": self._instructions,
                }
            )

        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        frames_data = []
        for frame_bytes in self._get_frames_bytes():
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frame_msg = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            }
            frames_data.append(frame_msg)
        if frames_data:
            logger.debug(f'Forwarding {len(frames_data)} to the LLM "{self.model}"')
            messages.append(
                {
                    "role": "user",
                    "content": frames_data,
                }
            )
        return messages
