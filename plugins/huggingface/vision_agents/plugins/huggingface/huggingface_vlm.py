import base64
import logging
from collections import deque
from typing import Iterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from huggingface_hub import AsyncInferenceClient
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
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
                text_chunks.append(content)
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        content_index=None,
                        item_id=chunk_id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                    )
                )

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )
                total_text = "".join(text_chunks)
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk_id,
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
