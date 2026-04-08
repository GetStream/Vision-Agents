import base64
import logging
import time
import uuid
from collections import deque
from typing import Any, Iterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._providers import PROVIDER_OR_POLICY_T
from vision_agents.core.llm.events import (
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.llm.llm_types import ToolSchema
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

from . import events
from ._hf_tool_calling import convert_tools_to_hf_format, create_hf_response

logger = logging.getLogger(__name__)


PLUGIN_NAME = "huggingface_vlm"


class HuggingFaceVLM(VideoLLM):
    """HuggingFace Inference integration for vision language models.

    This plugin allows developers to interact with vision models via HuggingFace's
    Inference Providers API. Supports models that accept both text and images.

    Features:
        - Video understanding: Automatically buffers and forwards video frames
        - Streaming responses with real-time chunk events
        - Function/tool calling support
        - Configurable frame rate and buffer duration

    Examples:

        from vision_agents.plugins import huggingface
        vlm = huggingface.VLM(model="Qwen/Qwen2-VL-7B-Instruct")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        base_url: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        client: Optional[AsyncInferenceClient] = None,
    ):
        """Initialize the HuggingFaceVLM class.

        Args:
            model: The HuggingFace model ID to use.
            api_key: HuggingFace API token. Defaults to HF_TOKEN environment variable.
            provider: Inference provider (e.g., "hf-inference"). Auto-selects if omitted.
            base_url: Custom API base URL for OpenAI-compatible endpoints (e.g., Baseten).
                Mutually exclusive with provider.
            fps: Number of video frames per second to handle.
            frame_buffer_seconds: Number of seconds to buffer for the model's input.
            client: Optional AsyncInferenceClient instance for dependency injection.
        """
        super().__init__()
        self.model = model
        self.provider = provider
        self.events.register_events_from_module(events)

        if base_url and provider:
            raise ValueError("`base_url` and `provider` are mutually exclusive.")

        if client is not None:
            self._client = client
        elif base_url:
            self._client = AsyncInferenceClient(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self._client = AsyncInferenceClient(
                token=api_key,
                model=model,
                provider=provider,
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
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """Create an LLM response from text input with video context.

        This method is called when a new STT transcript is received.

        Args:
            text: The text to respond to.
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

        frames_count = len(self._frame_buffer)
        inference_id = str(uuid.uuid4())

        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model,
                frames_count=frames_count,
            )
        )

        request_start_time = time.perf_counter()
        response = await self.create_response(messages=messages)
        latency_ms = (time.perf_counter() - request_start_time) * 1000

        if response.exception is not None:
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=response.exception,
                    context="generation",
                )
            )
        else:
            self.events.send(
                VLMInferenceCompletedEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    model=self.model,
                    text=response.text,
                    latency_ms=latency_ms,
                    frames_processed=frames_count,
                )
            )

        return response

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        stream: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Create a response using HuggingFace's Inference API with video context.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                If not provided, builds from conversation history + buffered frames.
            stream: Whether to stream the response.
            **kwargs: Additional arguments passed to the API.

        Returns:
            LLMResponseEvent with the response.
        """
        if messages is None:
            messages = await self._build_model_request()

        tools_param = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_hf_format(tools_spec)

        return await create_hf_response(
            self,
            self._client,
            self.model,
            PLUGIN_NAME,
            messages,
            tools_param,
            stream,
            **kwargs,
        )

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        return convert_tools_to_hf_format(tools)

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Setup video forwarding and start buffering video frames.

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: A shared VideoForwarder instance if present.
        """
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

        logger.info(f'Subscribing plugin "{PLUGIN_NAME}" to VideoForwarder')
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
            logger.info(f"Stopped video forwarding to {PLUGIN_NAME}")

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

    async def close(self) -> None:
        await self._client.close()
