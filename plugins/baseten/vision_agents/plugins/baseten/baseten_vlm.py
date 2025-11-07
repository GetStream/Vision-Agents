import base64
import io
import logging
import os
from collections import deque
from typing import Iterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from PIL.Image import Resampling
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder

from . import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "baseten_vlm"


class BasetenVLM(VideoLLM):
    """
    TODO: Docs
    TODO: Tool calling support?

    Examples:

        from vision_agents.plugins import baseten
        llm = baseten.VLM(model="qwen3vl")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the BasetenVLM class.

        Args:
            model (str): The Baseten-hosted model to use.
            api_key: optional API key. By default, loads from BASETEN_API_KEY environment variable.
            base_url: optional base url. By default, loads from BASETEN_BASE_URL environment variable.
            fps: the number of video frames per second to handle.
            frame_buffer_seconds: the number of seconds to buffer for the model's input.
                Total buffer size = fps * frame_buffer_seconds.
            client: optional `AsyncOpenAI` client. By default, creates a new client object.
        """
        super().__init__()
        self.model = model
        self.events.register_events_from_module(events)

        api_key = api_key or os.getenv("BASETEN_API_KEY")
        base_url = base_url or os.getenv("BASETEN_BASE_URL")
        if client is not None:
            self._client = client
        elif not api_key:
            raise ValueError("api_key must be provided")
        elif not base_url:
            raise ValueError("base_url must be provided")
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None

        # Buffer latest 10s of the video track to forward it to the model
        # together with the user transcripts
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
        simple_response is a standardized way to create an LLM response.

        This method is also called every time the new STT transcript is received.

        Args:
            text: The text to respond to.
            processors: list of processors (which contain state) about the video/voice AI.
            participant: the Participant object, optional.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """

        # TODO: Clean up the `_build_enhanced_instructions` and use that. The should be compiled at the agent probably.

        if self._conversation is None:
            # The agent hasn't joined the call yet.
            logger.warning(
                "Cannot create an LLM response - the conversation has not been initialized yet."
            )
            return LLMResponseEvent(original=None, text="")

        messages: list[dict] = []
        # Add Agent's instructions as system prompt.
        if self.instructions:
            messages.append(
                {
                    "role": "system",
                    "content": self.instructions,
                }
            )

        # TODO: Do we need to limit how many messages we send?
        # Add all messages from the conversation to the prompt
        for message in self._conversation.messages:
            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                }
            )

        # Attach the latest bufferred frames to the request
        frames_data = []
        for frame_bytes in self._get_frames_bytes():
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frame_msg = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            }
            frames_data.append(frame_msg)

        logger.debug(
            f'Forwarding {len(frames_data)} to the Baseten model "{self.model}"'
        )

        messages.append(
            {
                "role": "user",
                "content": frames_data,
            }
        )

        # TODO: Maybe move it to a method, too much code
        try:
            response = await self._client.chat.completions.create(  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                model=self.model,
                stream=True,
            )
        except Exception as e:
            # Send an error event if the request failed
            logger.exception(
                f'Failed to get a response from the Baseten model "{self.model}"'
            )
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        i = 0
        llm_response_event: LLMResponseEvent[Optional[ChatCompletionChunk]] = (
            LLMResponseEvent(original=None, text="")
        )
        text_chunks: list[str] = []
        total_text = ""
        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            if content:
                text_chunks.append(content)
                # Emit delta events for each response chunk.
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME,
                        content_index=None,
                        item_id=chunk.id,
                        output_index=0,
                        sequence_number=i,
                        delta=content,
                    )
                )

            elif finish_reason:
                # Emit the completion event when the response stream is finished.
                total_text = "".join(text_chunks)
                self.events.send(
                    LLMResponseCompletedEvent(
                        plugin_name=PLUGIN_NAME,
                        original=chunk,
                        text=total_text,
                        item_id=chunk.id,
                    )
                )

            llm_response_event = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        return llm_response_event

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start bufferring video frames.
        This method is called by the `Agent`.

        Args:
            track: instance of VideoStreamTrack.
            shared_forwarder: a shared VideoForwarder instance if present. Defaults to None.

        Returns: None
        """

        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

        logger.info("🎥 BasetenVLM subscribing to VideoForwarder")
        if not shared_forwarder:
            self._video_forwarder = shared_forwarder or VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=1.0,  # Low FPS for VLM
                name="baseten_vlm_forwarder",
            )
            await self._video_forwarder.start()
        else:
            self._video_forwarder = shared_forwarder

        # Start buffering video frames
        await self._video_forwarder.start_event_consumer(self._frame_buffer.append)

    def _get_frames_bytes(self) -> Iterator[bytes]:
        """
        Iterate over all bufferred video frames.
        """
        for frame in self._frame_buffer:
            yield _frame_to_jpeg_bytes(
                frame=frame,
                target_width=self._frame_width,
                target_height=self._frame_height,
                quality=85,
            )


# TODO: Move it to some core utils
def _frame_to_jpeg_bytes(
    frame: av.VideoFrame, target_width: int, target_height: int, quality: int = 85
) -> bytes:
    """
    Convert a frame to JPEG bytes with resizing.

    Args:
        frame: an instance of `av.VideoFrame`
        target_width: target width in pixels
        target_height: target height in pixels
        quality: JPEG quality. Default is 85.

    Returns: frame as JPEG bytes.

    """
    # Convert frame to a PIL image
    img = frame.to_image()

    # Calculate scaling to maintain aspect ratio
    src_width, src_height = img.size
    # Calculate scale factor (fit within target dimensions)
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)

    # Resize with aspect ratio maintained
    resized = img.resize((new_width, new_height), Resampling.LANCZOS)

    # Save as JPEG with quality control
    buf = io.BytesIO()
    resized.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()
