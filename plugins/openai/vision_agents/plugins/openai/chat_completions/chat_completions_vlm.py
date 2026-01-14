import asyncio
import base64
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional, cast

import av
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
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

from .. import events

logger = logging.getLogger(__name__)


PLUGIN_NAME = "chat_completions_vlm"


# TODO: Update openai.LLM description to point here for legacy APIs


class ChatCompletionsVLM(VideoLLM):
    """
    This plugin allows developers to easily interact with visual models that use Chat Completions API.
    The model is expected to accept text and video and respond with text.

    Features:
        - Video understanding: Automatically buffers and forwards video frames to VLM models
        - Streaming responses: Supports streaming text responses with real-time chunk events
        - Frame buffering: Configurable frame rate and buffer duration for optimal performance
        - Event-driven: Emits LLM events (chunks, completion, errors) for integration with other components

    Examples:

        from vision_agents.plugins import openai
        llm = openai.ChatCompletionsVLM(model="qwen-3-vl-32b")

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        frame_width: int = 800,
        frame_height: int = 600,
        max_workers: int = 4,
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the ChatCompletionsVLM class.

        Args:
            model (str): The model id to use.
            api_key: optional API key. By default, loads from OPENAI_API_KEY environment variable.
            base_url: optional base API url. By default, loads from OPENAI_BASE_URL environment variable.
            fps: the number of video frames per second to handle.
            frame_buffer_seconds: the number of seconds to buffer for the model's input.
                Total buffer size = fps * frame_buffer_seconds.
            frame_width: the width of the video frame to send. Default - `800`.
            frame_height: the height of the video frame to send. Default - `600`.
            max_workers: the maximum number of worker threads to use for frame-to-image conversion.
                Default - `4`.
            client: optional `AsyncOpenAI` client. By default, creates a new client object.
        """
        super().__init__()
        self.model = model
        self.events.register_events_from_module(events)

        if client is not None:
            self._client = client
        else:
            self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None

        # Buffer latest 10s of the video track to forward it to the model
        # together with the user transcripts
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

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
            participant: the Participant object, optional. If not provided, the message will be sent with the "user" role.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """

        if self._conversation is None:
            # The agent hasn't joined the call yet.
            logger.warning(
                f'Cannot request a response from the LLM "{self.model}" - the conversation has not been initialized yet.'
            )
            return LLMResponseEvent(original=None, text="")

        # The simple_response is called directly without providing the participant -
        # assuming it's an initial prompt.
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
            response = await self._client.chat.completions.create(  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                model=self.model,
                stream=True,
            )
        except Exception as e:
            # Send an error event if the request failed
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
        llm_response: LLMResponseEvent[Optional[ChatCompletionChunk]] = (
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
                # Track time to first token
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                is_first = len(text_chunks) == 0
                ttft_ms = None
                if is_first:
                    ttft_ms = (first_token_time - request_start_time) * 1000

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
                        is_first_chunk=is_first,
                        time_to_first_token_ms=ttft_ms,
                    )
                )

            if finish_reason:
                if finish_reason in ("length", "content"):
                    logger.warning(
                        f'The model finished the response due to reason "{finish_reason}"'
                    )
                # Emit the completion event when the response stream is finished.
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
                        item_id=chunk.id,
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

        # Start buffering video frames
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

    async def _get_frames_bytes(self) -> AsyncIterator[bytes]:
        """
        Convert the buffered video frames to bytes and yield them.
        The conversion happens asynchronously in the background threads.
        """
        loop = asyncio.get_running_loop()

        # Convert frames to bytes in parallel in background threads
        coroutines = [
            loop.run_in_executor(
                self._executor,
                frame_to_jpeg_bytes,
                frame,
                self._frame_width,
                self._frame_height,
                85,
            )
            for frame in self._frame_buffer
        ]
        for frame_bytes in await asyncio.gather(*coroutines):
            yield frame_bytes

    async def _build_model_request(self) -> list[dict]:
        messages: list[dict] = []
        # Add Agent's instructions as system prompt.
        if self._instructions:
            messages.append(
                {
                    "role": "system",
                    "content": self._instructions,
                }
            )

        # Add all messages from the conversation to the prompt
        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        # Attach the latest buffered frames to the request
        frames_data = []
        async for frame_bytes in self._get_frames_bytes():
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
        self._executor.shutdown(wait=False)
