import base64
import logging
import os
import uuid
from collections import deque
from typing import Iterator, Optional, cast

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from getstream.video.rtc import PcmData
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

from . import events

logger = logging.getLogger(__name__)

PLUGIN_NAME = "qwen_omni"


class QwenOmni(VideoLLM):
    """
    Qwen Omni LLM plugin with native audio output using Chat Completions API.

    This plugin uses STT for speech input and provides native audio output (no TTS needed).
    Inherits from VideoLLM to allow STT transcripts while still supporting video input.

    Features:
        - Native audio output: No TTS service needed, audio comes directly from model
        - STT: Required for speech input
        - Video understanding: Set `include_video=True` for video frames
        - Streaming responses: Real-time text and audio streaming

    Examples:

        from vision_agents.plugins import qwen, deepgram
        llm = qwen.QwenOmni(model="qwen3-omni-flash", voice="Cherry")
        agent = Agent(llm=llm, stt=deepgram.STT())  # STT for input, Qwen for audio output

    """

    def __init__(
        self,
        model: str = "qwen3-omni-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "Cherry",
        audio_format: str = "wav",
        fps: int = 1,
        include_video: bool = False,
        client: Optional[AsyncOpenAI] = None,
    ):
        """
        Initialize the QwenOmni class.

        Args:
            model: The Qwen Omni model identifier (default: "qwen3-omni-flash").
            api_key: Optional API key. By default, loads from DASHSCOPE_API_KEY or ALIBABA_API_KEY.
            base_url: Optional base API url. By default, uses DashScope compatible endpoint.
            voice: Voice for audio output (default: "Cherry").
            audio_format: Audio format for output (default: "wav").
            fps: The number of video frames per second to handle (default: 1).
            include_video: Whether to include video frames in API requests (default: False).
            client: Optional `AsyncOpenAI` client. By default, creates a new client object.
        """
        super().__init__()
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.provider_name = PLUGIN_NAME
        self.session_id = str(uuid.uuid4())
        self.events.register_events_from_module(events)

        if client is not None:
            self._client = client
        else:
            default_base_url = (
                base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            resolved_api_key = api_key or os.getenv("ALIBABA_API_KEY")
            self._client = AsyncOpenAI(
                api_key=resolved_api_key, base_url=default_base_url
            )

        self._fps = fps
        self._video_forwarder: Optional[VideoForwarder] = None

        self._frame_buffer: deque[av.VideoFrame] = deque(maxlen=fps * 10)
        self._frame_width = 800
        self._frame_height = 600
        self._include_video = include_video

    @property
    def provides_audio_output(self) -> bool:
        """Indicates that this LLM provides native audio output via RealtimeAudioOutputEvent."""
        return True

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        Create an LLM response with optional audio and video input.

        This method is called every time a new STT transcript is received or
        when a direct text prompt is sent.

        Args:
            text: The text to respond to.
            processors: List of processors (which contain state) about the video/voice AI.
            participant: The Participant object, optional. If not provided, the message
                will be sent with the "user" role.

        Examples:

            llm.simple_response("say hi to the user, be nice")
        """
        if self._conversation is None:
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

        try:
            response = await self._client.chat.completions.create(  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                model=self.model,
                modalities=["text", "audio"],
                audio={"voice": self.voice, "format": self.audio_format},
                stream=True,
                stream_options={"include_usage": True},
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
        llm_response: LLMResponseEvent[Optional[ChatCompletionChunk]] = (
            LLMResponseEvent(original=None, text="")
        )
        text_chunks: list[str] = []
        total_text = ""
        audio_base64_string = ""

        async for chunk in cast(AsyncStream[ChatCompletionChunk], response):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            content = choice.delta.content
            finish_reason = choice.finish_reason

            # Process text content
            if content:
                text_chunks.append(content)
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

            # Process audio content
            # Audio may be in delta as a dict-like object
            # This is taken from the official Alibaba API documentation
            audio_data = ""
            if hasattr(choice.delta, "audio") and choice.delta.audio:
                logger.debug(f"Found audio in delta: {type(choice.delta.audio)}")
                if isinstance(choice.delta.audio, dict):
                    audio_data = choice.delta.audio.get("data", "")
                elif hasattr(choice.delta.audio, "data"):
                    audio_data = choice.delta.audio.data
            elif hasattr(choice.delta, "model_dump"):
                delta_dict = choice.delta.model_dump()
                if "audio" in delta_dict and delta_dict["audio"]:
                    logger.debug(
                        f"Found audio in model_dump: {type(delta_dict['audio'])}"
                    )
                    if isinstance(delta_dict["audio"], dict):
                        audio_data = delta_dict["audio"].get("data", "")
                    elif hasattr(delta_dict["audio"], "data"):
                        audio_data = delta_dict["audio"].data

            if audio_data:
                audio_base64_string += audio_data
                logger.debug(f"Accumulated {len(audio_data)} bytes of audio data")

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
                        item_id=chunk.id,
                    )
                )

                if audio_base64_string:
                    logger.info(
                        f"Processing {len(audio_base64_string)} bytes of accumulated audio"
                    )
                    await self._process_audio_output(audio_base64_string)
                else:
                    logger.warning("No audio data received from Qwen Omni response")

            llm_response = LLMResponseEvent(original=chunk, text=total_text)
            i += 1

        return llm_response

    async def _process_audio_output(self, base64_string: str):
        """Process base64-encoded audio output from API response.

        Args:
            base64_string: Base64-encoded WAV audio data.
        """
        try:
            # Decode base64 to bytes
            wav_bytes = base64.b64decode(base64_string)
            logger.debug(f"Decoded {len(wav_bytes)} bytes of WAV audio")

            # Convert to numpy array (int16, 24000 Hz based on reference code)
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
            logger.debug(f"Converted to numpy array with {len(audio_np)} samples")

            # Create PcmData from numpy array
            # Qwen Omni outputs at 24000 Hz, mono, int16
            pcm = PcmData.from_numpy(
                audio_np, sample_rate=24000, format="s16", channels=1
            )

            # Emit audio output event - Agent subscribes to this and forwards to audio track
            logger.info(
                f"🔊 Emitting audio output event with {len(audio_np)} samples at 24kHz"
            )
            self.events.send(
                RealtimeAudioOutputEvent(
                    session_id=self.session_id,
                    plugin_name=PLUGIN_NAME,
                    data=pcm,
                )
            )

        except Exception:
            logger.exception("Failed to process audio output from Qwen Omni")

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Setup video forwarding and start buffering video frames.

        This method is called by the `Agent`.

        Args:
            track: Instance of VideoStreamTrack.
            shared_forwarder: A shared VideoForwarder instance if present. Defaults to None.

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

    def _get_frames_bytes(self) -> Iterator[bytes]:
        """
        Iterate over all buffered video frames.

        Yields:
            JPEG-encoded frame bytes.
        """
        for frame in self._frame_buffer:
            yield frame_to_jpeg_bytes(
                frame=frame,
                target_width=self._frame_width,
                target_height=self._frame_height,
                quality=85,
            )

    async def _build_model_request(self) -> list[dict]:
        """Build the messages list for the Chat Completions API request.

        Supports text input (from STT) and optional video frames (as base64 images).
        Audio output is decoded from streaming responses.

        Returns:
            List of message dictionaries for the API request.
        """
        messages: list[dict] = []

        # Add Agent's instructions as system prompt
        if self._instructions:
            messages.append(
                {
                    "role": "system",
                    "content": self._instructions,
                }
            )

        # Add all messages from the conversation
        if self._conversation is not None:
            for message in self._conversation.messages:
                messages.append(
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                )

        # Attach video frames if enabled and available
        # Qwen supports base64 images: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        if self._include_video and self._frame_buffer:
            frames_content: list[dict] = []
            for frame_bytes in self._get_frames_bytes():
                if frame_bytes and len(frame_bytes) > 100:
                    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
                    frames_content.append(
                        {
                            "type": "image_url",  # TODO(Nash): Is there another way to send images?
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                        }
                    )

            if frames_content:
                logger.debug(f"Including {len(frames_content)} video frames in request")
                messages.append(
                    {
                        "role": "user",
                        "content": frames_content,
                    }
                )

        return messages

    async def close(self):
        """Close the connection and clean up resources."""
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
