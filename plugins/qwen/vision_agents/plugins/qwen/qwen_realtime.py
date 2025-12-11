import asyncio
import base64
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import aiortc
import av
from aiortc import VideoStreamTrack
from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import Realtime
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_utils import frame_to_jpeg_bytes

from . import events
from .client import Qwen3RealtimeClient

DEFAULT_BASE_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
PLUGIN_NAME = "Qwen3Realtime"

logger = logging.getLogger(__name__)


class Qwen3Realtime(Realtime):
    def __init__(
        self,
        model: str = "qwen3-omni-flash-realtime",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "Cherry",
        fps: int = 1,
        include_video: bool = False,
        video_width: int = 1280,
        video_height: int = 720,
        audio_transcription_model: str = "gummy-realtime-v1",
        vad_threshold: float = 0.1,
        vad_prefix_padding_ms: int = 500,
        vad_silence_duration_ms: int = 900,
    ):
        super().__init__(fps=fps)
        self.model = model
        self.voice = voice
        self.session_id = str(uuid.uuid4())
        self.events.register_events_from_module(events)

        self._base_url = base_url or DEFAULT_BASE_URL

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = cast(str, api_key)

        self._video_forwarder: Optional[VideoForwarder] = None
        self._include_video = include_video
        self._real_client: Optional[Qwen3RealtimeClient] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._video_width = video_width
        self._video_height = video_height
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._is_responding = False
        self._current_response_id = None
        self._current_item_id = None
        self._current_participant: Optional[Participant] = None
        # The model requires us not to send any video frames until the audio is sent
        self._audio_emitted_once = False
        self._audio_transcription_model = audio_transcription_model
        self._vad_threshold = vad_threshold
        self._vad_prefix_padding_ms = vad_prefix_padding_ms
        self._vad_silence_duration_ms = vad_silence_duration_ms

    async def connect(self):
        # Stop the processing task first in case we're reconnecting
        await self._stop_processing_task()

        # Session configuration
        session_config = {
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "instructions": self._instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm24",
            "input_audio_transcription": {"model": self._audio_transcription_model},
            "turn_detection": {
                "type": "server_vad",
                "threshold": self._vad_threshold,
                "prefix_padding_ms": self._vad_prefix_padding_ms,
                "silence_duration_ms": self._vad_silence_duration_ms,
            },
        }
        self._real_client = Qwen3RealtimeClient(
            api_key=self._api_key,
            base_url=self._base_url,
            model=self.model,
            config=session_config,
        )
        await self._real_client.connect()
        self.connected = True
        logger.debug(f"Started Qwen3Realtime session at {self._base_url}")

        # Start the loop task
        self._start_processing_task()

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        if not self.connected:
            return
        self._current_participant = participant
        await self._client.send_audio(pcm=pcm)
        self._audio_emitted_once = True

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        logger.warning(
            f'Cannot reply to "{text}"; reason - Qwen3Realtime does not support text inputs'
        )
        return LLMResponseEvent(text="", original=None)

    async def close(self):
        self.connected = False
        await self._stop_watching_video_track()
        if self._processing_task is not None:
            self._processing_task.cancel()
            await self._processing_task

        self._executor.shutdown(wait=False)

        if self._real_client is not None:
            await self._real_client.close()
            self._real_client = None

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Start sending video frames using VideoForwarder.

        Args:
            track: Video track to watch
            shared_forwarder: Optional shared VideoForwarder to use instead of creating a new one
        """

        # This method can be called multiple times with different forwarders
        # Remove handler from old forwarder if it exists
        await self._stop_watching_video_track()
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_video_frame)

        self._video_forwarder = shared_forwarder or VideoForwarder(
            input_track=cast(VideoStreamTrack, track),
            max_buffer=5,
            fps=float(self.fps),
            name="qwen3realtime_forwarder",
        )

        # Add frame handler (starts automatically)
        self._video_forwarder.add_frame_handler(self._send_video_frame, fps=self.fps)
        logger.info(f"Started video forwarding with {self.fps} FPS")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Send a video frame to Qwen3 Realtime API using send_realtime_input

        Parameters:
            frame: Video frame to send.
        """
        if not self._audio_emitted_once:
            # Wait until the audio is sent at least once before forwarding frames
            # per the model spec.
            return

        loop = asyncio.get_running_loop()

        # Run frame conversion in a separate thread to avoid blocking the loop.
        jpg_bytes = await loop.run_in_executor(
            self._executor,
            frame_to_jpeg_bytes,
            frame,
            self._video_width,
            self._video_height,
        )

        try:
            await self._client.send_frame(jpg_bytes)
        except Exception:
            logger.exception("Failed to send a video frame to Qwen3 Realtime API")

    async def _stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._send_video_frame)

    @property
    def _client(self) -> Qwen3RealtimeClient:
        if self._real_client is None:
            raise ValueError("The Qwen3Realtime session is not established yet")
        return self._real_client

    async def _processing_loop(self) -> None:
        logger.debug("Start processing events by Qwen3Realtime")
        try:
            await self._process_events()
        except asyncio.CancelledError:
            logger.debug("Stop processing events by Qwen3Realtime")

    def _start_processing_task(self) -> None:
        self._processing_task = asyncio.create_task(self._processing_loop())

    async def _stop_processing_task(self) -> None:
        if self._processing_task is not None:
            self._processing_task.cancel()
            await self._processing_task

    async def _process_events(self):
        async for event in self._client.read():
            event_type = event.get("type")
            if event_type == "error":
                error = event["error"]
                logger.error(
                    f"Error received from Qwen3Realtime API: {error}",
                )
                self.events.send(
                    events.LLMErrorEvent(plugin_name=PLUGIN_NAME, error_message=error)
                )
                continue

            elif event_type == "session.created":
                logger.debug("Qwen3Realtime session initialized successfully")

            elif event_type == "response.created":
                self._current_response_id = event.get("response", {}).get("id")
                self._is_responding = True
            elif event_type == "response.output_item.added":
                self._current_item_id = event.get("item", {}).get("id")
            elif event_type == "response.done":
                self._is_responding = False
                self._current_response_id = None
                self._current_item_id = None
            elif event_type == "input_audio_buffer.speech_started":
                if self._is_responding:
                    await self._on_interruption()
            elif event_type == "response.text.delta":
                self.events.send(
                    LLMResponseChunkEvent(
                        plugin_name=PLUGIN_NAME, delta=str(event["delta"])
                    )
                )
            elif event_type == "response.audio.delta":
                audio_bytes = base64.b64decode(event["delta"])
                pcm = PcmData.from_bytes(audio_bytes, 24000)
                self._emit_audio_output_event(audio_data=pcm)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                if transcript:
                    self._emit_user_speech_transcription(text=transcript)
            elif event_type == "response.audio_transcript.delta":
                delta = event.get("delta", "")
                if delta:
                    self._emit_agent_speech_transcription(text=delta)

    async def _on_interruption(self):
        """Handle user interruption of the current response."""
        if not self._is_responding:
            return

        if self._current_response_id:
            await self._client.cancel_response()

        self._is_responding = False
        self._current_response_id = None
        self._current_item_id = None
