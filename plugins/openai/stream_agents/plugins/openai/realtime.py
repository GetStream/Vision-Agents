import asyncio
from typing import Any, Optional
from stream_agents.core.llm import realtime
import logging
import numpy as np
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from .rtc_manager import RTCManager
from openai.types.realtime import *

load_dotenv()

logger = logging.getLogger(__name__)


class Realtime(realtime.Realtime):
    def __init__(self, model: str = "gpt-realtime", voice: str = "marin", send_video: bool = False):
        super().__init__()
        self.model = model
        self.voice = voice
        self.send_video = send_video
        self.rtc = RTCManager(self.model, self.voice, self.send_video)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.connect())
        except RuntimeError:
            # Not in an event loop; caller will invoke connect() later
            pass

    async def connect(self):
        # Wire callbacks so we can emit audio/events upstream
        self.rtc.set_event_callback(self._handle_openai_event)
        self.rtc.set_audio_callback(self._handle_audio_output)
        if self.send_video:
            self.rtc.set_video_callback(self._handle_video_output)
        await self.rtc.connect()
        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio"],
        )

    async def send_audio_pcm(self, audio: PcmData):
        await self.rtc.send_audio_pcm(audio)

    async def send_text(self, text: str, role="user"):
        await self.rtc.send_text(text, role)

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        ...

    async def request_session_info(self) -> None:
        await self.rtc.request_session_info()

    async def _close_impl(self):
        await self.rtc.close()

    async def _handle_openai_event(self, event: dict) -> None:
        et = event.get("type")
        if et == "response.audio_transcript.done":
            event: ResponseAudioTranscriptDoneEvent = ResponseAudioTranscriptDoneEvent.model_validate(event)
            self._emit_transcript_event(text=event.transcript, user_metadata={"role": "assistant", "source": "openai"})
            self._emit_response_event(text=event.transcript, response_id=event.response_id, is_complete=True, conversation_item_id=event.item_id)
        if et == "input_audio_buffer.speech_started":
            event: InputAudioBufferSpeechStartedEvent = InputAudioBufferSpeechStartedEvent.model_validate(event)
            await self.output_track.flush()            

    async def _handle_audio_output(self, audio_bytes: bytes) -> None:
        # Forward audio as event and to output track if available
        logger.debug(f"🎵 Forwarding audio output: {len(audio_bytes)}")
        if self.output_track is not None:
            await self.output_track.write(audio_bytes)
        else:
            logger.info("Can't find output track to set bytes")

    async def _handle_video_output(self, video_array: np.ndarray) -> None:
        """Handle incoming video frames from OpenAI Realtime API.
        
        Args:
            video_array: RGB video frame as numpy array from frame.to_ndarray()
        """
        logger.debug(f"🎥 Forwarding video frame: shape={video_array.shape}, dtype={video_array.dtype}")
    
        # Write to output track for remote participants to see
        if self.output_track is not None:
            try:
                await self.output_track.write(video_array)
                logger.debug(f"✅ Video frame written to output track: {video_array.shape}")
            except Exception as e:
                logger.error(f"❌ Failed to write video frame to output track: {e}")
        else:
            logger.warning("No output_track set - video will not be visible to remote participants")

    async def start_video_sender(self, track, fps: int = 1) -> None:
        # Delegate to RTC manager to swap the negotiated sender's track
        await self.rtc.start_video_sender(track, fps)

    async def stop_video_sender(self) -> None:
        await self.rtc.stop_video_sender()
