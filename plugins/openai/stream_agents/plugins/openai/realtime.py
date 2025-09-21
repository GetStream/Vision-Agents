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
    """
    OpenAI Realtime API implementation for real-time AI audio and video communication over WebRTC.

    Extends the base Realtime class with WebRTC-based audio and optional video
    streaming to OpenAI's servers. Supports speech-to-speech conversation, text
    messaging, and multimodal interactions.

    Args:
        model: OpenAI model to use (e.g., "gpt-realtime").
        voice: Voice for audio responses (e.g., "marin", "alloy").
        send_video: Enable video streaming capabilities. Defaults to False.

        This class uses:
        - RTCManager to handle WebRTC connection and media streaming.
        - Output track to forward audio and video to the remote participant.
    """
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
        """Establish the WebRTC connection to OpenAI's Realtime API.

        Sets up callbacks and connects to OpenAI's servers. Emits connected event
        with session configuration when ready.
        """
        # Wire callbacks so we can emit audio/events upstream
        self.rtc.set_event_callback(self._handle_openai_event)
        self.rtc.set_audio_callback(self._handle_audio_output)
        await self.rtc.connect()
        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio"],
        )

    async def send_audio_pcm(self, audio: PcmData):
        """Send raw PCM audio data to the OpenAI session.

        Args:
            audio: PCM audio data to transmit via WebRTC.
        """
        await self.rtc.send_audio_pcm(audio)

    async def send_text(self, text: str, role="user"):
        """Send a text message to the OpenAI session.

        Args:
            text: Message text to send to the AI model.
            role: Message role. Defaults to "user".
        """
        await self.rtc.send_text(text, role)

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        """Send native OpenAI Realtime API input directly to the session.

        Args:
            text: Optional text input to send.
            audio: Optional audio input to send.
            media: Optional media input to send.

        Note:
        Currently not implemented.
        """
        ...

    async def request_session_info(self) -> None:
        """Request session information from the OpenAI API.

        Delegates to the RTC manager to query session metadata.
        """
        await self.rtc.request_session_info()

    async def _close_impl(self):
        """Close the OpenAI Realtime session.

        Delegates cleanup to the RTC manager. Called by the base class close() method.
        """
        await self.rtc.close()

    async def _handle_openai_event(self, event: dict) -> None:
        """Process events received from the OpenAI Realtime API.

        Handles OpenAI event types and emits standardized events.

        Args:
            event: Raw event dictionary from OpenAI API.

        Event Handling:
            - response.audio_transcript.done: Emits transcript/response events
            - input_audio_buffer.speech_started: Flushes output audio track

        Note:
            Registered as callback with RTC manager.
        """
        et = event.get("type")
        if et == "response.audio_transcript.done":
            event: ResponseAudioTranscriptDoneEvent = ResponseAudioTranscriptDoneEvent.model_validate(event)
            self._emit_transcript_event(text=event.transcript, user_metadata={"role": "assistant", "source": "openai"})
            self._emit_response_event(text=event.transcript, response_id=event.response_id, is_complete=True, conversation_item_id=event.item_id)
        if et == "input_audio_buffer.speech_started":
            event: InputAudioBufferSpeechStartedEvent = InputAudioBufferSpeechStartedEvent.model_validate(event)
            await self.output_track.flush()

    async def _handle_audio_output(self, audio_bytes: bytes) -> None:
        """Process audio output received from the OpenAI API.

        Forwards audio data to the output track for playback.

        Args:
            audio_bytes: Raw audio data bytes from OpenAI session.

        Note:
            Registered as callback with RTC manager.
        """
        # Forward audio as event and to output track if available
        logger.debug(f"🎵 Forwarding audio output: {len(audio_bytes)}")
        if self.output_track is not None:
            await self.output_track.write(audio_bytes)
        else:
            logger.info("Can't find output track to set bytes")

    async def start_video_sender(self, track, fps: int = 1) -> None:
        """Start sending video data to the OpenAI session.

        Args:
            track: Video track to send via WebRTC.
            fps: Target frames per second. Defaults to 1.

        Note:
            Delegates to RTC manager. Requires send_video=True during initialization.
        """
        # Delegate to RTC manager to swap the negotiated sender's track
        await self.rtc.start_video_sender(track, fps)

    async def stop_video_sender(self) -> None:
        """Stop sending video data to the OpenAI session.

        Note:
            Delegates to RTC manager to stop video transmission.
        """
        await self.rtc.stop_video_sender()
