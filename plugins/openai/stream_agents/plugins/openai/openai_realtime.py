import asyncio
from typing import Any, Optional, List

from getstream.video.rtc.audio_track import AudioStreamTrack

from stream_agents.core.llm import realtime
import logging
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from .rtc_manager import RTCManager
from openai.types.realtime import *

from stream_agents.core.edge.types import Participant
from stream_agents.core.processors import Processor

load_dotenv()

logger = logging.getLogger(__name__)


"""
TODO
- Docs for this file
- MCP support
- instructions with @mentions
- event handling is not using either pyee or internal system in self.rtc
- The base class Realtime has a ton of junk

"""



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
    def __init__(self, model: str = "gpt-realtime", voice: str = "marin"):
        super().__init__()
        self.model = model
        self.voice = voice
        # TODO: send video should depend on if the RTC connection with stream is sending video.
        self.rtc = RTCManager(self.model, self.voice, True)
        # audio output track?
        self.output_track = AudioStreamTrack(
            framerate=48000, stereo=True, format="s16"
        )

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

    async def simple_response(self, text: str, processors: Optional[List[Processor]] = None,
      participant: Participant = None):
        """Send a simple text input to the OpenAI Realtime session.

        This is a convenience wrapper that forwards a text prompt upstream via
        the underlying realtime connection. It does not stream partial deltas
        back; callers should subscribe to the provider's events to receive
        responses.

        Args:
            text: Text prompt to send.
            processors: Optional processors list (not used here; included for
                interface parity with the core `LLM` API).
            participant: Optional participant metadata (ignored here).
        """
        await self.rtc.send_text(text)

    async def simple_audio_response(self, audio: PcmData):
        """Send a single PCM audio frame to the OpenAI Realtime session.

        The audio should be raw PCM matching the realtime session's expected
        format (typically 48 kHz mono, 16-bit). For continuous audio capture,
        call this repeatedly with consecutive frames.

        Args:
            audio: PCM audio frame to forward upstream.
        """
        await self.rtc.send_audio_pcm(audio)

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
        logger.info(f"🎵 Forwarding audio output: {len(audio_bytes)}")

        await self.output_track.write(audio_bytes)

    async def _watch_video_track(self, track, fps: int = 1) -> None:
        # TODO: only do this once?
        #self.rtc.set_video_callback(self._handle_video_output)
        # Delegate to RTC manager to swap the negotiated sender's track
        await self.rtc.start_video_sender(track, fps)

    async def _stop_watching_video_track(self) -> None:
        await self.rtc.stop_video_sender()
