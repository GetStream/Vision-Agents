import asyncio
from typing import Any, Optional, List

from getstream.video.rtc.audio_track import AudioStreamTrack

from stream_agents.core.llm import realtime
import logging
import numpy as np
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from .rtc_manager import RTCManager
from openai.types.realtime import *

from ...core.edge.types import Participant
from ...core.processors import BaseProcessor

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
        # Wire callbacks so we can emit audio/events upstream
        self.rtc.set_event_callback(self._handle_openai_event)
        self.rtc.set_audio_callback(self._handle_audio_output)

        await self.rtc.connect()
        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio"],
        )

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None,
      participant: Participant = None):
        await self.rtc.send_text(text)

    async def simple_audio_response(self, audio: PcmData):
        await self.rtc.send_audio_pcm(audio)

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
        logger.info(f"🎵 Forwarding audio output: {len(audio_bytes)}")

        await self.output_track.write(audio_bytes)

    async def _watch_video_track(self, track, fps: int = 1) -> None:
        # TODO: only do this once?
        #self.rtc.set_video_callback(self._handle_video_output)
        # Delegate to RTC manager to swap the negotiated sender's track
        await self.rtc.start_video_sender(track, fps)

    async def _stop_watching_video_track(self) -> None:
        await self.rtc.stop_video_sender()
