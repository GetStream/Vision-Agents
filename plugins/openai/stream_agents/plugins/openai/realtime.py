import asyncio
from typing import Any, Optional
from stream_agents.core.llm import realtime
import logging
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData
from .rtc_manager import RTCManager

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
        await self.rtc.connect()
        # Emit connected/ready
        self._emit_connected_event(
            session_config={"model": self.model, "voice": self.voice},
            capabilities=["text", "audio"],
        )

    async def send_audio_pcm(self, audio: PcmData):
        await self.rtc.send_audio_pcm(audio)

    async def send_text(self, text):
        await self.rtc.send_text(str(text))

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        ...

    async def _close_impl(self):
        await self.rtc.close()

    async def _handle_openai_event(self, event: dict) -> None:
        et = event.get("type")
        if et == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                self._emit_transcript_event(text=transcript, user_metadata={"role": "assistant", "source": "openai"})
                self._emit_response_event(text=transcript, response_id=event.get("response_id"), is_complete=True, conversation_item_id=event.get("item_id"))
        elif et == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                self._emit_transcript_event(text=transcript, user_metadata={"role": "user", "source": "openai"})

    async def _handle_audio_output(self, audio_bytes: bytes) -> None:
        # Forward audio as event and to output track if available
        listeners_fn = getattr(self, "listeners", None)
        has_listeners = bool(listeners_fn("audio_output")) if callable(listeners_fn) else False
        if has_listeners:
            self._emit_audio_output_event(audio_data=audio_bytes, sample_rate=48000)
        output_track = getattr(self, "output_track", None)
        if output_track is not None:
            await output_track.write(audio_bytes)
