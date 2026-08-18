import asyncio
import logging
from typing import Any, Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

from ._backend import Backend
from ._socket import Socket

logger = logging.getLogger(__name__)

# SAMPLE_RATE is what the router transcribes at, and what every provider behind it wants.
SAMPLE_RATE = 16_000


class STT(stt.STT):
    """Transcription routed through the acceleration backend.

    For a pipeline that stays in Python: the turns, the model and the conversation are all
    here, and only the transcribing is somewhere else. Failover and cost tracking work
    exactly as they do inside a session, because it is the same router doing the routing.
    """

    def __init__(
        self,
        target: str,
        language: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Route transcription to `target`.

        Args:
            target: A `provider/model` name or a capability shortcut such as
                `en-low-latency`.
            language: A language hint, which narrows the candidates.
            tags: Cost labels carried onto every request.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        super().__init__(provider_name="stream")
        self.model = target
        self.language = language
        self.tags = tags or {}
        self.backend = Backend(url=url, customer_id=customer_id)

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task] = None
        self._speaker: Optional[Participant] = None

    async def start(self):
        """Open the socket and start reading transcripts off it."""
        await super().start()
        self._socket = Socket(
            self.backend.socket("/v1/stt/stream"), self.backend.headers
        )
        await self._socket.connect()
        await self._socket.send(
            {
                "type": "start",
                "target": self.model,
                "languages": [self.language] if self.language else [],
                "tags": self.tags,
                "sample_rate": SAMPLE_RATE,
            }
        )
        self._reader = asyncio.create_task(self._read())
        self._on_connected()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """Send audio to be transcribed, resampled to what the router expects."""
        if self.closed or self._socket is None or not self._socket.open:
            return

        self._speaker = participant
        resampled = pcm_data.resample(SAMPLE_RATE, 1)
        await self._socket.send_audio(resampled.samples.tobytes())

    async def close(self):
        """Close the socket and stop reading it."""
        if self._reader is not None:
            await cancel_and_wait(self._reader)
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None
            self._on_disconnected()
        await super().close()

    async def _read(self) -> None:
        """Turn what the router says into transcripts on the output stream."""
        if self._socket is None:
            return

        async for frame in self._socket.frames():
            if isinstance(frame, bytes):
                continue
            self._received(frame)

    def _received(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type", "")

        if kind == "transcript":
            text = frame.get("text", "")
            if not text or self._speaker is None:
                return
            self._emit_transcript_event(
                text=text,
                participant=self._speaker,
                response=TranscriptResponse(
                    confidence=frame.get("confidence"),
                    language=frame.get("language"),
                    processing_time_ms=frame.get("processing_time_ms"),
                    audio_duration_ms=frame.get("audio_duration_ms"),
                    model_name=frame.get("model"),
                ),
                mode="final" if frame.get("final") else "delta",
            )
        elif kind == "error":
            self._emit_error_event(
                RuntimeError(frame.get("error", "")), context=frame.get("context", "")
            )
