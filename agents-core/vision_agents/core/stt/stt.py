import abc
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.transcript import TranscriptMode
from vision_agents.core.edge.types import Participant
from vision_agents.core.events.manager import EventManager
from vision_agents.core.observability import MetricsCollector
from vision_agents.core.stt.events import (
    STTConnectedEvent,
    STTDisconnectedEvent,
    STTErrorEvent,
)
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.core.base import Component
from vision_agents.core.utils.stream import Stream

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResponse:
    confidence: float | None = None
    language: str | None = None
    processing_time_ms: float | None = None
    audio_duration_ms: float | None = None
    model_name: str | None = None
    other: dict | None = None


@dataclass
class Transcript:
    """Event emitted when a complete transcript is available."""

    participant: Participant
    mode: TranscriptMode
    text: str
    response: TranscriptResponse = field(default_factory=TranscriptResponse)

    def __post_init__(self):
        if not self.text:
            raise ValueError("Transcript text cannot be empty")

    @property
    def final(self) -> bool:
        return self.mode == "final"

    @property
    def confidence(self) -> float | None:
        return self.response.confidence

    @property
    def language(self) -> str | None:
        return self.response.language

    @property
    def processing_time_ms(self) -> float | None:
        return self.response.processing_time_ms

    @property
    def audio_duration_ms(self) -> float | None:
        return self.response.audio_duration_ms

    @property
    def model_name(self) -> str | None:
        return self.response.model_name


class STT(Component):
    """
    Abstract base class for Speech-to-Text implementations.
    """

    closed: bool = False
    started: bool = False
    turn_detection: bool = False  # if the STT supports turn detection
    eager_turn_detection: bool = False  # if the STT supports turn detection

    def __init__(
        self,
        provider_name: Optional[str] = None,
    ):
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__

        self.events = EventManager()
        self.metrics = MetricsCollector()

        self._output: Stream[Transcript | TurnEnded | TurnStarted] = Stream()

    @property
    def output(self) -> Stream[Transcript | TurnEnded | TurnStarted]:
        """Pipeline output stream: consumers iterate, subclasses push via send_nowait."""
        return self._output

    @abc.abstractmethod
    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        pass

    async def start(self):
        if self.started:
            raise ValueError("STT is already started, dont call this method twice")
        self.started = True

    async def clear(self):
        """Clear any pending audio or state. Override in subclasses if needed."""
        self._output.clear()

    async def close(self):
        self.closed = True
        self._output.close()

    def _emit_transcript_event(
        self,
        text: str,
        participant: Participant,
        response: TranscriptResponse,
        *,
        mode: TranscriptMode = "final",
    ) -> None:
        """Push a Transcript to the output stream and record metrics on final."""
        transcript = Transcript(
            participant=participant,
            mode=mode,
            text=text,
            response=response,
        )
        self._output.send_nowait(transcript)
        if transcript.final:
            self.metrics.on_stt_transcript(
                provider=self.provider_name,
                model=transcript.model_name,
                language=transcript.language,
                processing_time_ms=transcript.processing_time_ms,
                audio_duration_ms=transcript.audio_duration_ms,
            )

    def _emit_turn_ended_event(
        self,
        participant: Participant,
        *,
        confidence: float = 0.5,
        eager: bool = False,
        duration_ms: Optional[float] = None,
        trailing_silence_ms: Optional[float] = None,
    ) -> None:
        """Push a TurnEnded to the output stream and record the turn metric."""
        self._output.send_nowait(
            TurnEnded(
                participant=participant,
                confidence=confidence,
                eager=eager,
                duration_ms=duration_ms,
                trailing_silence_ms=trailing_silence_ms,
            )
        )
        self.metrics.on_turn_ended(
            provider=self.provider_name,
            duration_ms=duration_ms,
            trailing_silence_ms=trailing_silence_ms,
        )

    def _emit_turn_started_event(
        self,
        participant: Participant,
        *,
        confidence: float = 0.5,
    ) -> None:
        """Push a TurnStarted to the output stream."""
        self._output.send_nowait(
            TurnStarted(participant=participant, confidence=confidence)
        )

    def _emit_error_event(self, error: Exception, *, context: str = "") -> None:
        """Record metric and emit STTErrorEvent. Caller may also re-raise."""
        self.metrics.on_stt_error(
            provider=self.provider_name,
            error_type=type(error).__name__,
        )
        self.events.send(
            STTErrorEvent(
                plugin_name=self.provider_name,
                error=error,
                context=context,
            )
        )

    def _on_connected(self) -> None:
        """Emit STTConnectedEvent."""
        self.events.send(STTConnectedEvent(plugin_name=self.provider_name))

    def _on_disconnected(
        self, reason: Optional[str] = None, clean: bool = True
    ) -> None:
        """Emit STTDisconnectedEvent."""
        self.events.send(
            STTDisconnectedEvent(
                plugin_name=self.provider_name, reason=reason, clean=clean
            )
        )
