import uuid
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from vision_agents.core.observability import MetricsCollector

from ..edge.types import Participant
from ..base import Component
from ..utils.stream import Stream

if TYPE_CHECKING:
    from vision_agents.core.agents.conversation import Conversation


@dataclass
class TurnStarted:
    """
    Event emitted when a speaker starts their turn.
    """

    participant: Participant
    confidence: float


@dataclass
class TurnEnded:
    participant: Participant
    confidence: float
    eager: bool = False
    trailing_silence_ms: Optional[float] = None
    duration_ms: Optional[float] = None


class TurnDetector(Component):
    """Base implementation for turn detection with common functionality."""

    def __init__(
        self, confidence_threshold: float = 0.5, provider_name: Optional[str] = None
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self.is_active = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.metrics = MetricsCollector()
        self._output: Stream[TurnEnded | TurnStarted] = Stream()

    @property
    def output(self) -> Stream[TurnEnded | TurnStarted]:
        """Pipeline output stream: consumers iterate, subclasses push via send_nowait."""
        return self._output

    @abstractmethod
    async def process_audio(
        self,
        data: PcmData,
        participant: Participant,
        conversation: "Conversation | None" = None,
    ) -> None:
        """Process the audio and trigger turn start or turn end events

        Args:
            data: PcmData object containing audio samples from Stream
            participant: Participant that's speaking, includes user data
            conversation: Transcription/ chat history, sometimes useful for turn detection
        """

    async def start(self) -> None:
        """Some turn detection systems want to run warmup etc here"""
        if self.is_active:
            raise ValueError(f"start() has already been called for {self}")
        self.is_active = True

    async def close(self) -> None:
        """Again, some turn detection systems want to run cleanup here"""
        self.is_active = False

    async def _emit_turn_ended_event(
        self,
        participant: Participant,
        *,
        confidence: float = 0.5,
        eager: bool = False,
        duration_ms: Optional[float] = None,
        trailing_silence_ms: Optional[float] = None,
    ) -> None:
        """Send TurnEnded to output stream and record the turn metric."""
        await self._output.send(
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

    async def _emit_turn_started_event(
        self,
        participant: Participant,
        *,
        confidence: float = 0.5,
    ) -> None:
        """Send TurnStarted to output stream."""
        await self._output.send(
            TurnStarted(participant=participant, confidence=confidence)
        )
