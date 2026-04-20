import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager

from ..edge.types import Participant
from ..utils.stream import Stream
from . import events

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


class TurnDetector(ABC):
    """Base implementation for turn detection with common functionality."""

    def __init__(
        self, confidence_threshold: float = 0.5, provider_name: Optional[str] = None
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self.is_active = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)
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

    async def stop(self) -> None:
        """Again, some turn detection systems want to run cleanup here"""
        self.is_active = False
