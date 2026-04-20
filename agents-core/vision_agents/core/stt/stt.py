import abc
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.transcript import TranscriptMode
from vision_agents.core.edge.types import Participant
from vision_agents.core.events.manager import EventManager
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
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
    text: str = ""
    confidence: float | None = None
    language: str | None = None
    processing_time_ms: float | None = None
    audio_duration_ms: float | None = None
    model_name: str | None = None
    response: TranscriptResponse = field(default_factory=TranscriptResponse)

    def __post_init__(self):
        if not self.text:
            raise ValueError("Transcript text cannot be empty")

    @property
    def final(self) -> bool:
        return self.mode == "final"


class STT(abc.ABC):
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
