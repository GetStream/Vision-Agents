from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import uuid
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from . import events
from ..agents.conversation import Conversation
from ..edge.types import Participant


class TurnEvent(Enum):
    """Events that can occur during turn detection (deprecated - use TurnStartedEvent/TurnEndedEvent)."""

    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"


@dataclass
class TurnEventData:
    """Data associated with a turn detection event (deprecated - use TurnStartedEvent/TurnEndedEvent)."""

    timestamp: float
    speaker_id: Optional[str] = (
        None  # User id of the speaker who just finished speaking
    )
    duration: Optional[float] = None
    confidence: Optional[float] = None  # confidence level of speaker detection
    custom: Optional[Dict[str, Any]] = None  # extensible custom data


# Type alias for event listener callbacks (deprecated)
EventListener = Callable[[TurnEventData], None]


class TurnDetector(ABC):
    """Base implementation for turn detection with common functionality."""

    def __init__(
        self, 
        confidence_threshold: float = 0.5,
        provider_name: Optional[str] = None
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self.is_active = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)

    def _emit_turn_event(
        self, event_type: TurnEvent, event_data: TurnEventData
    ) -> None:
        """
        Emit a turn detection event using the new event system.
        
        Args:
            event_type: The type of turn event (TURN_STARTED or TURN_ENDED)
            event_data: Data associated with the event
        """
        if event_type == TurnEvent.TURN_STARTED:
            self.events.send(events.TurnStartedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                speaker_id=event_data.speaker_id,
                confidence=event_data.confidence,
                duration=event_data.duration,
                custom=event_data.custom,
            ))
        elif event_type == TurnEvent.TURN_ENDED:
            self.events.send(events.TurnEndedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                speaker_id=event_data.speaker_id,
                confidence=event_data.confidence,
                duration=event_data.duration,
                custom=event_data.custom,
            ))

    @abstractmethod
    async def process_audio(
        self,
        audio_data: PcmData,
        participant: Participant,
        conversation: Optional[Conversation],
    ) -> None:
        """Process the audio and trigger turn start or turn end events

        Args:
            audio_data: PcmData object containing audio samples from Stream
            participant: Participant that's speaking, includes user data
            conversation: Transcription/ chat history, sometimes useful for turn detection
        """

    ...

    def start(self) -> None:
        """Some turn detection systems want to run warmup etc here"""
        self.is_active = True

    def stop(self) -> None:
        """Again, some turn detection systems want to run cleanup here"""
        self.is_active = False
