from typing import Optional, Dict, Any, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import uuid
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from vision_agents.core.events import PluginInitializedEvent
from . import events


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


class TurnDetection(Protocol):
    """Turn Detection shape definition used by the Agent class"""

    events: EventManager

    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        ...

    # --- Unified high-level interface used by Agent ---
    def start(self) -> None:
        """Start detection (convenience alias to start_detection)."""
        ...

    def stop(self) -> None:
        """Stop detection (convenience alias to stop_detection)."""
        ...

    async def process_audio(
        self,
        audio_data: PcmData,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest PcmData audio for a user.

        The implementation should track participants internally as audio comes in.
        Use the event system (events.send) to notify when turns change.

        Args:
            audio_data: PcmData object containing audio samples from Stream
            user_id: Identifier for the user providing the audio
            metadata: Optional additional metadata about the audio
        """
        ...


class TurnDetector(ABC):
    """Base implementation for turn detection with common functionality."""

    def __init__(
        self, 
        confidence_threshold: float = 0.5,
        provider_name: Optional[str] = None
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._is_detecting = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)
        self.events.send(PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="TurnDetection",
            provider=self.provider_name,
        ))

    @abstractmethod
    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        return self._is_detecting

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
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest PcmData audio for a user.

        The implementation should track participants internally as audio comes in.
        Use the event system (emit/on) to notify when turns change.

        Args:
            audio_data: PcmData object containing audio samples from Stream
            user_id: Identifier for the user providing the audio
            metadata: Optional additional metadata about the audio
        """

    ...

    # Convenience aliases to align with the unified protocol expected by Agent
    @abstractmethod
    def start(self) -> None:
        """Start detection (alias for start_detection)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop detection (alias for stop_detection)."""
        ...
