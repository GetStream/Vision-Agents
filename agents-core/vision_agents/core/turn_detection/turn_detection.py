from typing import Optional, Dict, Any, Union, Callable, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pyee import EventEmitter
from getstream.video.rtc.track_util import PcmData


class TurnEvent(Enum):
    """Events that can occur during turn detection."""

    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"


@dataclass
class TurnEventData:
    """Data associated with a turn detection event."""

    timestamp: float
    speaker_id: Optional[str] = (
        None  # User id of the speaker who just finished speaking
    )
    duration: Optional[float] = None
    confidence: Optional[float] = None  # confidence level of speaker detection
    custom: Optional[Dict[str, Any]] = None  # extensible custom data


# Type alias for event listener callbacks
EventListener = Callable[[TurnEventData], None]


class TurnDetection(Protocol):
    """Turn Detection shape definition used by the Agent class"""

    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        ...

    def on(
        self, event: str, listener: Optional[EventListener] = None
    ) -> Union[None, Callable]:
        """Add an event listener or use as decorator (from EventEmitter)."""
        ...

    def emit(self, event: str, *args: Any) -> bool:
        """Emit an event (from EventEmitter)."""
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
        Use the event system (emit/on) to notify when turns change.

        Args:
            audio_data: PcmData object containing audio samples from Stream
            user_id: Identifier for the user providing the audio
            metadata: Optional additional metadata about the audio
        """
        ...


class TurnDetector(ABC, EventEmitter):
    """Base implementation for turn detection with common functionality."""

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        super().__init__()  # Initialize EventEmitter
        self._confidence_threshold = confidence_threshold
        self._is_detecting = False

    @abstractmethod
    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        return self._is_detecting

    def _emit_turn_event(
        self, event_type: TurnEvent, event_data: TurnEventData
    ) -> None:
        """Emit a turn detection event."""
        self.emit(event_type.value, event_data)

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
