from typing import Optional, Dict, Any, Union, Callable, Protocol
from dataclasses import dataclass
from enum import Enum
from pyee import EventEmitter
from getstream.models import User


class TurnEvent(Enum):
    """Events that can occur during turn detection."""

    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    MINI_PAUSE_DETECTED = "mini_pause_detected"
    MAX_PAUSE_REACHED = "max_pause_reached"


@dataclass
class TurnEventData:
    """Data associated with a turn detection event."""

    timestamp: float
    speaker: Optional[User] = None
    duration: Optional[float] = None
    confidence: Optional[float] = None  # confidence level of speaker detection
    audio_level: Optional[float] = None  # volume/energy level
    custom: Optional[Dict[str, Any]] = None  # extensible custom data


# Type alias for event listener callbacks
EventListener = Callable[[TurnEventData], None]


class TurnDetectionProtocol(Protocol):
    """Protocol defining the interface for turn detection implementations."""

    @property
    def mini_pause_duration(self) -> float:
        """Get the mini pause duration in seconds."""
        ...

    @property
    def max_pause_duration(self) -> float:
        """Get the max pause duration in seconds."""
        ...

    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        ...

    def start_detection(self) -> None:
        """Start the turn detection process."""
        ...

    def stop_detection(self) -> None:
        """Stop the turn detection process."""
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

    def add_participant(
        self, user: User
    ) -> None:  # pragma: no cover - optional by implementation
        """Register a participant for turn tracking."""
        ...

    async def process_audio(
        self,
        audio_data: Any,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:  # pragma: no cover - optional by implementation
        """Ingest raw PCM/bytes audio for a user."""
        ...

    async def process_audio_track(
        self, track: Any, user_id: str
    ) -> None:  # pragma: no cover - optional by implementation
        """Process audio directly from a WebRTC/Media track if supported."""
        ...

    def on_agent_turn(
        self, callback: Callable
    ) -> None:  # pragma: no cover - optional by implementation
        """Register callback when it's the agent's turn to respond."""
        ...

    def on_participant_turn(
        self, callback: Callable
    ) -> None:  # pragma: no cover - optional by implementation
        """Register callback when a participant starts a turn."""
        ...

    def detect_turn(
        self, audio_data: bytes
    ) -> bool:  # pragma: no cover - optional by implementation
        """Simple polling gate indicating whether the agent should respond now."""
        ...

    def get_stats(
        self,
    ) -> Dict[str, Any]:  # pragma: no cover - optional by implementation
        """Return implementation-specific stats for observability."""
        ...

    def get_insights(
        self,
    ) -> Dict[str, Any]:  # pragma: no cover - optional by implementation
        """Return higher-level conversation insights if available."""
        ...


class BaseTurnDetector(EventEmitter):
    """Base implementation for turn detection with common functionality."""

    def __init__(self, mini_pause_duration: float, max_pause_duration: float) -> None:
        super().__init__()  # Initialize EventEmitter
        self._validate_durations(mini_pause_duration, max_pause_duration)
        self._mini_pause_duration = mini_pause_duration
        self._max_pause_duration = max_pause_duration
        self._is_detecting = False

    @staticmethod
    def _validate_durations(
        mini_pause_duration: float, max_pause_duration: float
    ) -> None:
        """Validate the pause duration parameters."""
        if mini_pause_duration <= 0:
            raise ValueError("mini_pause_duration must be positive")
        if max_pause_duration <= 0:
            raise ValueError("max_pause_duration must be positive")
        if mini_pause_duration >= max_pause_duration:
            raise ValueError("mini_pause_duration must be less than max_pause_duration")

    @property
    def mini_pause_duration(self) -> float:
        """Get the mini pause duration in seconds."""
        return self._mini_pause_duration

    @property
    def max_pause_duration(self) -> float:
        """Get the max pause duration in seconds."""
        return self._max_pause_duration

    def is_detecting(self) -> bool:
        """Check if turn detection is currently active."""
        return self._is_detecting

    def _emit_turn_event(
        self, event_type: TurnEvent, event_data: TurnEventData
    ) -> None:
        """Emit a turn detection event."""
        self.emit(event_type.value, event_data)

    def start_detection(self) -> None:
        """Start the turn detection process."""
        self._is_detecting = True

    def stop_detection(self) -> None:
        """Stop the turn detection process."""
        self._is_detecting = False

    # Convenience aliases to align with the unified protocol expected by Agent
    def start(self) -> None:
        """Start detection (alias for start_detection)."""
        self.start_detection()

    def stop(self) -> None:
        """Stop detection (alias for stop_detection)."""
        self.stop_detection()
