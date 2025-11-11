from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum
import uuid
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.events.manager import EventManager
from . import events
from .events import TurnStartedEvent, TurnEndedEvent
from ..agents.conversation import Conversation
from ..edge.types import Participant
from ..observability.metrics import turn_detection_latency_ms, turn_errors, Timer


class TurnEvent(Enum):
    """Events that can occur during turn detection (deprecated - use TurnStartedEvent/TurnEndedEvent)."""

    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"


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

    async def warmup(self) -> None:
        """
        Warm up the turn detection service.

        This method can be overridden by implementations to perform
        model loading, connection establishment, or other initialization
        that should happen before the first audio processing.
        """
        pass

    def _emit_start_turn_event(self, event: TurnStartedEvent) -> None:
        event.session_id = self.session_id
        event.plugin_name = self.provider_name
        self.events.send(event)

    def _emit_end_turn_event(self, event: TurnEndedEvent) -> None:
        event.session_id = self.session_id
        event.plugin_name = self.provider_name
        self.events.send(event)

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
        with Timer(turn_detection_latency_ms) as timer:
            timer.attributes["class"] = (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            )
            timer.attributes["provider"] = self.provider_name
            try:
                await self.detect_turn(audio_data, participant, conversation)
            except Exception as e:
                timer.attributes["error"] = type(e).__name__
                turn_errors.add(
                    1, {"provider": self.provider_name, "error_type": type(e).__name__}
                )
                raise

    @abstractmethod
    async def detect_turn(
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

    async def start(self) -> None:
        """Some turn detection systems want to run warmup etc here"""
        self.is_active = True

    async def stop(self) -> None:
        """Again, some turn detection systems want to run cleanup here"""
        self.is_active = False
