import abc
from typing import Any, Generic, Optional, TypeVar

import aiortc
from getstream.video.rtc import AudioStreamTrack
from vision_agents.core.events.manager import EventManager

from .call import Call
from .events import (
    AudioReceivedEvent,
    CallEndedEvent,
    TrackAddedEvent,
    TrackRemovedEvent,
)
from .types import User

T_AudioTrack = TypeVar("T_AudioTrack", bound=aiortc.mediastreams.MediaStreamTrack)
T_VideoTrack = TypeVar("T_VideoTrack", bound=aiortc.mediastreams.MediaStreamTrack)


class EdgeTransport(abc.ABC, Generic[T_AudioTrack, T_VideoTrack]):
    """Abstract base class for edge transports.

    Required Events (implementations must emit these):
        - AudioReceivedEvent: When audio is received from a participant
        - TrackAddedEvent: When a media track is added to the call
        - TrackRemovedEvent: When a media track is removed from the call
        - CallEndedEvent: When the call ends
    """

    events: EventManager

    def __init__(self):
        super().__init__()
        self.events = EventManager()
        # Register required events that all EdgeTransport implementations must emit
        self.events.register(
            AudioReceivedEvent,
            TrackAddedEvent,
            TrackRemovedEvent,
            CallEndedEvent,
        )

    @abc.abstractmethod
    async def create_user(self, user: User):
        pass

    @abc.abstractmethod
    def create_audio_track(self) -> AudioStreamTrack:
        pass

    @abc.abstractmethod
    async def close(self):
        pass

    @abc.abstractmethod
    def open_demo(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def join(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def publish_tracks(
        self, audio_track: Optional[T_AudioTrack], video_track: Optional[T_VideoTrack]
    ):
        pass

    @abc.abstractmethod
    async def create_conversation(self, call: Call, user: User, instructions):
        pass

    @abc.abstractmethod
    def add_track_subscriber(
        self, track_id: str
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        pass

    @abc.abstractmethod
    async def send_custom_event(self, data: dict[str, Any]) -> None:
        """Send a custom event to all participants watching the call.

        Args:
            data: Custom event payload (must be JSON-serializable, max 5KB).
        """
        pass
