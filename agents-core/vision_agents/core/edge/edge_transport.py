import abc
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

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
from .types import Connection, User

if TYPE_CHECKING:
    from vision_agents.core import Agent

T_Call = TypeVar("T_Call", bound=Call)


class EdgeTransport(abc.ABC, Generic[T_Call]):
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
        """Create or update a user in the transport system.

        Args:
            user: User object containing id, name, and optional image.
        """
        pass

    @abc.abstractmethod
    async def create_call(
        self, call_id: str, agent_user_id: Optional[str] = None, **kwargs
    ) -> T_Call:
        """Create a new call or retrieve an existing one.

        Args:
            call_id: Unique identifier for the call.
            agent_user_id: Optional user ID for the agent in this call.
            **kwargs: Additional transport-specific call configuration.

        Returns:
            Call: A Call object representing the call session.
        """
        pass

    @abc.abstractmethod
    def create_audio_track(self) -> AudioStreamTrack:
        """Create an audio stream track for sending audio to the call.

        Returns:
            AudioStreamTrack: A track that can be used to stream audio data.
        """
        pass

    @abc.abstractmethod
    async def close(self):
        """Close the transport and clean up all resources.

        This should disconnect from any active calls, release network resources,
        and perform any necessary cleanup.
        """
        pass

    @abc.abstractmethod
    def open_demo(self, *args, **kwargs):
        """Open a demo/preview interface for the call.

        Args:
            *args: Transport-specific positional arguments.
            **kwargs: Transport-specific keyword arguments.
        """
        pass

    @abc.abstractmethod
    async def join(self, agent: "Agent", call: T_Call, **kwargs) -> Connection:
        """Join a call and establish a connection.

        This method connects the agent to an active call session, setting up
        the necessary infrastructure for real-time audio/video communication.
        Implementations should configure media subscriptions, set up event handlers,
        and establish the transport-specific connection.

        Args:
            agent: The Agent instance joining the call.
            call: Call object representing the call session to join.
            **kwargs: Additional transport-specific configuration options.

        Returns:
            Connection: An active connection implementing the Connection interface,
                which provides methods for managing the connection lifecycle.
        """
        pass

    @abc.abstractmethod
    async def publish_tracks(
        self,
        audio_track: Optional[aiortc.MediaStreamTrack],
        video_track: Optional[aiortc.MediaStreamTrack],
    ):
        """Publish audio and/or video tracks to the active call.

        Args:
            audio_track: Optional audio track to publish.
            video_track: Optional video track to publish.
        """
        pass

    @abc.abstractmethod
    async def create_conversation(self, call: Call, user: User, instructions: str):
        pass

    @abc.abstractmethod
    def add_track_subscriber(self, track_id: str) -> Optional[aiortc.VideoStreamTrack]:
        pass

    @abc.abstractmethod
    async def send_custom_event(self, data: dict[str, Any]) -> None:
        """Send a custom event to all participants watching the call.

        Args:
            data: Custom event payload (must be JSON-serializable, max 5KB).
        """
        pass
