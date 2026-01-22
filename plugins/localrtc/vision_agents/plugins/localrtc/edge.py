"""Local RTC Edge Transport implementation."""

from typing import Any, Callable, Optional

import aiortc
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.types import OutputAudioTrack, User
from vision_agents.core.protocols import Room
from vision_agents.core.types import PcmData

from .room import LocalRoom


class LocalEdge(EdgeTransport):
    """Local RTC Edge Transport for managing local audio/video streams."""

    def __init__(self) -> None:
        """Initialize the local edge transport."""
        super().__init__()

    async def create_user(self, user: User) -> None:
        """Create a user in the local edge transport.

        Args:
            user: User object containing user information.
        """
        pass

    def create_audio_track(self) -> OutputAudioTrack:
        """Create an output audio track.

        Returns:
            An OutputAudioTrack instance for audio streaming.
        """
        raise NotImplementedError("create_audio_track not yet implemented")

    async def close(self) -> None:
        """Close the edge transport and clean up resources."""
        pass

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """Open a demo session.

        Args:
            *args: Positional arguments for demo configuration.
            **kwargs: Keyword arguments for demo configuration.
        """
        pass

    async def join(self, *args: Any, **kwargs: Any) -> Room:
        """Join a room.

        Args:
            *args: Positional arguments for join configuration.
            **kwargs: Keyword arguments for join configuration.

        Returns:
            A Room instance representing the joined room.
        """
        room_id = kwargs.get("room_id", "local-room")
        return LocalRoom(room_id=room_id)

    async def publish_tracks(
        self, room: Room, audio_track: Any, video_track: Any
    ) -> None:
        """Publish audio and video tracks to the room.

        Args:
            room: The room to publish tracks to.
            audio_track: Audio track to publish.
            video_track: Video track to publish.
        """
        pass

    async def create_conversation(
        self, call: Any, user: User, instructions: Any
    ) -> None:
        """Create a conversation in the call.

        Args:
            call: The call object.
            user: User object.
            instructions: Conversation instructions.
        """
        pass

    def add_track_subscriber(
        self, track_id: str, callback: Callable[[PcmData], None]
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        """Add a subscriber to a track.

        Args:
            track_id: The ID of the track to subscribe to.
            callback: Callback function to handle PCM data.

        Returns:
            A MediaStreamTrack if available, None otherwise.
        """
        return None
