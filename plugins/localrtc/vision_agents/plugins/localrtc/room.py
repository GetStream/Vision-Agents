"""Local RTC room implementation."""

from typing import Any, Dict

from vision_agents.core.types import TrackType


class LocalRoom:
    """Local RTC room for managing local audio/video streams."""

    def __init__(self, room_id: str, room_type: str = "default") -> None:
        """Initialize the local room.

        Args:
            room_id: Unique identifier for this room instance.
            room_type: Type or category of the room (default: 'default').
        """
        self._id = room_id
        self._type = room_type
        self._active = True
        self._tracks: Dict[TrackType, Any] = {}

    @property
    def id(self) -> str:
        """Unique identifier for the room/session.

        Returns:
            A unique string identifier for this room instance.
        """
        return self._id

    @property
    def type(self) -> str:
        """Type or category of the room.

        Returns:
            A string representing the room type.
        """
        return self._type

    async def leave(self) -> None:
        """Asynchronously leave/disconnect from the room.

        This method gracefully disconnects from the room, cleaning up
        any resources such as media tracks and event listeners.
        """
        self._active = False
        self._tracks.clear()
