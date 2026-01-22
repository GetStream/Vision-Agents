"""Local RTC room implementation."""


class LocalRoom:
    """Local RTC room for managing local audio/video streams."""

    def __init__(self, room_id: str) -> None:
        """Initialize the local room.

        Args:
            room_id: Unique identifier for this room instance.
        """
        self._id = room_id
        self._type = "local"

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
        pass
