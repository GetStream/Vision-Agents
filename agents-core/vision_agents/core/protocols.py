"""Protocol definitions for Vision Agents framework.

This module defines protocols (structural types) that framework components
can implement, enabling flexible integration with different transport layers
while maintaining type safety.
"""

from typing import Protocol


class Room(Protocol):
    """Protocol for edge transport session/call objects.

    This protocol defines the interface that edge transport implementations
    (e.g., GetStream RTC, Daily, Twilio) must provide to represent a room,
    session, or call object.

    Implementations must provide an identifier, type information, and the
    ability to gracefully disconnect from the room.
    """

    @property
    def id(self) -> str:
        """Unique identifier for the room/session.

        Returns:
            A unique string identifier for this room instance.
        """
        ...

    @property
    def type(self) -> str:
        """Type or category of the room.

        Returns:
            A string representing the room type (e.g., "rtc", "sfu", "mesh").
        """
        ...

    async def leave(self) -> None:
        """Asynchronously leave/disconnect from the room.

        This method should gracefully disconnect from the room, cleaning up
        any resources such as media tracks, network connections, and event
        listeners.

        The implementation should be idempotent - calling leave() multiple
        times should be safe and not raise errors.

        Raises:
            Exception: Implementation-specific exceptions may be raised if
                      disconnection fails critically, though implementations
                      should attempt to handle errors gracefully.
        """
        ...
