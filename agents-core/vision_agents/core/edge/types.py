import abc
import enum
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class User:
    id: Optional[str] = ""
    name: Optional[str] = ""
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any  # Original participant object received from the connectivty provider
    user_id: str  # A user id (doesn't have to be unique)
    id: str  # A unique id of the participant during the call


class TrackType(enum.IntEnum):
    UNSPECIFIED = 0
    AUDIO = 1
    VIDEO = 2
    SCREEN_SHARE = 3
    SCREEN_SHARE_AUDIO = 4


class Connection(abc.ABC):
    """
    Represents an active connection to a real-time communication session.

    A Connection manages the lifecycle of an agent's participation in a call or session,
    tracking participant presence and providing control over the connection state.

    This abstraction allows different transport implementations (e.g., WebRTC)
    to provide consistent connection management to the
    Agent without exposing transport-specific details.

    Lifecycle:
        1. Connection is established by EdgeTransport.join()
        2. wait_for_participant() can be used to wait for other participants
        3. idle_since() tracks when all participants (except agent) have left
        4. close() terminates the connection and cleans up resources

    Example:
        connection = await edge.join(agent, call)
        await connection.wait_for_participant(timeout=30.0)
        # ... call is active ...
        await connection.close()
    """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the connection and clean up resources."""
        pass

    @abc.abstractmethod
    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        """
        Wait for at least one participant (other than the agent) to join.

        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.

        Raises:
            asyncio.TimeoutError: If timeout is reached before a participant joins.
        """
        pass

    @abc.abstractmethod
    def idle_since(self) -> float:
        """
        Return the timestamp when all participants left (except the agent).

        Returns:
            Timestamp (from time.time()) when connection became idle, or 0.0 if active.
        """
        pass
