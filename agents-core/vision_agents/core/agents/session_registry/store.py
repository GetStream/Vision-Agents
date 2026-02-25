import abc
from collections.abc import AsyncIterator


class SessionKVStore(abc.ABC):
    """
    Abstract TTL key-value storage backend with pub/sub for the SessionRegistry.

    The storage layer is a generic key-value store that works with bytes,
    plus a pub/sub mechanism for real-time event delivery.
    It knows nothing about nodes or sessions — the SessionRegistry owns
    the key scheme and all serialization.

    Implementations should use TTL-based key expiry. Records that are not
    refreshed within the TTL period are considered expired and may be
    garbage-collected by the backend.

    Key conventions (managed by SessionRegistry):
        - ``sessions/{session_id}`` → JSON-serialized SessionInfo
        - ``call_sessions/{call_id}/{session_id}`` → session_id (secondary index)
        - ``close_requests/{session_id}`` → empty bytes (close flag)

    Channel conventions (managed by SessionRegistry):
        - ``events/{session_id}`` → per-session real-time events
    """

    async def start(self) -> None:
        """
        Initialize the storage backend (open connections, etc.).

        Default implementation is a no-op.
        """

    async def close(self) -> None:
        """
        Close any connections held by this storage backend.

        Default implementation is a no-op.
        """

    async def __aenter__(self) -> "SessionKVStore":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @abc.abstractmethod
    async def set(self, key: str, value: bytes, ttl: float) -> None:
        """
        Store a value with a TTL.

        If the key already exists, the value and TTL are overwritten (upsert).
        The record should expire after ``ttl`` seconds if not refreshed.

        Args:
            key: The key to store.
            value: The value as bytes.
            ttl: Time-to-live in seconds.
        """
        ...

    @abc.abstractmethod
    async def mset(self, items: list[tuple[str, bytes, float]]) -> None:
        """
        Store multiple values with TTLs.

        Each item is a ``(key, value, ttl)`` tuple. Semantics per key are
        the same as :meth:`set`.

        Args:
            items: A list of (key, value, ttl) tuples.
        """
        ...

    @abc.abstractmethod
    async def expire(self, *keys: str, ttl: float) -> None:
        """
        Refresh the TTL on one or more existing keys without changing their values.

        Keys that do not exist are silently ignored.

        Args:
            *keys: One or more keys to update.
            ttl: New time-to-live in seconds.
        """
        ...

    @abc.abstractmethod
    async def get(self, key: str) -> bytes | None:
        """
        Retrieve a value by key.

        Returns:
            The value as bytes, or None if the key does not exist or has expired.
        """
        ...

    @abc.abstractmethod
    async def mget(self, keys: list[str]) -> list[bytes | None]:
        """
        Retrieve multiple values by key.

        Returns a list of values in the same order as the input keys.
        Missing or expired keys are returned as None.

        Args:
            keys: The keys to retrieve.

        Returns:
            A list of values (or None) in the same order as the input keys.
        """
        ...

    @abc.abstractmethod
    async def keys(self, prefix: str) -> list[str]:
        """
        Return all non-expired keys that start with ``prefix``.

        Args:
            prefix: The key prefix to match.

        Returns:
            A list of matching key strings.
        """
        ...

    @abc.abstractmethod
    async def delete(self, keys: list[str]) -> None:
        """
        Delete one or more keys.

        Keys that do not exist are silently ignored.

        Args:
            keys: The keys to delete.
        """
        ...

    @abc.abstractmethod
    async def publish(self, channel: str, message: bytes) -> None:
        """
        Publish a message to a channel.

        Fire-and-forget: there is no guarantee that any subscriber
        will receive the message.

        Args:
            channel: The channel name to publish to.
            message: The message payload as bytes.
        """
        ...

    @abc.abstractmethod
    def subscribe(self, channel: str) -> AsyncIterator[bytes]:
        """
        Subscribe to a channel.

        Returns an async iterator that yields messages as they arrive.
        The iterator should be used with ``async for``::

            async for message in storage.subscribe("events/session-123"):
                handle(message)

        Args:
            channel: The channel name to subscribe to.

        Returns:
            An async iterator yielding message payloads as bytes.
        """
        ...
