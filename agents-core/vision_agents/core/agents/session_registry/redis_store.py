import inspect
import logging
from collections.abc import AsyncIterator

import redis.asyncio as redis

from .store import SessionKVStore

logger = logging.getLogger(__name__)


class RedisSessionKVStore(SessionKVStore):
    """Redis-backed TTL key-value store with pub/sub.

    Suitable for multi-node deployments where session state must be
    shared across processes or machines.

    Args:
        client: An existing ``redis.asyncio.Redis`` client. Caller owns
            the lifecycle.
        url: A Redis connection URL (e.g. ``redis://localhost:6379/0``).
            The store creates and owns the client.
        key_prefix: Prefix prepended to every key for namespacing.
    """

    def __init__(
        self,
        *,
        client: redis.Redis | None = None,
        url: str | None = None,
        key_prefix: str = "vision_agents:",
    ) -> None:
        if client is not None and url is not None:
            raise ValueError("Provide either a Redis client or a URL, not both")

        self._redis: redis.Redis
        if client is not None:
            self._owns_client = False
            self._redis = client
        elif url is not None:
            self._owns_client = True
            self._redis = redis.from_url(url)
        else:
            raise ValueError("Provide either a Redis client or a URL")

        self._key_prefix = key_prefix

    def _prefixed(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        return key[len(self._key_prefix) :]

    async def start(self) -> None:
        """Open the Redis connection and verify it with a PING."""
        # Handle non-specific Union return type here
        ping = self._redis.ping()
        if inspect.iscoroutine(ping):
            await ping

        connection_kwargs = self._redis.connection_pool.connection_kwargs
        host = connection_kwargs.get("host", "unknown")
        port = connection_kwargs.get("port", 6379)
        logger.info("RedisSessionKVStore connected to %s:%s", host, port)

    async def close(self) -> None:
        """Close the Redis connection if this store owns it."""
        if self._owns_client:
            await self._redis.aclose()

    async def set(self, key: str, value: bytes, ttl: float) -> None:
        """Store a value via SET with PX (millisecond TTL)."""
        await self._redis.set(self._prefixed(key), value, px=int(ttl * 1000))

    async def mset(self, items: list[tuple[str, bytes, float]]) -> None:
        """Atomically store multiple values via a MULTI/EXEC pipeline."""
        async with self._redis.pipeline() as pipe:
            for key, value, ttl in items:
                await pipe.set(self._prefixed(key), value, px=int(ttl * 1000))
            await pipe.execute()

    async def expire(self, *keys: str, ttl: float) -> None:
        """Refresh TTL on one or more keys via a transactional PEXPIRE pipeline."""
        if not keys:
            return
        async with self._redis.pipeline() as pipe:
            for key in keys:
                await pipe.pexpire(self._prefixed(key), int(ttl * 1000))
            await pipe.execute()

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key via GET."""
        return await self._redis.get(self._prefixed(key))

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        """Retrieve multiple values by key via MGET, preserving order."""
        if not keys:
            return []
        prefixed = [self._prefixed(k) for k in keys]
        return await self._redis.mget(prefixed)

    async def keys(self, prefix: str) -> list[str]:
        """Return all keys matching a prefix via SCAN (non-blocking)."""
        pattern = f"{self._prefixed(prefix)}*"
        result: list[str] = []
        async for key in self._redis.scan_iter(match=pattern, count=100):
            decoded = key.decode() if isinstance(key, bytes) else key
            result.append(self._strip_prefix(decoded))
        return result

    async def delete(self, keys: list[str]) -> None:
        """Delete one or more keys via DEL. Missing keys are ignored."""
        if not keys:
            return
        prefixed = [self._prefixed(k) for k in keys]
        await self._redis.delete(*prefixed)

    async def publish(self, channel: str, message: bytes) -> None:
        """Publish a message to a channel via PUBLISH."""
        await self._redis.publish(self._prefixed(channel), message)

    async def subscribe(self, channel: str) -> AsyncIterator[bytes]:
        """Subscribe to a channel via PubSub. Yields messages as bytes."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(self._prefixed(channel))
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield message["data"]
        finally:
            await pubsub.unsubscribe(self._prefixed(channel))
            await pubsub.aclose()
