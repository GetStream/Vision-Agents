import asyncio
import logging
import time

from vision_agents.core.utils.utils import cancel_and_wait

from .store import SessionKVStore

logger = logging.getLogger(__name__)


class InMemorySessionKVStore(SessionKVStore):
    """In-memory TTL key-value store. Single-node only.

    Useful for development, testing, and single-node deployments.
    For multi-node, swap to a Redis or other networked backend.

    Expired keys are cleaned up both lazily (on access) and periodically
    (via a background task).
    """

    def __init__(self, *, cleanup_interval: float = 60.0) -> None:
        """Initialize the in-memory store.

        Args:
            cleanup_interval: Seconds between periodic expired-key sweeps.
        """
        self._data: dict[str, tuple[bytes, float]] = {}
        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be > 0")
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is not None:
            await cancel_and_wait(self._cleanup_task)
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def close(self) -> None:
        """Cancel the cleanup task and clear all data."""
        if self._cleanup_task is not None:
            await cancel_and_wait(self._cleanup_task)
            self._cleanup_task = None
        self._data.clear()

    async def set(self, key: str, value: bytes, ttl: float) -> None:
        self._data[key] = (value, time.monotonic() + ttl)

    async def mset(self, items: list[tuple[str, bytes, float]]) -> None:
        now = time.monotonic()
        for key, value, ttl in items:
            self._data[key] = (value, now + ttl)

    async def expire(self, *keys: str, ttl: float) -> None:
        now = time.monotonic()
        for key in keys:
            entry = self._data.get(key)
            if entry is None:
                continue
            _, expires_at = entry
            if now >= expires_at:
                del self._data[key]
                continue
            self._data[key] = (entry[0], now + ttl)

    async def get(self, key: str) -> bytes | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._data[key]
            return None
        return value

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        now = time.monotonic()
        results: list[bytes | None] = []
        for key in keys:
            entry = self._data.get(key)
            if entry is None:
                results.append(None)
            elif now >= entry[1]:
                del self._data[key]
                results.append(None)
            else:
                results.append(entry[0])
        return results

    async def keys(self, prefix: str) -> list[str]:
        now = time.monotonic()
        result: list[str] = []
        expired: list[str] = []
        for key, (_, expires_at) in self._data.items():
            if not key.startswith(prefix):
                continue
            if now >= expires_at:
                expired.append(key)
            else:
                result.append(key)
        for key in expired:
            del self._data[key]
        return result

    async def delete(self, keys: list[str]) -> None:
        for key in keys:
            self._data.pop(key, None)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self._cleanup_interval)
            now = time.monotonic()
            expired = [k for k, (_, exp) in self._data.items() if now >= exp]
            for key in expired:
                del self._data[key]
