import json
import logging
import time
from dataclasses import asdict
from typing import Self
from uuid import uuid4

from .in_memory_store import InMemorySessionKVStore
from .store import SessionKVStore
from .types import SessionInfo

logger = logging.getLogger(__name__)


class SessionRegistry:
    """Stateless facade over shared storage for multi-node session management.

    The registry handles serialization, key naming, and TTL management.
    It holds no session state — the caller (AgentLauncher) owns all session
    tracking and drives refreshes.

    When no storage backend is provided, an :class:`InMemorySessionKVStore`
    is used by default (suitable for single-node / development).
    """

    def __init__(
        self,
        store: SessionKVStore | None = None,
        *,
        node_id: str | None = None,
        ttl: float = 30.0,
    ) -> None:
        self._store = store or InMemorySessionKVStore()
        self._node_id = node_id or str(uuid4())
        if ttl <= 0:
            raise ValueError("ttl must be > 0")

        self._ttl = ttl

    @property
    def node_id(self) -> str:
        return self._node_id

    async def start(self) -> None:
        """Initialize the storage backend."""
        await self._store.start()

    async def stop(self) -> None:
        """Close the storage backend."""
        await self._store.close()

    async def register(self, call_id: str, session_id: str) -> None:
        """Write a new session record to storage."""
        now = time.time()
        info = SessionInfo(
            session_id=session_id,
            call_id=call_id,
            node_id=self._node_id,
            started_at=now,
            metrics_updated_at=now,
        )
        await self._store.set(
            self._session_key(call_id, session_id),
            json.dumps(asdict(info)).encode(),
            self._ttl,
        )

    async def remove(self, call_id: str, session_id: str) -> None:
        """Delete all storage keys for a session."""
        await self._store.delete(
            [
                self._session_key(call_id, session_id),
                self._close_key(call_id, session_id),
            ]
        )

    async def update_metrics(
        self, call_id: str, session_id: str, metrics: dict[str, int | float | None]
    ) -> None:
        """Push updated metrics for a session into storage."""
        key = self._session_key(call_id, session_id)
        raw = await self._store.get(key)
        if raw is None:
            return
        data = json.loads(raw)
        data["metrics"] = metrics
        data["metrics_updated_at"] = time.time()
        await self._store.set(
            key, json.dumps(data).encode(), self._ttl, only_if_exists=True
        )

    async def refresh(self, sessions: dict[str, str]) -> None:
        """Refresh TTLs for the given sessions.

        Args:
            sessions: mapping of session_id to call_id.
        """
        if not sessions:
            return
        keys = [
            self._session_key(call_id, session_id)
            for session_id, call_id in sessions.items()
        ]
        await self._store.expire(*keys, ttl=self._ttl)

    async def get_close_requests(self, sessions: dict[str, str]) -> list[str]:
        """Return session IDs that have a pending close request.

        Args:
            sessions: mapping of session_id to call_id.
        """
        if not sessions:
            return []
        session_ids = list(sessions.keys())
        keys = [self._close_key(sessions[sid], sid) for sid in session_ids]
        values = await self._store.mget(keys)
        return [sid for sid, val in zip(session_ids, values) if val is not None]

    async def request_close(self, call_id: str, session_id: str) -> None:
        """Set a close flag for a session (async close from any node)."""
        await self._store.set(self._close_key(call_id, session_id), b"", self._ttl)

    async def get(self, call_id: str, session_id: str) -> SessionInfo | None:
        """Look up a session by ID from shared storage."""
        raw = await self._store.get(self._session_key(call_id, session_id))
        if raw is None:
            return None
        return SessionInfo.from_dict(json.loads(raw))

    async def get_for_call(self, call_id: str) -> list[SessionInfo]:
        """Return all sessions for a given call across all nodes."""
        keys = await self._store.keys(f"sessions/{call_id}/")
        if not keys:
            return []
        values = await self._store.mget(keys)
        return [
            SessionInfo.from_dict(json.loads(raw)) for raw in values if raw is not None
        ]

    @staticmethod
    def _session_key(call_id: str, session_id: str) -> str:
        return f"sessions/{call_id}/{session_id}"

    @staticmethod
    def _close_key(call_id: str, session_id: str) -> str:
        return f"close_requests/{call_id}/{session_id}"

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
