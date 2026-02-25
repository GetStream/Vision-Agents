import json
import logging
import time
from dataclasses import asdict
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

    async def register(self, session_id: str, call_id: str) -> None:
        """Write a new session record to storage."""
        now = time.time()
        info = SessionInfo(
            session_id=session_id,
            call_id=call_id,
            node_id=self._node_id,
            started_at=now,
            metrics_updated_at=now,
        )
        await self._store.mset(
            [
                (
                    f"sessions/{session_id}",
                    json.dumps(asdict(info)).encode(),
                    self._ttl,
                ),
                (
                    f"call_sessions/{call_id}/{session_id}",
                    session_id.encode(),
                    self._ttl,
                ),
            ]
        )

    async def remove(self, session_id: str) -> None:
        """Delete all storage keys for a session."""
        raw = await self._store.get(f"sessions/{session_id}")
        if raw is None:
            return
        call_id = json.loads(raw)["call_id"]
        await self._delete_keys(session_id, call_id)

    async def update_metrics(
        self, session_id: str, metrics: dict[str, int | float | None]
    ) -> None:
        """Push updated metrics for a session into storage."""
        raw = await self._store.get(f"sessions/{session_id}")
        if raw is None:
            return
        data = json.loads(raw)
        data["metrics"] = metrics
        data["metrics_updated_at"] = time.time()
        await self._store.set(
            f"sessions/{session_id}",
            json.dumps(data).encode(),
            self._ttl,
        )

    async def refresh(self, sessions: dict[str, str]) -> None:
        """Refresh TTLs for the given sessions.

        Args:
            sessions: mapping of session_id to call_id.
        """
        if not sessions:
            return
        keys: list[str] = []
        for session_id, call_id in sessions.items():
            keys.append(f"sessions/{session_id}")
            keys.append(f"call_sessions/{call_id}/{session_id}")
        await self._store.expire(*keys, ttl=self._ttl)

    async def get_close_requests(self, session_ids: list[str]) -> list[str]:
        """Return session IDs that have a pending close request."""
        if not session_ids:
            return []
        keys = [f"close_requests/{sid}" for sid in session_ids]
        values = await self._store.mget(keys)
        return [sid for sid, val in zip(session_ids, values) if val is not None]

    async def request_close(self, session_id: str) -> None:
        """Set a close flag for a session (async close from any node)."""
        await self._store.set(f"close_requests/{session_id}", b"", self._ttl)

    async def get(self, session_id: str) -> SessionInfo | None:
        """Look up a session by ID from shared storage."""
        raw = await self._store.get(f"sessions/{session_id}")
        if raw is None:
            return None
        return SessionInfo(**json.loads(raw))

    async def get_for_call(self, call_id: str) -> list[SessionInfo]:
        """Return all sessions for a given call across all nodes."""
        index_keys = await self._store.keys(f"call_sessions/{call_id}/")
        if not index_keys:
            return []
        session_ids = [k.rsplit("/", 1)[-1] for k in index_keys]
        session_keys = [f"sessions/{sid}" for sid in session_ids]
        values = await self._store.mget(session_keys)
        return [SessionInfo(**json.loads(raw)) for raw in values if raw is not None]

    async def _delete_keys(self, session_id: str, call_id: str) -> None:
        await self._store.delete(
            [
                f"sessions/{session_id}",
                f"call_sessions/{call_id}/{session_id}",
                f"close_requests/{session_id}",
            ]
        )
