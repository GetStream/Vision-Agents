import asyncio
import logging
import time
from dataclasses import dataclass

from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    TrackType as StreamTrackType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TrackKey:
    user_id: str
    session_id: str
    stream_track_type: StreamTrackType.ValueType


@dataclass
class _TrackEntry:
    track_id: str
    published: bool


@dataclass
class _PendingTrack:
    user_id: str | None
    session_id: str | None
    webrtc_kind: str
    registered_at: float


def _get_webrtc_kind(stream_track_type: StreamTrackType.ValueType) -> str:
    """Get the expected WebRTC kind (audio/video) for an SFU track type."""
    if stream_track_type in (
        StreamTrackType.TRACK_TYPE_AUDIO,
        StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
    ):
        return "audio"
    elif stream_track_type in (
        StreamTrackType.TRACK_TYPE_VIDEO,
        StreamTrackType.TRACK_TYPE_SCREEN_SHARE,
    ):
        return "video"
    else:
        return "video"


class TrackResolver:
    """Correlate SFU TrackPublished events with WebRTC track_added callbacks."""

    def __init__(self, poll_interval: float = 0.01, pending_ttl: float = 60.0) -> None:
        self._poll_interval = poll_interval
        self._pending_ttl = pending_ttl
        self._track_map: dict[_TrackKey, _TrackEntry] = {}
        self._pending: dict[str, _PendingTrack] = {}
        # in-flight resolves; event set when participant leaves to abort the poll
        self._resolving: dict[_TrackKey, asyncio.Event] = {}
        # serialize resolves per key so duplicate TrackPublishedEvents don't race
        self._resolve_locks: dict[_TrackKey, asyncio.Lock] = {}

    def register(
        self,
        *,
        track_id: str,
        user_id: str | None,
        session_id: str | None,
        webrtc_kind: str,
    ) -> None:
        """Store a WebRTC track_added until the SFU confirms the semantic type."""
        self._pending[track_id] = _PendingTrack(
            user_id=user_id,
            session_id=session_id,
            webrtc_kind=webrtc_kind,
            registered_at=time.monotonic(),
        )

    async def resolve(
        self,
        *,
        user_id: str,
        session_id: str,
        stream_track_type: StreamTrackType.ValueType,
        timeout: float = 10.0,
    ) -> str | None:
        """Resolve track_id for an SFU TrackPublishedEvent.

        Handles known-track reuse, session migration, and the await-pending poll.
        Returns None if the participant left mid-resolve (via cancel()).
        Raises TimeoutError if the WebRTC track_added never arrives within `timeout`.
        """
        track_key = _TrackKey(
            user_id=user_id,
            session_id=session_id,
            stream_track_type=stream_track_type,
        )

        lock = self._resolve_locks.setdefault(track_key, asyncio.Lock())
        async with lock:
            track_id = self._reuse_known(track_key)
            if track_id is None:
                track_id = self._migrate_session(track_key=track_key)
            if track_id is None:
                track_id = await self._await_pending(
                    track_key=track_key, timeout=timeout
                )
                if track_id is None:
                    return None

            self._track_map[track_key] = _TrackEntry(track_id=track_id, published=True)
            return track_id

    def unpublish(
        self,
        *,
        user_id: str,
        session_id: str,
        stream_track_type: StreamTrackType.ValueType,
    ) -> str | None:
        """Flip the track to unpublished. Returns track_id if known, else None."""
        track_key = _TrackKey(
            user_id=user_id,
            session_id=session_id,
            stream_track_type=stream_track_type,
        )
        entry = self._track_map.get(track_key)
        if entry is None:
            return None
        entry.published = False
        return entry.track_id

    def cancel(self, *, user_id: str, session_id: str) -> None:
        """Abort any in-flight resolve for this participant and drop their pending entries."""
        for key, event in self._resolving.items():
            if key.user_id == user_id and key.session_id == session_id:
                event.set()

        orphaned = [
            tid
            for tid, pending in self._pending.items()
            if pending.user_id == user_id and pending.session_id == session_id
        ]
        for tid in orphaned:
            del self._pending[tid]

    def _reuse_known(self, track_key: _TrackKey) -> str | None:
        entry = self._track_map.get(track_key)
        if entry is None:
            return None
        entry.published = True
        return entry.track_id

    def _migrate_session(self, *, track_key: _TrackKey) -> str | None:
        # User reconnected with a new session — the WebRTC media track is reused
        # so track_added won't fire again. Migrate the stale session entry.
        for old_key, old_entry in list(self._track_map.items()):
            if (
                old_key.user_id == track_key.user_id
                and old_key.stream_track_type == track_key.stream_track_type
                and old_key.session_id != track_key.session_id
            ):
                del self._track_map[old_key]
                logger.debug(
                    f"Migrated track for {track_key.user_id} from session "
                    f"{old_key.session_id} to {track_key.session_id}"
                )
                return old_entry.track_id
        return None

    async def _await_pending(
        self, *, track_key: _TrackKey, timeout: float
    ) -> str | None:
        # SFU might send TrackPublishedEvent before WebRTC processes track_added.
        webrtc_kind = _get_webrtc_kind(track_key.stream_track_type)
        cancelled = asyncio.Event()
        self._resolving[track_key] = cancelled
        try:
            elapsed = 0.0
            while elapsed < timeout and not cancelled.is_set():
                track_id = self._match_pending(
                    user_id=track_key.user_id,
                    session_id=track_key.session_id,
                    webrtc_kind=webrtc_kind,
                )
                if track_id is not None:
                    return track_id
                await asyncio.sleep(self._poll_interval)
                elapsed += self._poll_interval

            if cancelled.is_set():
                logger.debug(
                    f"Resolve cancelled: user={track_key.user_id} "
                    f"session={track_key.session_id} left during resolve"
                )
                return None
            raise TimeoutError(
                f"No track_added for user={track_key.user_id} session={track_key.session_id} "
                f"type={StreamTrackType.Name(track_key.stream_track_type)} after {timeout}s "
                f"(pending={self._pending}, map={self._track_map})"
            )
        finally:
            self._resolving.pop(track_key, None)

    def _match_pending(
        self,
        *,
        user_id: str,
        session_id: str,
        webrtc_kind: str,
    ) -> str | None:
        self._clear_stale_pending()

        for tid, pending in list(self._pending.items()):
            if (
                pending.user_id == user_id
                and pending.session_id == session_id
                and pending.webrtc_kind == webrtc_kind
            ):
                del self._pending[tid]
                return tid

        # Fallback: some video track_added callbacks can arrive with user=None.
        # In that case we can still match by WebRTC kind, but only if there
        # is exactly one anonymous candidate — multiple anonymous entries
        # with the same kind would be ambiguous and could misbind.
        anonymous_candidates = [
            tid
            for tid, pending in self._pending.items()
            if pending.user_id is None
            and pending.session_id is None
            and pending.webrtc_kind == webrtc_kind
        ]
        if len(anonymous_candidates) == 1:
            tid = anonymous_candidates[0]
            del self._pending[tid]
            return tid
        return None

    def _clear_stale_pending(self) -> None:
        now = time.monotonic()
        stale = [
            tid
            for tid, pending in self._pending.items()
            if pending.registered_at <= now - self._pending_ttl
        ]
        for tid in stale:
            evicted = self._pending.pop(tid)
            logger.debug(
                f"Evicting stale pending track: id={tid} user={evicted.user_id} "
                f"session={evicted.session_id} kind={evicted.webrtc_kind} "
                f"age={now - evicted.registered_at:.1f}s"
            )

        # Drop resolve locks with no holder and no waiters; safe because
        # setdefault will create a fresh one if a future resolve needs it.
        free_keys = [
            key for key, lock in self._resolve_locks.items() if not lock.locked()
        ]
        for key in free_keys:
            del self._resolve_locks[key]
