import asyncio

import pytest
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    TrackType as StreamTrackType,
)
from vision_agents.plugins.getstream._track_resolver import TrackResolver


@pytest.fixture
def resolver():
    return TrackResolver(poll_interval=0.005)


class TestTrackResolver:
    async def test_known_track_reuse(self, resolver):
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s1",
            webrtc_kind="audio",
        )
        first = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )

        resolver.unpublish(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )

        second = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )

        assert first == "t1"
        assert second == "t1"

    async def test_session_migration(self, resolver):
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s_old",
            webrtc_kind="audio",
        )
        await resolver.resolve(
            user_id="u1",
            session_id="s_old",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )

        # New session arrives without a fresh register() — same WebRTC track is reused.
        migrated = await resolver.resolve(
            user_id="u1",
            session_id="s_new",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
            timeout=0.1,
        )
        assert migrated == "t1"

        # The old session entry is gone; the new one owns the track now.
        old_unpublish = resolver.unpublish(
            user_id="u1",
            session_id="s_old",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )
        new_unpublish = resolver.unpublish(
            user_id="u1",
            session_id="s_new",
            stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
        )
        assert old_unpublish is None
        assert new_unpublish == "t1"

    async def test_pending_arrives_first(self, resolver):
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s1",
            webrtc_kind="video",
        )
        track_id = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
        )
        assert track_id == "t1"

    async def test_track_published_arrives_first(self, resolver):
        resolve_task = asyncio.create_task(
            resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
                timeout=1.0,
            )
        )
        await asyncio.sleep(0.02)
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s1",
            webrtc_kind="video",
        )
        track_id = await resolve_task
        assert track_id == "t1"

    async def test_anonymous_fallback_success(self, resolver):
        resolver.register(
            track_id="t_anon",
            user_id=None,
            session_id=None,
            webrtc_kind="video",
        )
        track_id = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
        )
        assert track_id == "t_anon"

    async def test_anonymous_fallback_ambiguous(self, resolver):
        resolver.register(
            track_id="t_anon_a",
            user_id=None,
            session_id=None,
            webrtc_kind="video",
        )
        resolver.register(
            track_id="t_anon_b",
            user_id=None,
            session_id=None,
            webrtc_kind="video",
        )
        with pytest.raises(TimeoutError):
            await resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
                timeout=0.05,
            )

    async def test_timeout_no_pending(self, resolver):
        with pytest.raises(TimeoutError):
            await resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
                timeout=0.05,
            )

    async def test_cancel_during_resolve(self, resolver):
        resolve_task = asyncio.create_task(
            resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
                timeout=10.0,
            )
        )
        await asyncio.sleep(0.02)

        resolver.cancel(user_id="u1", session_id="s1")

        track_id = await asyncio.wait_for(resolve_task, timeout=0.5)
        assert track_id is None

    async def test_stale_pending_is_evicted(self):
        # Short TTL so we can verify the eviction without long sleeps.
        resolver = TrackResolver(poll_interval=0.005, pending_ttl=0.05)

        # Stale anonymous video registered first; would normally make the
        # fallback ambiguous when a second anonymous video arrives.
        resolver.register(
            track_id="t_stale",
            user_id=None,
            session_id=None,
            webrtc_kind="video",
        )
        await asyncio.sleep(0.08)

        resolver.register(
            track_id="t_fresh",
            user_id=None,
            session_id=None,
            webrtc_kind="video",
        )
        track_id = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
            timeout=0.5,
        )
        assert track_id == "t_fresh"

    async def test_concurrent_resolves_serialized(self, resolver):
        # Duplicate TrackPublishedEvent (e.g. from republish_tracks) starts two
        # resolves for the same key. Both must succeed with the same track_id.
        task_a = asyncio.create_task(
            resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
                timeout=0.5,
            )
        )
        task_b = asyncio.create_task(
            resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_AUDIO,
                timeout=0.5,
            )
        )
        await asyncio.sleep(0.02)
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s1",
            webrtc_kind="audio",
        )
        results = await asyncio.gather(task_a, task_b)
        assert results == ["t1", "t1"]

    async def test_cancel_drops_named_pending(self, resolver):
        # Named pending was registered (track_added fired), participant leaves
        # before TrackPublishedEvent. cancel() should drop the orphan.
        resolver.register(
            track_id="t_orphan",
            user_id="u1",
            session_id="s1",
            webrtc_kind="video",
        )
        resolver.cancel(user_id="u1", session_id="s1")

        # If the orphan were still around, the next anonymous video would be
        # ambiguous against an exact-tuple lookup attempt — but here we just
        # verify a fresh resolve for the same key times out (no stale match).
        with pytest.raises(TimeoutError):
            await resolver.resolve(
                user_id="u1",
                session_id="s1",
                stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
                timeout=0.05,
            )

    async def test_cancel_before_resolve_is_noop(self, resolver):
        # Cancel arrives first; nothing in flight, no-op.
        resolver.cancel(user_id="u1", session_id="s1")

        # Subsequent resolve runs normally and finds the matching pending.
        resolver.register(
            track_id="t1",
            user_id="u1",
            session_id="s1",
            webrtc_kind="video",
        )
        track_id = await resolver.resolve(
            user_id="u1",
            session_id="s1",
            stream_track_type=StreamTrackType.TRACK_TYPE_VIDEO,
            timeout=0.5,
        )
        assert track_id == "t1"
