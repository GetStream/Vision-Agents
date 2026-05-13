import asyncio

import pytest
from getstream.video.rtc import audio_track
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import (
    RealtimeAudioOutputDoneEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.turn_detection import TurnStartedEvent
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.anam.anam_avatar_publisher import AnamAvatarPublisher


def _make_publisher(**overrides) -> AnamAvatarPublisher:
    default_kwargs = {
        "avatar_id": "test-avatar",
        "api_key": "test-key",
    }
    return AnamAvatarPublisher(**{**default_kwargs, **overrides})


class DummyAgent:
    def __init__(self):
        self.events = EventManager()
        self.events.register(TTSAudioEvent)
        self.events.register(RealtimeAudioOutputEvent)
        self.events.register(RealtimeAudioOutputDoneEvent)
        self.events.register(TurnStartedEvent)


class TestAnamAvatarPublisher:
    def test_init_with_all_args(self):
        pub = _make_publisher()
        assert pub.name == "anam_avatar"

    def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            _make_publisher(api_key=None)

    def test_init_missing_avatar_id_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_AVATAR_ID", raising=False)
        with pytest.raises(ValueError, match="avatar ID"):
            _make_publisher(avatar_id=None)

    def test_init_custom_resolution(self):
        pub = _make_publisher(width=640, height=480)
        track = pub.publish_video_track()
        assert isinstance(track, QueuedVideoTrack)
        assert track.width == 640
        assert track.height == 480

    def test_init_odd_width_raises(self):
        with pytest.raises(ValueError, match="width must be a positive even integer"):
            _make_publisher(width=641, height=480)

    def test_init_odd_height_raises(self):
        with pytest.raises(ValueError, match="height must be a positive even integer"):
            _make_publisher(width=640, height=481)

    def test_publish_video_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_video_track(), QueuedVideoTrack)

    def test_publish_audio_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_audio_track(), audio_track.AudioStreamTrack)

    async def test_attach_agent_subscribes_to_events(self):
        pub = _make_publisher()
        agent = DummyAgent()

        pub.attach_agent(agent)

        assert pub._real_agent is agent
        assert agent.events.has_subscribers(TTSAudioEvent)
        assert agent.events.has_subscribers(RealtimeAudioOutputEvent)
        assert agent.events.has_subscribers(RealtimeAudioOutputDoneEvent)
        assert agent.events.has_subscribers(TurnStartedEvent)

    async def test_start_is_idempotent(self):
        """Repeated `start()` runs `_connect()` once and short-circuits after.

        Callers warm up the publisher before `agent.join()` to keep Anam's
        ~3s session creation off the join critical path. The framework
        then calls `start()` again from `_apply("start")` — that second
        call must be a no-op.
        """
        pub = _make_publisher()

        call_count = 0

        async def fake_connect():
            nonlocal call_count
            call_count += 1

        pub._connect = fake_connect  # type: ignore[method-assign]

        await pub.start()
        await pub.start()
        await pub.start()

        assert call_count == 1
        assert pub._started is True

    async def test_start_serializes_concurrent_callers(self):
        """Two tasks calling `start()` at the same time produce one connect.

        Without the lock, both would see `_started is False` and race into
        `_connect()`, double-opening the Anam session.
        """
        pub = _make_publisher()

        call_count = 0
        proceed = asyncio.Event()

        async def slow_connect():
            nonlocal call_count
            call_count += 1
            await proceed.wait()

        pub._connect = slow_connect  # type: ignore[method-assign]

        task_a = asyncio.create_task(pub.start())
        task_b = asyncio.create_task(pub.start())
        await asyncio.sleep(0)  # let both tasks acquire/queue on the lock
        proceed.set()
        await asyncio.gather(task_a, task_b)

        assert call_count == 1

    async def test_start_failure_allows_retry(self):
        """A failing first `start()` leaves `_started=False` so the next
        attempt can retry instead of silently no-op'ing on a half-connected
        publisher."""
        pub = _make_publisher()

        attempts = 0

        async def flaky_connect():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("anam unreachable")

        pub._connect = flaky_connect  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="anam unreachable"):
            await pub.start()
        assert pub._started is False

        await pub.start()
        assert pub._started is True
        assert attempts == 2
