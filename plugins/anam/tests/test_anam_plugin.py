import av
import numpy as np
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
from vision_agents.plugins.anam.anam_avatar_publisher import (
    AnamAvatarPublisher,
    _resize_frame,
)


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

    def test_init_buffer_seconds_sets_queue_size(self):
        pub = _make_publisher(buffer_seconds=2.5)
        assert pub._sync._video_track._pending.maxlen == 75

    def test_init_zero_buffer_seconds_raises(self):
        with pytest.raises(ValueError, match="buffer_seconds must be > 0"):
            _make_publisher(buffer_seconds=0)

    def test_init_custom_fps(self):
        pub = _make_publisher(fps=60, buffer_seconds=2.0)
        assert pub._sync._video_track.fps == 60
        assert pub._sync._video_track._pending.maxlen == 120

    def test_init_zero_fps_raises(self):
        with pytest.raises(ValueError, match="fps must be > 0"):
            _make_publisher(fps=0)

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


def _solid_rgb_frame(
    width: int, height: int, color: tuple[int, int, int]
) -> av.VideoFrame:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return av.VideoFrame.from_ndarray(arr, format="rgb24")


class TestResizeFrame:
    def test_resizes_down_to_target_size(self):
        src = _solid_rgb_frame(1280, 720, (200, 50, 50))

        out = _resize_frame(src, 640, 480)

        assert out.width == 640
        assert out.height == 480

    def test_aspect_preserved_letterboxes_top_and_bottom(self):
        # 2:1 source into 4:3 target: inner box 640x320, vertical bars at top/bottom.
        src = _solid_rgb_frame(200, 100, (50, 200, 50))

        out = _resize_frame(src, 640, 480)
        arr = out.to_ndarray(format="rgb24")

        assert out.width == 640
        assert out.height == 480
        assert np.array_equal(arr[0, 320, :], np.array([0, 0, 0], dtype=np.uint8))
        assert np.array_equal(arr[479, 320, :], np.array([0, 0, 0], dtype=np.uint8))
        # Center of frame is inside the inner box; reformat dithers, so allow tolerance.
        center = arr[240, 320, :].astype(int)
        assert center[1] > center[0] and center[1] > center[2]

    def test_aspect_preserved_letterboxes_left_and_right(self):
        # 1:2 source into 4:3 target: inner box 240x480, bars on left/right.
        src = _solid_rgb_frame(100, 200, (50, 50, 200))

        out = _resize_frame(src, 640, 480)
        arr = out.to_ndarray(format="rgb24")

        assert out.width == 640
        assert out.height == 480
        assert np.array_equal(arr[240, 0, :], np.array([0, 0, 0], dtype=np.uint8))
        assert np.array_equal(arr[240, 639, :], np.array([0, 0, 0], dtype=np.uint8))
        center = arr[240, 320, :].astype(int)
        assert center[2] > center[0] and center[2] > center[1]

    def test_exact_size_returns_same_dimensions(self):
        src = _solid_rgb_frame(640, 480, (100, 100, 100))

        out = _resize_frame(src, 640, 480)

        assert out.width == 640
        assert out.height == 480
