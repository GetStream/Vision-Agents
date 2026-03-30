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
