import pytest
from getstream.video.rtc import audio_track
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import RealtimeAudioOutputEvent
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.lemonslice.lemonslice_avatar_publisher import (
    LemonSliceAvatarPublisher,
)


def _make_publisher(**overrides) -> LemonSliceAvatarPublisher:
    default_kwargs = {
        "agent_id": "test-agent",
        "api_key": "ls-test-key",
        "livekit_url": "wss://test.livekit.cloud",
        "livekit_api_key": "devkey",
        "livekit_api_secret": "devsecret",
    }
    return LemonSliceAvatarPublisher(**{**default_kwargs, **overrides})


class DummyAgent:
    def __init__(self):
        self.events = EventManager()
        self.events.register(TTSAudioEvent)
        self.events.register(RealtimeAudioOutputEvent)


class TestLemonSliceAvatarPublisher:
    def test_init_with_all_args(self):
        pub = _make_publisher()
        assert pub._connected is False
        assert pub.name == "lemonslice_avatar"

    def test_init_with_agent_image_url_instead_of_id(self):
        pub = _make_publisher(
            agent_id=None, agent_image_url="https://example.com/img.png"
        )
        assert pub._client._agent_image_url == "https://example.com/img.png"

    def test_init_missing_agent_identity_raises(self):
        with pytest.raises(ValueError, match="agent_id or agent_image_url"):
            _make_publisher(agent_id=None)

    def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LEMONSLICE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            _make_publisher(api_key=None)

    def test_init_missing_livekit_url_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LIVEKIT_URL", raising=False)
        with pytest.raises(ValueError, match="LiveKit URL required"):
            _make_publisher(livekit_url=None)

    def test_init_missing_livekit_secret_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        with pytest.raises(ValueError, match="LiveKit API key and secret required"):
            _make_publisher(livekit_api_key=None, livekit_api_secret=None)

    def test_init_custom_resolution(self):
        pub = _make_publisher(width=640, height=480)
        track = pub.publish_video_track()
        assert isinstance(track, QueuedVideoTrack)

    def test_publish_video_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_video_track(), QueuedVideoTrack)

    def test_publish_audio_track(self):
        pub = _make_publisher()
        assert isinstance(pub.publish_audio_track(), audio_track.AudioStreamTrack)

    async def test_attach_agent_subscribes_to_tts_and_realtime(self):
        pub = _make_publisher()
        agent = DummyAgent()

        pub.attach_agent(agent)

        assert pub._agent is agent
        assert agent.events.has_subscribers(TTSAudioEvent)
        assert agent.events.has_subscribers(RealtimeAudioOutputEvent)
