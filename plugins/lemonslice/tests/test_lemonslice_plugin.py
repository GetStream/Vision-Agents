import pytest
from vision_agents.core.agents.inference import AudioOutputStream
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.lemonslice.lemonslice_avatar import LemonSliceAvatar


def _make_avatar(**overrides) -> LemonSliceAvatar:
    default_kwargs = {
        "agent_id": "test-agent",
        "api_key": "ls-test-key",
        "livekit_url": "wss://test.livekit.cloud",
        "livekit_api_key": "devkey",
        "livekit_api_secret": "devsecret",
    }
    return LemonSliceAvatar(**{**default_kwargs, **overrides})


class TestLemonSliceAvatar:
    async def test_init_with_agent_image_url_instead_of_id(self):
        avatar = _make_avatar(
            agent_id=None, agent_image_url="https://example.com/img.png"
        )
        assert avatar._client._agent_image_url == "https://example.com/img.png"

    async def test_init_missing_agent_identity_raises(self, monkeypatch):
        monkeypatch.delenv("LEMONSLICE_AGENT_ID", raising=False)
        with pytest.raises(ValueError, match="agent_id or agent_image_url"):
            _make_avatar(agent_id=None)

    async def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LEMONSLICE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            _make_avatar(api_key=None)

    async def test_init_missing_livekit_url_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LIVEKIT_URL", raising=False)
        with pytest.raises(ValueError, match="LiveKit URL required"):
            _make_avatar(livekit_url=None)

    async def test_init_missing_livekit_secret_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        with pytest.raises(ValueError, match="LiveKit API key and secret required"):
            _make_avatar(livekit_api_key=None, livekit_api_secret=None)

    async def test_video_output(self):
        avatar = _make_avatar(width=640, height=480)
        track = avatar.video_output()
        assert isinstance(track, QueuedVideoTrack)
        assert track.width == 640
        assert track.height == 480

    async def test_init_odd_width_raises(self):
        with pytest.raises(ValueError, match="width must be a positive even integer"):
            _make_avatar(width=641, height=480)

    async def test_init_odd_height_raises(self):
        with pytest.raises(ValueError, match="height must be a positive even integer"):
            _make_avatar(width=640, height=481)

    async def test_audio_output(self):
        avatar = _make_avatar()
        assert isinstance(avatar.audio_output(), AudioOutputStream)
