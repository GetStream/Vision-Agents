import pytest
from vision_agents.core.agents.inference import AudioOutputStream
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.anam.anam_avatar import AnamAvatar


def _make_avatar(**overrides) -> AnamAvatar:
    default_kwargs = {
        "avatar_id": "test-avatar",
        "api_key": "test-key",
    }
    return AnamAvatar(**{**default_kwargs, **overrides})


class TestAnamAvatar:
    async def test_init_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            _make_avatar(api_key=None)

    async def test_init_missing_avatar_id_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ANAM_AVATAR_ID", raising=False)
        with pytest.raises(ValueError, match="avatar ID"):
            _make_avatar(avatar_id=None)

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
