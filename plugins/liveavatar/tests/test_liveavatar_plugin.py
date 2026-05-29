import pytest
from vision_agents.core.agents.inference import AudioOutputStream
from vision_agents.core.utils.video_track import QueuedVideoTrack
from vision_agents.plugins.liveavatar.liveavatar_avatar import LiveAvatar


@pytest.fixture
async def avatar() -> LiveAvatar:
    return LiveAvatar(avatar_id="test-avatar", api_key="lv-test-key")


class TestLiveAvatar:
    async def test_init_missing_api_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LIVEAVATAR_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            LiveAvatar(avatar_id="x", api_key=None)

    async def test_init_missing_avatar_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LIVEAVATAR_AVATAR_ID", raising=False)
        with pytest.raises(ValueError, match="Avatar ID required"):
            LiveAvatar(avatar_id=None, api_key="k")

    async def test_video_output_dimensions(self) -> None:
        avatar = LiveAvatar(avatar_id="x", api_key="k", width=640, height=480)
        track = avatar.video_output()
        assert isinstance(track, QueuedVideoTrack)
        assert track.width == 640
        assert track.height == 480

    async def test_audio_output_type(self, avatar: LiveAvatar) -> None:
        assert isinstance(avatar.audio_output(), AudioOutputStream)
