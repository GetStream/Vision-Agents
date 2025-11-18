import pytest
from unittest.mock import patch
from vision_agents.plugins.heygen import AvatarPublisher, VideoQuality
from vision_agents.plugins.heygen.heygen_video_track import HeyGenVideoTrack
from vision_agents.plugins.heygen.heygen_rtc_manager import HeyGenRTCManager
from vision_agents.plugins.heygen.heygen_session import HeyGenSession


class TestHeyGenSession:
    """Tests for HeyGenSession."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        session = HeyGenSession(
            avatar_id="test_avatar",
            quality=VideoQuality.HIGH,
            api_key="test_key",
        )

        assert session.avatar_id == "test_avatar"
        assert session.quality == VideoQuality.HIGH
        assert session.api_key == "test_key"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="HeyGen API key required"):
                HeyGenSession(avatar_id="test_avatar")


class TestHeyGenVideoTrack:
    """Tests for HeyGenVideoTrack."""

    def test_init(self):
        """Test video track initialization."""
        track = HeyGenVideoTrack(width=1920, height=1080)

        assert track.width == 1920
        assert track.height == 1080
        assert not track._stopped

    def test_stop(self):
        """Test stopping the video track."""
        track = HeyGenVideoTrack()
        track.stop()

        assert track._stopped


class TestHeyGenRTCManager:
    """Tests for HeyGenRTCManager."""

    def test_init(self):
        """Test RTC manager initialization."""
        with patch.object(HeyGenSession, "__init__", return_value=None):
            manager = HeyGenRTCManager(
                avatar_id="test_avatar",
                quality=VideoQuality.MEDIUM,
                api_key="test_key",
            )

            assert manager.pc is None
            assert not manager._connected

    def test_is_connected_property(self):
        """Test is_connected property."""
        with patch.object(HeyGenSession, "__init__", return_value=None):
            manager = HeyGenRTCManager(api_key="test_key")

            assert not manager.is_connected

            manager._connected = True
            assert manager.is_connected


class TestAvatarPublisher:
    """Tests for AvatarPublisher."""

    def test_init(self):
        """Test avatar publisher initialization."""
        with patch.object(HeyGenRTCManager, "__init__", return_value=None):
            publisher = AvatarPublisher(
                avatar_id="test_avatar",
                quality=VideoQuality.HIGH,
                resolution=(1920, 1080),
                api_key="test_key",
            )

            assert publisher.avatar_id == "test_avatar"
            assert publisher.quality == VideoQuality.HIGH
            assert publisher.resolution == (1920, 1080)
            assert not publisher._connected

    def test_publish_video_track(self):
        """Test publishing video track."""
        with patch.object(HeyGenRTCManager, "__init__", return_value=None):
            publisher = AvatarPublisher(api_key="test_key")
            # Set _connected to True to avoid creating async task
            publisher._connected = True
            publisher._connection_task = None

            track = publisher.publish_video_track()

            assert isinstance(track, HeyGenVideoTrack)

    def test_state(self):
        """Test state method."""
        with patch.object(HeyGenRTCManager, "__init__", return_value=None):
            publisher = AvatarPublisher(
                avatar_id="test_avatar",
                quality=VideoQuality.MEDIUM,
                api_key="test_key",
            )
            # Mock the _connected attribute on the RTC manager
            publisher.rtc_manager._connected = False

            state = publisher.state()

            assert state["avatar_id"] == "test_avatar"
            assert state["quality"] == VideoQuality.MEDIUM
            assert "connected" in state
            assert "rtc_connected" in state
