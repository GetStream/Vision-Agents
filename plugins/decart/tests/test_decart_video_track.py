"""Tests for DecartVideoTrack."""

import av
import pytest
from PIL import Image

from vision_agents.plugins.decart.decart_video_track import DecartVideoTrack


@pytest.fixture
def sample_image():
    """Test image fixture."""
    return Image.new("RGB", (640, 480), color="blue")


@pytest.fixture
def sample_frame(sample_image):
    """Test av.VideoFrame fixture."""
    return av.VideoFrame.from_image(sample_image)


@pytest.fixture
def sample_frame_large():
    """Test av.VideoFrame fixture with different size."""
    image = Image.new("RGB", (1920, 1080), color="red")
    return av.VideoFrame.from_image(image)


class TestDecartVideoTrack:
    """Tests for DecartVideoTrack."""

    def test_init_default_dimensions(self):
        """Test initialization with default dimensions."""
        track = DecartVideoTrack()
        assert track.width == 1280
        assert track.height == 720
        assert not track._stopped
        track.stop()

    def test_init_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        track = DecartVideoTrack(width=1920, height=1080)
        assert track.width == 1920
        assert track.height == 1080
        assert not track.is_stopped
        track.stop()

    @pytest.mark.asyncio
    async def test_add_frame_correct_size(self, sample_frame):
        """Test adding frame with correct size."""
        track = DecartVideoTrack(width=640, height=480)
        await track.add_frame(sample_frame)
        assert track.frame_queue.qsize() == 1
        track.stop()

    @pytest.mark.asyncio
    async def test_add_frame_requires_resize(self, sample_frame_large):
        """Test adding frame that requires resize."""
        track = DecartVideoTrack(width=1280, height=720)
        await track.add_frame(sample_frame_large)
        assert track.frame_queue.qsize() == 1
        received_frame = await track.frame_queue.get()
        assert received_frame.width == 1280
        assert received_frame.height == 720
        track.stop()

    @pytest.mark.asyncio
    async def test_add_frame_ignored_when_stopped(self, sample_frame):
        """Test that add_frame is ignored when track is stopped."""
        track = DecartVideoTrack()
        track.stop()
        await track.add_frame(sample_frame)
        assert track.frame_queue.qsize() == 0
        track.stop()

    @pytest.mark.asyncio
    async def test_recv_returns_frame(self, sample_frame):
        """Test that recv returns a frame when available."""
        track = DecartVideoTrack(width=640, height=480)
        await track.add_frame(sample_frame)
        received_frame = await track.recv()
        assert received_frame is not None
        assert received_frame.width == 640
        assert received_frame.height == 480
        assert received_frame.pts is not None
        assert received_frame.time_base is not None
        track.stop()

    @pytest.mark.asyncio
    async def test_recv_returns_placeholder_when_no_frames(self):
        """Test that recv returns placeholder frame when no frames available."""
        track = DecartVideoTrack()
        received_frame = await track.recv()
        assert received_frame is not None
        assert received_frame.width == 1280
        assert received_frame.height == 720
        track.stop()

    @pytest.mark.asyncio
    async def test_recv_raises_when_stopped(self):
        """Test that recv raises exception when track is stopped."""
        track = DecartVideoTrack()
        track.stop()
        with pytest.raises(Exception, match="Track stopped"):
            await track.recv()

    @pytest.mark.asyncio
    async def test_recv_returns_latest_frame(self, sample_frame):
        """Test that recv returns the latest frame."""
        track = DecartVideoTrack(width=640, height=480)
        await track.add_frame(sample_frame)
        frame1 = await track.recv()
        await track.add_frame(sample_frame)
        frame2 = await track.recv()
        assert frame1 is not None
        assert frame2 is not None
        track.stop()

    def test_stop(self):
        """Test stopping the video track."""
        track = DecartVideoTrack()
        assert not track._stopped
        track.stop()
        assert track._stopped
