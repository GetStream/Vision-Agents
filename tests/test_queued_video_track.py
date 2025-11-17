import asyncio

import av
import numpy as np
import pytest
from PIL import Image

from vision_agents.core.utils.video_track import QueuedVideoTrack
from tests.base_test import BaseTest


class TestQueuedVideoTrack(BaseTest):
    async def test_queued_video_track_initialization(self):
        """Test that QueuedVideoTrack initializes with correct default values."""
        track = QueuedVideoTrack(width=640, height=480, fps=30)

        assert track.width == 640
        assert track.height == 480
        assert track.fps == 30
        assert track._stopped is False
        assert track.last_frame is not None
        assert track.empty_frame is not None

    async def test_queued_video_track_initial_frame_is_blue(self):
        """Test that the initial empty frame is blue."""
        track = QueuedVideoTrack(width=640, height=480, fps=30)

        # Recv should return the initial blue frame
        frame = await track.recv()

        assert frame is not None
        assert frame.width == 640
        assert frame.height == 480

        # Convert to numpy and check if it's blue (RGB: 0, 0, 255)
        frame_array = frame.to_ndarray(format="rgb24")
        # Blue color should dominate
        mean_blue = np.mean(frame_array[:, :, 2])
        mean_red = np.mean(frame_array[:, :, 0])
        mean_green = np.mean(frame_array[:, :, 1])

        assert mean_blue > 200, "Should be predominantly blue"
        assert mean_red < 50, "Should have minimal red"
        assert mean_green < 50, "Should have minimal green"

    async def test_queued_video_track_add_and_recv_frame(self):
        """Test adding a frame and receiving it."""
        track = QueuedVideoTrack(width=640, height=480, fps=30)

        # Create a red frame
        red_image = Image.new("RGB", (640, 480), color="red")
        red_frame = av.VideoFrame.from_image(red_image)

        # Add the frame
        await track.add_frame(red_frame)

        # Receive the frame
        received_frame = await track.recv()

        assert received_frame is not None

        # Check that it's red
        frame_array = received_frame.to_ndarray(format="rgb24")
        mean_red = np.mean(frame_array[:, :, 0])

        assert mean_red > 200, "Should be predominantly red"

    async def test_queued_video_track_returns_last_frame_when_empty(self):
        """Test that recv returns the last frame when queue is empty."""
        track = QueuedVideoTrack(width=640, height=480, fps=30)

        # Create and add a green frame (using RGB values for bright green)
        green_image = Image.new("RGB", (640, 480), color=(0, 255, 0))
        green_frame = av.VideoFrame.from_image(green_image)
        await track.add_frame(green_frame)

        # First recv should get the green frame
        frame1 = await track.recv()
        frame_array1 = frame1.to_ndarray(format="rgb24")
        mean_green1 = np.mean(frame_array1[:, :, 1])
        mean_red1 = np.mean(frame_array1[:, :, 0])
        assert mean_green1 > mean_red1, "First frame should be predominantly green"
        assert mean_green1 > 100, "First frame should have significant green"

        # Second recv with empty queue should return the same green frame
        frame2 = await track.recv()
        frame_array2 = frame2.to_ndarray(format="rgb24")
        mean_green2 = np.mean(frame_array2[:, :, 1])
        mean_red2 = np.mean(frame_array2[:, :, 0])
        assert mean_green2 > mean_red2, (
            "Second frame should still be predominantly green (last frame)"
        )
        assert mean_green2 > 100, "Second frame should still have significant green"

    async def test_queued_video_track_multiple_frames(self):
        """Test adding multiple frames and receiving them."""
        track = QueuedVideoTrack(width=320, height=240, fps=30)

        colors = ["red", "green", "blue", "yellow"]

        # Add multiple frames
        for color in colors:
            image = Image.new("RGB", (320, 240), color=color)
            frame = av.VideoFrame.from_image(image)
            await track.add_frame(frame)

        # Receive frames - should get them in order
        received_count = 0
        for _ in range(len(colors)):
            frame = await track.recv()
            assert frame is not None
            received_count += 1

        assert received_count == len(colors), "Should receive all added frames"

    async def test_queued_video_track_stop(self):
        """Test that track can be stopped and won't accept new frames."""
        track = QueuedVideoTrack(width=640, height=480, fps=30)

        # Add a frame before stopping
        red_image = Image.new("RGB", (640, 480), color="red")
        red_frame = av.VideoFrame.from_image(red_image)
        await track.add_frame(red_frame)

        # Stop the track
        track.stop()

        assert track._stopped is True

        # Try to add another frame - should be ignored
        blue_image = Image.new("RGB", (640, 480), color="blue")
        blue_frame = av.VideoFrame.from_image(blue_image)
        await track.add_frame(blue_frame)

        # Try to receive - should raise exception
        with pytest.raises(Exception, match="Track stopped"):
            await track.recv()

    async def test_queued_video_track_fps_timing(self):
        """Test that recv waits when queue is empty based on FPS."""
        track = QueuedVideoTrack(
            width=320, height=240, fps=10
        )  # 10 FPS = 0.1s per frame

        # First recv should return the initial blue frame quickly
        frame1 = await track.recv()
        assert frame1 is not None

        # Recv without adding a frame should wait approximately 1/fps seconds
        # before returning the last frame
        start = asyncio.get_event_loop().time()
        frame2 = await track.recv()
        end = asyncio.get_event_loop().time()

        elapsed = end - start

        assert frame2 is not None
        # Should wait approximately 0.1s (1/10 FPS) but with some tolerance
        assert elapsed >= 0.08, f"Should wait for FPS interval, took {elapsed}s"
        assert elapsed < 0.2, f"Should not wait too long, took {elapsed}s"

    async def test_queued_video_track_queue_overflow(self):
        """Test behavior when adding more frames than queue capacity."""
        track = QueuedVideoTrack(width=320, height=240, fps=30)

        # Add more than 10 frames (queue maxlen is 10)
        for i in range(15):
            color = (i * 17 % 256, i * 23 % 256, i * 31 % 256)
            image = Image.new("RGB", (320, 240), color=color)
            frame = av.VideoFrame.from_image(image)
            await track.add_frame(frame)

        # Should still be able to receive frames
        frame = await track.recv()
        assert frame is not None

        # Track should still be functional
        assert track._stopped is False

    async def test_queued_video_track_with_real_video(self, bunny_video_track):
        """Test QueuedVideoTrack with frames from actual video."""
        queued_track = QueuedVideoTrack(width=640, height=480, fps=15)

        # Get a few frames from bunny video and add to queued track
        frames_added = 0
        for _ in range(5):
            try:
                frame = await bunny_video_track.recv()
                await queued_track.add_frame(frame)
                frames_added += 1
            except asyncio.CancelledError:
                break

        assert frames_added > 0, "Should have added some frames"

        # Receive frames from queued track
        frames_received = 0
        for _ in range(frames_added):
            frame = await queued_track.recv()
            assert frame is not None
            frames_received += 1

        assert frames_received == frames_added, "Should receive all added frames"
