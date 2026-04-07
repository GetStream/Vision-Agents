import asyncio

import av
import numpy as np
import pytest
from getstream.video.rtc.track_util import AudioFormat, PcmData
from PIL import Image
from vision_agents.core.utils.av_synchronizer import AVSynchronizer
from vision_agents.core.utils.video_track import VideoTrackClosedError


def _make_frame(width: int, height: int, color: str = "red") -> av.VideoFrame:
    return av.VideoFrame.from_image(Image.new("RGB", (width, height), color=color))


def _make_stereo_pcm(num_samples: int, sample_rate: int = 48000) -> PcmData:
    """Create stereo s16 PCM silence with a known number of samples per channel."""
    samples = np.zeros(num_samples * 2, dtype=np.int16)
    return PcmData(
        samples=samples,
        sample_rate=sample_rate,
        format=AudioFormat.S16,
        channels=2,
    )


class TestAVSynchronizer:
    @pytest.fixture
    def sync(self) -> AVSynchronizer:
        return AVSynchronizer(
            width=640,
            height=480,
            fps=30,
            max_queue_size=10,
            sample_rate=48000,
            channels=2,
        )

    async def test_default_initialization(self):
        sync = AVSynchronizer()
        assert sync.video_track.width == 1920
        assert sync.video_track.height == 1080
        assert sync.video_track.fps == 30
        assert sync.audio_track.sample_rate == 48000
        assert sync.audio_track.channels == 2

    async def test_custom_initialization(self):
        sync = AVSynchronizer(
            width=640,
            height=480,
            fps=15,
            sample_rate=16000,
            channels=1,
        )
        assert sync.video_track.width == 640
        assert sync.video_track.height == 480
        assert sync.video_track.fps == 15
        assert sync.audio_track.sample_rate == 16000
        assert sync.audio_track.channels == 1

    async def test_video_frame_immediate_when_no_audio(self, sync: AVSynchronizer):
        """With no audio buffered, a video frame should be available immediately."""
        await sync.write_video(_make_frame(640, 480, "red"))

        received = await sync.video_track.recv()
        arr = received.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 0]) > 200, "Should be the red frame"

    async def test_video_frame_delayed_by_audio_buffer(self, sync: AVSynchronizer):
        """A video frame written after audio should be held until the delay elapses."""
        # Buffer 0.5s of audio
        await sync.write_audio(_make_stereo_pcm(num_samples=24000, sample_rate=48000))

        await sync.write_video(_make_frame(640, 480, "red"))

        # recv() immediately — the red frame hasn't been released yet
        received = await sync.video_track.recv()
        arr = received.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 2]) > 200, "Should still be the blue empty frame"
        assert np.mean(arr[:, :, 0]) < 50, "Red channel should be low"

        # Wait for the delay to elapse
        await asyncio.sleep(0.55)

        received = await sync.video_track.recv()
        arr = received.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 0]) > 200, "Should now be the red frame"

    async def test_multiple_frames_released_in_order(self, sync: AVSynchronizer):
        """Frames queued with delay should release in FIFO order."""
        await sync.write_audio(_make_stereo_pcm(num_samples=14400, sample_rate=48000))

        await sync.write_video(_make_frame(640, 480, "red"))
        await sync.write_video(_make_frame(640, 480, "lime"))

        # Wait for release
        await asyncio.sleep(0.35)

        first = await sync.video_track.recv()
        arr = first.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 0]) > 200, "First frame should be red"

        second = await sync.video_track.recv()
        arr = second.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 1]) > 200, "Second frame should be lime/green"

    async def test_recv_repeats_last_frame(self, sync: AVSynchronizer):
        """After a frame is released, recv keeps returning it until a new one arrives."""
        await sync.write_video(_make_frame(640, 480, "red"))

        first = await sync.video_track.recv()
        second = await sync.video_track.recv()

        arr1 = first.to_ndarray(format="rgb24")
        arr2 = second.to_ndarray(format="rgb24")
        assert np.mean(arr1[:, :, 0]) > 200
        assert np.mean(arr2[:, :, 0]) > 200, "Should repeat the red frame"

    async def test_flush_clears_pending_video_and_audio(self, sync: AVSynchronizer):
        """After flush, buffered audio is gone and pending video frames are discarded."""
        # Buffer audio and a delayed video frame
        await sync.write_audio(_make_stereo_pcm(num_samples=24000, sample_rate=48000))
        await sync.write_video(_make_frame(640, 480, "red"))

        await sync.flush()

        # Wait past when the frame would have been released
        await asyncio.sleep(0.55)

        # recv should return the blue empty frame, not the red one
        received = await sync.video_track.recv()
        arr = received.to_ndarray(format="rgb24")
        assert np.mean(arr[:, :, 2]) > 200, "Should be the blue empty frame after flush"
        assert np.mean(arr[:, :, 0]) < 50, "Red frame should have been discarded"

    async def test_write_video_after_stop_is_ignored(self, sync: AVSynchronizer):
        sync.video_track.stop()
        # Should not raise
        await sync.write_video(_make_frame(640, 480))

    async def test_recv_after_stop_raises(self, sync: AVSynchronizer):
        sync.video_track.stop()
        with pytest.raises(VideoTrackClosedError):
            await sync.video_track.recv()
