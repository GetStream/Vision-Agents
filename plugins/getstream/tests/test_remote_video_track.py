from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError
from vision_agents.plugins.getstream.remote_video_track import RemoteVideoTrack


@dataclass
class _I420Frame:
    """Plain frame with the attributes that RemoteVideoTrack reads."""

    width: int
    height: int
    data: np.ndarray
    rtp_timestamp: int


@pytest.fixture
def odd_frame() -> _I420Frame:
    # 5x3: Y is 5x3, U and V are 3x2 each.
    return _I420Frame(
        width=5,
        height=3,
        data=np.arange(5 * 3 + 2 * 3 * 2, dtype=np.uint8),
        rtp_timestamp=90_000,
    )


@pytest.fixture
def even_frame() -> _I420Frame:
    return _I420Frame(
        width=4,
        height=2,
        data=np.arange(4 * 2 + 2 * 2 * 1, dtype=np.uint8),
        rtp_timestamp=180_000,
    )


@pytest.fixture
def odd_track(odd_frame: _I420Frame) -> RemoteVideoTrack:
    async def frames():
        yield odd_frame

    return RemoteVideoTrack(frames())


@pytest.fixture
def even_track(even_frame: _I420Frame) -> RemoteVideoTrack:
    async def frames():
        yield even_frame

    return RemoteVideoTrack(frames())


class TestRemoteVideoTrack:
    async def test_odd_size_frame_keeps_every_plane_row(
        self, odd_track: RemoteVideoTrack, odd_frame: _I420Frame
    ):
        frame = await odd_track.recv()

        assert frame.format.name == "yuv420p"
        assert (frame.width, frame.height) == (5, 3)
        planes = [
            np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)[
                :, : plane.width
            ]
            for plane in frame.planes
        ]
        assert planes[0].tolist() == odd_frame.data[:15].reshape(3, 5).tolist()
        assert planes[1].tolist() == odd_frame.data[15:21].reshape(2, 3).tolist()
        assert planes[2].tolist() == odd_frame.data[21:27].reshape(2, 3).tolist()

    async def test_even_size_frame_keeps_every_plane_row(
        self, even_track: RemoteVideoTrack, even_frame: _I420Frame
    ):
        frame = await even_track.recv()

        assert (frame.width, frame.height) == (4, 2)
        assert frame.to_ndarray().reshape(-1).tolist() == even_frame.data.tolist()

    async def test_frame_timing_uses_rtp_clock(
        self, odd_track: RemoteVideoTrack, odd_frame: _I420Frame
    ):
        frame = await odd_track.recv()

        assert frame.pts == odd_frame.rtp_timestamp
        assert frame.time_base == Fraction(1, 90000)

    async def test_end_of_frames_ends_track(self, odd_track: RemoteVideoTrack):
        await odd_track.recv()

        with pytest.raises(MediaStreamError):
            await odd_track.recv()
        assert odd_track.readyState == "ended"

    async def test_stopped_track_gives_no_frames(self, odd_track: RemoteVideoTrack):
        odd_track.stop()

        with pytest.raises(MediaStreamError):
            await odd_track.recv()
        assert odd_track.readyState == "ended"
