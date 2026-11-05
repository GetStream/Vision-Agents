from fractions import Fraction
from typing import AsyncIterator, Optional

import aiortc
import av
import numpy as np
from aiortc.mediastreams import MediaStreamError
from getstream.video.rtc import VideoFrame

_VIDEO_TIME_BASE = Fraction(1, 90000)


class RemoteVideoTrack(aiortc.VideoStreamTrack):
    """aiortc video track that reads the decoded frames of a remote Stream track."""

    def __init__(self, frames: AsyncIterator[VideoFrame]):
        super().__init__()
        self._frames: Optional[AsyncIterator[VideoFrame]] = frames

    async def recv(self) -> av.VideoFrame:
        if self._frames is None:
            raise MediaStreamError
        try:
            frame = await anext(self._frames)
        except StopAsyncIteration:
            self.stop()
            raise MediaStreamError
        video_frame = av.VideoFrame(frame.width, frame.height, "yuv420p")
        # frame.data is packed I420 without row padding; PyAV planes are padded
        # to line_size, and chroma planes are ceil(w/2) x ceil(h/2).
        offset = 0
        for plane in video_frame.planes:
            size = plane.width * plane.height
            rows = np.frombuffer(plane, dtype=np.uint8).reshape(
                plane.height, plane.line_size
            )
            rows[:, : plane.width] = frame.data[offset : offset + size].reshape(
                plane.height, plane.width
            )
            offset += size
        video_frame.pts = frame.rtp_timestamp
        video_frame.time_base = _VIDEO_TIME_BASE
        return video_frame

    def stop(self) -> None:
        super().stop()
        # Dropping the stream lets the SDK stop decoding for this reader.
        self._frames = None
