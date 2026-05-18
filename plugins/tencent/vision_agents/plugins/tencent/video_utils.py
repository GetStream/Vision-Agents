"""Video frame conversion utilities for the Tencent TRTC edge transport."""

import av
import numpy as np


def yuv420p_to_av_frame(yuv_bytes: bytes, width: int, height: int) -> av.VideoFrame:
    """Convert raw YUV420p bytes into an av.VideoFrame.

    Requires even ``width`` and ``height`` because YUV420p subsamples
    chroma 2x2. A short ``yuv_bytes`` buffer is zero-padded; a longer
    one (e.g. when the SDK adds stride past ``width``) is truncated so
    the reshape always succeeds.
    """
    if width % 2 or height % 2:
        raise ValueError(f"YUV420p requires even dimensions, got {width}x{height}")
    expected = width * height * 3 // 2
    if len(yuv_bytes) < expected:
        yuv_bytes = yuv_bytes + b"\x00" * (expected - len(yuv_bytes))
    elif len(yuv_bytes) > expected:
        yuv_bytes = yuv_bytes[:expected]
    arr = np.frombuffer(yuv_bytes, dtype=np.uint8).reshape((height * 3 // 2, width))
    return av.VideoFrame.from_ndarray(arr, format="yuv420p")


def av_frame_to_yuv420p(frame: av.VideoFrame) -> tuple[bytes, int, int]:
    """Convert an av.VideoFrame to YUV420p bytes, width, height."""
    yuv = frame.reformat(format="yuv420p")
    arr = yuv.to_ndarray(format="yuv420p")
    return arr.tobytes(), yuv.width, yuv.height
