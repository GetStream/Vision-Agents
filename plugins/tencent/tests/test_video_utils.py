"""Unit tests for the YUV420p ↔ av.VideoFrame helpers."""

import av
import numpy as np
import pytest

from vision_agents.plugins.tencent.video_utils import (
    av_frame_to_yuv420p,
    yuv420p_to_av_frame,
)


class TestYuv420pToAvFrame:
    def test_rejects_odd_width(self) -> None:
        with pytest.raises(ValueError, match="even dimensions"):
            yuv420p_to_av_frame(b"\x00" * 100, width=15, height=16)

    def test_rejects_odd_height(self) -> None:
        with pytest.raises(ValueError, match="even dimensions"):
            yuv420p_to_av_frame(b"\x00" * 100, width=16, height=15)

    def test_pads_short_buffer_to_expected_size(self) -> None:
        # 4x4 YUV420p needs 4*4*3/2 = 24 bytes; we hand it only 10.
        frame = yuv420p_to_av_frame(b"\xff" * 10, width=4, height=4)
        assert frame.width == 4
        assert frame.height == 4
        assert frame.format.name == "yuv420p"

    def test_truncates_oversized_buffer(self) -> None:
        # 4x4 YUV420p needs 24 bytes; we hand it 200 (e.g. SDK row stride).
        # Without truncation, np.frombuffer().reshape(6, 4) would raise.
        frame = yuv420p_to_av_frame(b"\xab" * 200, width=4, height=4)
        assert frame.width == 4
        assert frame.height == 4

    def test_roundtrip_through_av_frame(self) -> None:
        # 16x16 YUV420p: 16*16*3//2 = 384 bytes. Fill Y, U, V with distinct
        # byte patterns so we'd notice any plane mix-up after a roundtrip.
        width, height = 16, 16
        y_size = width * height
        uv_size = width * height // 4
        original = (
            bytes([0x11]) * y_size + bytes([0x22]) * uv_size + bytes([0x33]) * uv_size
        )
        frame = yuv420p_to_av_frame(original, width=width, height=height)
        out, w, h = av_frame_to_yuv420p(frame)
        assert (w, h) == (width, height)
        assert out == original


class TestAvFrameToYuv420p:
    def test_returns_native_dimensions(self) -> None:
        # Build a real av.VideoFrame at a non-square size so we can tell the
        # helper from a no-op.
        arr = np.zeros((24, 32, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        _, w, h = av_frame_to_yuv420p(frame)
        assert (w, h) == (32, 24)
