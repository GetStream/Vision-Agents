"""Tests for video_utils module."""

import av
import numpy as np
from fractions import Fraction

from vision_agents.core.utils.video_utils import ensure_even_dimensions, resize_frame


class TestEnsureEvenDimensions:
    """Tests for ensure_even_dimensions function."""

    def _create_frame(self, width: int, height: int) -> av.VideoFrame:
        """Create a test frame with given dimensions filled with a gradient."""
        # Create a gradient pattern so we can verify cropping vs rescaling
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = np.arange(width, dtype=np.uint8) % 256  # Red gradient horizontal
        arr[:, :, 1] = (
            np.arange(height, dtype=np.uint8).reshape(-1, 1) % 256
        )  # Green gradient vertical
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = 12345
        frame.time_base = Fraction(1, 30)
        return frame

    def test_even_dimensions_unchanged(self):
        """Frame with even dimensions should pass through unchanged."""
        frame = self._create_frame(100, 100)
        result = ensure_even_dimensions(frame)

        assert result.width == 100
        assert result.height == 100
        assert result is frame  # Should be same object

    def test_both_odd_cropped(self):
        """Frame with both odd dimensions should be cropped."""
        frame = self._create_frame(101, 101)
        result = ensure_even_dimensions(frame)

        assert result.width == 100
        assert result.height == 100
        assert result is not frame

    def test_preserves_properties(self):
        """PTS and time base should be preserved."""
        frame = self._create_frame(101, 100)
        result = ensure_even_dimensions(frame)

        assert result.pts == 12345

        assert result.time_base == Fraction(1, 30)

    def test_realistic_screen_share_dimensions(self):
        """Test with realistic odd screen share dimension (1728x1083)."""
        frame = self._create_frame(1728, 1083)
        result = ensure_even_dimensions(frame)

        assert result.width == 1728  # Already even
        assert result.height == 1082  # Cropped by 1


def _solid_rgb_frame(
    width: int, height: int, color: tuple[int, int, int]
) -> av.VideoFrame:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return av.VideoFrame.from_ndarray(arr, format="rgb24")


class TestResizeFrame:
    def test_resizes_down_to_target_size(self):
        src = _solid_rgb_frame(1280, 720, (200, 50, 50))

        out = resize_frame(src, 640, 480)

        assert out.width == 640
        assert out.height == 480

    def test_aspect_preserved_letterboxes_top_and_bottom(self):
        # 2:1 source into 4:3 target: inner box 640x320, vertical bars at top/bottom.
        src = _solid_rgb_frame(200, 100, (50, 200, 50))

        out = resize_frame(src, 640, 480)
        arr = out.to_ndarray(format="rgb24")

        assert out.width == 640
        assert out.height == 480
        assert np.array_equal(arr[0, 320, :], np.array([0, 0, 0], dtype=np.uint8))
        assert np.array_equal(arr[479, 320, :], np.array([0, 0, 0], dtype=np.uint8))
        # Center of frame is inside the inner box; reformat dithers, so allow tolerance.
        center = arr[240, 320, :].astype(int)
        assert center[1] > center[0] and center[1] > center[2]

    def test_aspect_preserved_letterboxes_left_and_right(self):
        # 1:2 source into 4:3 target: inner box 240x480, bars on left/right.
        src = _solid_rgb_frame(100, 200, (50, 50, 200))

        out = resize_frame(src, 640, 480)
        arr = out.to_ndarray(format="rgb24")

        assert out.width == 640
        assert out.height == 480
        assert np.array_equal(arr[240, 0, :], np.array([0, 0, 0], dtype=np.uint8))
        assert np.array_equal(arr[240, 639, :], np.array([0, 0, 0], dtype=np.uint8))
        center = arr[240, 320, :].astype(int)
        assert center[2] > center[0] and center[2] > center[1]

    def test_exact_size_returns_same_dimensions(self):
        src = _solid_rgb_frame(640, 480, (100, 100, 100))

        out = resize_frame(src, 640, 480)

        assert out.width == 640
        assert out.height == 480
