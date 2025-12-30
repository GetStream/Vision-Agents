"""Tests for video_utils module."""

import av
import numpy as np
from fractions import Fraction

from vision_agents.core.utils.video_utils import ensure_even_dimensions


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
