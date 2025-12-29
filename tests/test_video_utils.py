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

    def test_odd_width_cropped(self):
        """Frame with odd width should be cropped by 1 pixel."""
        frame = self._create_frame(101, 100)
        result = ensure_even_dimensions(frame)

        assert result.width == 100
        assert result.height == 100
        assert result is not frame

    def test_odd_height_cropped(self):
        """Frame with odd height should be cropped by 1 pixel."""
        frame = self._create_frame(100, 101)
        result = ensure_even_dimensions(frame)

        assert result.width == 100
        assert result.height == 100
        assert result is not frame

    def test_both_odd_cropped(self):
        """Frame with both odd dimensions should be cropped."""
        frame = self._create_frame(101, 101)
        result = ensure_even_dimensions(frame)

        assert result.width == 100
        assert result.height == 100
        assert result is not frame

    def test_crops_not_rescales(self):
        """Verify the operation crops (removes edge pixels) rather than rescales."""
        frame = self._create_frame(101, 101)
        original_arr = frame.to_ndarray(format="rgb24")

        result = ensure_even_dimensions(frame)
        result_arr = result.to_ndarray(format="rgb24")

        # If cropped correctly, the top-left 100x100 pixels should be identical
        np.testing.assert_array_equal(
            result_arr,
            original_arr[:100, :100],
            err_msg="Frame was rescaled instead of cropped",
        )

    def test_preserves_pts(self):
        """PTS should be preserved."""
        frame = self._create_frame(101, 100)
        result = ensure_even_dimensions(frame)

        assert result.pts == 12345

    def test_preserves_time_base(self):
        """Time base should be preserved."""
        frame = self._create_frame(101, 100)
        result = ensure_even_dimensions(frame)

        assert result.time_base == Fraction(1, 30)

    def test_realistic_screen_share_dimensions(self):
        """Test with realistic odd screen share dimension (1728x1083)."""
        frame = self._create_frame(1728, 1083)
        result = ensure_even_dimensions(frame)

        assert result.width == 1728  # Already even
        assert result.height == 1082  # Cropped by 1
