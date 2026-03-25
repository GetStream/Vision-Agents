"""Tests for VideoDisplay."""

import pytest

from vision_agents.plugins.local.display import VideoDisplay


class TestVideoDisplay:
    def test_fps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="fps must be > 0"):
            VideoDisplay(fps=0)

    def test_fps_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="fps must be > 0"):
            VideoDisplay(fps=-1)

    def test_width_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="width must be > 0"):
            VideoDisplay(width=0)

    def test_width_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="width must be > 0"):
            VideoDisplay(width=-1)

    def test_height_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="height must be > 0"):
            VideoDisplay(height=0)

    def test_height_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="height must be > 0"):
            VideoDisplay(height=-1)

    def test_valid_defaults(self) -> None:
        display = VideoDisplay()
        assert display._width == 640
        assert display._height == 480
