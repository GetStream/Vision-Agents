"""Tests for temporal frame encoding functions."""

import av
import numpy as np
import pytest
from PIL import Image

from vision_agents.core.utils.video_utils import (
    frames_to_motion_heatmap,
    frames_to_rgb_temporal_composite,
    frames_to_temporal_grid,
    temporal_composite_to_jpeg_bytes,
)

TARGET_W = 200
TARGET_H = 150


def _make_frame(arr: np.ndarray) -> av.VideoFrame:
    """Create an av.VideoFrame from an RGB uint8 numpy array."""
    return av.VideoFrame.from_ndarray(arr, format="rgb24")


def _solid_frame(r: int, g: int, b: int, w: int = TARGET_W, h: int = TARGET_H) -> av.VideoFrame:
    """Create a uniformly-colored frame."""
    arr = np.full((h, w, 3), [r, g, b], dtype=np.uint8)
    return _make_frame(arr)


def _frame_with_block(
    x: int, y: int, size: int = 30, w: int = TARGET_W, h: int = TARGET_H
) -> av.VideoFrame:
    """Create a black frame with a white block at (x, y)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[y : y + size, x : x + size] = 255
    return _make_frame(arr)


class TestRGBTemporalComposite:
    def test_rejects_fewer_than_3_frames(self):
        frames = [_solid_frame(128, 128, 128), _solid_frame(128, 128, 128)]
        with pytest.raises(ValueError, match="requires >= 3 frames"):
            frames_to_rgb_temporal_composite(frames, TARGET_W, TARGET_H)

    def test_static_scene_produces_gray(self):
        frames = [_solid_frame(200, 200, 200) for _ in range(6)]
        result = frames_to_rgb_temporal_composite(frames, TARGET_W, TARGET_H)
        arr = np.array(result)

        assert arr.shape == (TARGET_H, TARGET_W, 3)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        assert np.allclose(r, g, atol=2)
        assert np.allclose(g, b, atol=2)

    def test_moving_object_shows_color_separation(self):
        frames = [
            _frame_with_block(x=10, y=60),
            _frame_with_block(x=30, y=60),
            _frame_with_block(x=50, y=60),
            _frame_with_block(x=70, y=60),
            _frame_with_block(x=90, y=60),
            _frame_with_block(x=110, y=60),
        ]
        result = frames_to_rgb_temporal_composite(frames, TARGET_W, TARGET_H)
        arr = np.array(result)

        left_region = arr[60:90, 10:40, :]
        right_region = arr[60:90, 110:140, :]

        left_r_dominance = left_region[:, :, 0].mean() - left_region[:, :, 2].mean()
        assert left_r_dominance > 30, "Left region should be red-dominant (early position)"

        right_b_dominance = right_region[:, :, 2].mean() - right_region[:, :, 0].mean()
        assert right_b_dominance > 30, "Right region should be blue-dominant (late position)"

    def test_output_dimensions_match_target(self):
        frames = [_solid_frame(100, 100, 100) for _ in range(3)]
        result = frames_to_rgb_temporal_composite(frames, 320, 240)
        assert result.size == (320, 240)

    def test_three_frames_maps_one_per_channel(self):
        frames = [
            _solid_frame(255, 255, 255),
            _solid_frame(128, 128, 128),
            _solid_frame(0, 0, 0),
        ]
        result = frames_to_rgb_temporal_composite(frames, TARGET_W, TARGET_H)
        arr = np.array(result)

        center_pixel = arr[TARGET_H // 2, TARGET_W // 2]
        r, g, b = int(center_pixel[0]), int(center_pixel[1]), int(center_pixel[2])
        assert r > g > b, f"Expected descending R>G>B, got R={r} G={g} B={b}"

    def test_handles_non_divisible_by_3_frame_count(self):
        for count in [4, 5, 7, 10, 11]:
            frames = [_solid_frame(128, 128, 128) for _ in range(count)]
            result = frames_to_rgb_temporal_composite(frames, TARGET_W, TARGET_H)
            assert result.size == (TARGET_W, TARGET_H)


class TestMotionHeatmap:
    def test_rejects_fewer_than_2_frames(self):
        with pytest.raises(ValueError, match="requires >= 2 frames"):
            frames_to_motion_heatmap([_solid_frame(0, 0, 0)], TARGET_W, TARGET_H)

    def test_static_scene_has_minimal_heatmap(self):
        frames = [_solid_frame(100, 100, 100) for _ in range(5)]
        result = frames_to_motion_heatmap(frames, TARGET_W, TARGET_H)
        arr = np.array(result)
        assert arr.shape == (TARGET_H, TARGET_W, 3)

        expected_base = np.full((TARGET_H, TARGET_W), 100, dtype=np.uint8)
        for ch in range(3):
            assert np.allclose(arr[:, :, ch], expected_base, atol=5)

    def test_moving_object_produces_heatmap_on_path(self):
        frames = [
            _frame_with_block(x=20, y=60),
            _frame_with_block(x=40, y=60),
            _frame_with_block(x=60, y=60),
            _frame_with_block(x=80, y=60),
        ]
        result = frames_to_motion_heatmap(frames, TARGET_W, TARGET_H)
        arr = np.array(result)

        motion_region = arr[60:90, 20:110, :]
        static_region = arr[0:30, 0:30, :]

        motion_intensity = motion_region.astype(float).mean()
        static_intensity = static_region.astype(float).mean()
        assert motion_intensity > static_intensity + 10

    def test_output_is_rgb(self):
        frames = [_solid_frame(50, 50, 50) for _ in range(3)]
        result = frames_to_motion_heatmap(frames, TARGET_W, TARGET_H)
        assert result.mode == "RGB"


class TestTemporalGrid:
    def test_rejects_empty_frames(self):
        with pytest.raises(ValueError, match="At least one frame"):
            frames_to_temporal_grid([], TARGET_W, TARGET_H)

    def test_single_frame_grid(self):
        result = frames_to_temporal_grid(
            [_solid_frame(200, 200, 200)], TARGET_W, TARGET_H
        )
        assert result.size == (TARGET_W, TARGET_H)

    def test_four_frames_2x2_grid(self):
        frames = [
            _solid_frame(255, 0, 0),
            _solid_frame(0, 255, 0),
            _solid_frame(0, 0, 255),
            _solid_frame(255, 255, 0),
        ]
        result = frames_to_temporal_grid(frames, TARGET_W, TARGET_H, cols=2)
        arr = np.array(result)

        cell_w = TARGET_W // 2
        cell_h = TARGET_H // 2

        top_left = arr[cell_h // 4, cell_w // 4]
        assert top_left[0] > 200 and top_left[1] < 50, "Top-left should be red"

        top_right = arr[cell_h // 4, cell_w + cell_w // 4]
        assert top_right[1] > 200 and top_right[0] < 50, "Top-right should be green"

    def test_samples_evenly_from_many_frames(self):
        frames = [_solid_frame(i * 25, 0, 0) for i in range(10)]
        result = frames_to_temporal_grid(frames, TARGET_W, TARGET_H, cols=2)
        assert result.size == (TARGET_W, TARGET_H)

    def test_3_columns(self):
        frames = [_solid_frame(100, 100, 100) for _ in range(6)]
        result = frames_to_temporal_grid(frames, 300, 200, cols=3)
        assert result.size == (300, 200)


class TestTemporalCompositeToJPEG:
    def test_rgb_mode_returns_valid_jpeg(self):
        frames = [_solid_frame(128, 128, 128) for _ in range(3)]
        data = temporal_composite_to_jpeg_bytes(frames, TARGET_W, TARGET_H, mode="rgb")
        assert data[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_heatmap_mode_returns_valid_jpeg(self):
        frames = [_solid_frame(128, 128, 128) for _ in range(3)]
        data = temporal_composite_to_jpeg_bytes(frames, TARGET_W, TARGET_H, mode="heatmap")
        assert data[:2] == b"\xff\xd8"

    def test_grid_mode_returns_valid_jpeg(self):
        frames = [_solid_frame(128, 128, 128) for _ in range(4)]
        data = temporal_composite_to_jpeg_bytes(frames, TARGET_W, TARGET_H, mode="grid")
        assert data[:2] == b"\xff\xd8"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown temporal encoding mode"):
            temporal_composite_to_jpeg_bytes(
                [_solid_frame(0, 0, 0)] * 3, TARGET_W, TARGET_H, mode="bogus"
            )

    def test_rgb_composite_smaller_than_individual_frames(self):
        frames = [_solid_frame(100 + i * 10, 50, 50) for i in range(10)]
        composite_bytes = temporal_composite_to_jpeg_bytes(
            frames, TARGET_W, TARGET_H, mode="rgb"
        )
        individual_total = sum(
            len(
                temporal_composite_to_jpeg_bytes(
                    [f, f, f], TARGET_W, TARGET_H, mode="rgb"
                )
            )
            for f in frames
        )
        assert len(composite_bytes) < individual_total
