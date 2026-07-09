"""
Tests for the TwelveLabs PegasusVLM plugin.

Integration tests require the TWELVELABS_API_KEY environment variable:

    export TWELVELABS_API_KEY="your-key-here"
    uv run pytest plugins/twelvelabs/tests/test_pegasus_vlm.py -m integration -v

To run only unit tests (no API key needed):

    uv run pytest plugins/twelvelabs/tests/test_pegasus_vlm.py -m "not integration" -v
"""

import io
import os
from fractions import Fraction

import av
import numpy as np
import pytest

from vision_agents.core.llm.llm import LLMResponseFinal
from vision_agents.plugins.twelvelabs import PegasusVLM


def _solid_frame(width: int, height: int, pts: int = 0) -> av.VideoFrame:
    """Build a solid-color RGB frame of the given size."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
    frame.pts = pts
    frame.time_base = Fraction(1, 1)
    return frame


class TestPegasusVLM:
    """Unit tests that do not touch the network."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
        with pytest.raises(ValueError):
            PegasusVLM()

    def test_rejects_short_clip(self):
        with pytest.raises(ValueError):
            PegasusVLM(api_key="x", clip_seconds=3)

    def test_rejects_small_max_tokens(self):
        with pytest.raises(ValueError):
            PegasusVLM(api_key="x", max_tokens=128)

    def test_buffer_size_tracks_fps_and_duration(self):
        vlm = PegasusVLM(api_key="x", fps=2.0, clip_seconds=5)
        assert vlm._frame_buffer.maxlen == 10

    def test_encode_clip_upscales_to_minimum_and_is_decodable(self):
        """A tiny 64x64 frame must be encoded into a decodable >=360x360 MP4."""
        vlm = PegasusVLM(api_key="x", fps=1.0, clip_seconds=4)
        frames = [_solid_frame(64, 64, pts=0), _solid_frame(64, 64, pts=1)]

        clip = vlm._encode_clip(frames)

        assert isinstance(clip, bytes) and len(clip) > 0
        container = av.open(io.BytesIO(clip))
        try:
            stream = container.streams.video[0]
            assert stream.codec_context.width >= 360
            assert stream.codec_context.height >= 360
            decoded = [f for f in container.decode(video=0)]
            assert len(decoded) >= 1
            assert stream.duration is not None
            duration_s = float(stream.duration * stream.time_base)
            assert duration_s >= 4.0
        finally:
            container.close()


@pytest.mark.integration
class TestPegasusVLMIntegration:
    async def test_analyzes_buffered_clip(self):
        """Buffer a few real frames and run a live Pegasus analysis.

        This is an opt-in integration test: it is only collected when the
        ``integration`` marker is selected. If ``TWELVELABS_API_KEY`` is missing
        ``PegasusVLM()`` raises a clear, actionable error rather than skipping.
        """
        assert os.getenv("TWELVELABS_API_KEY"), (
            "TWELVELABS_API_KEY must be set to run the Pegasus integration tests"
        )
        vlm = PegasusVLM(fps=1.0, clip_seconds=5)
        try:
            for _ in range(5):
                await vlm._on_frame_received(_solid_frame(640, 360))

            items = [
                item async for item in vlm.simple_response("Describe this short video.")
            ]
            final = next(item for item in items if isinstance(item, LLMResponseFinal))
            assert len(final.text.strip()) > 0
        finally:
            await vlm.close()
