"""
Tests for the Moondream CloudVLM plugin.

Integration tests require MOONDREAM_API_KEY environment variable:

    export MOONDREAM_API_KEY="your-key-here"
    uv run pytest plugins/moondream/tests/test_moondream_vlm.py -m integration -v

To run only unit tests (no API key needed):

    uv run pytest plugins/moondream/tests/test_moondream_vlm.py -m "not integration" -v
"""

import os
from pathlib import Path
from typing import Iterator

import pytest
import av
from PIL import Image

from vision_agents.core.llm.llm import LLMResponseFinal
from vision_agents.plugins.moondream import CloudVLM


@pytest.fixture(scope="session")
def golf_image(assets_dir) -> Iterator[Image.Image]:
    """Load the local golf swing test image from tests/test_assets."""
    asset_path = Path(assets_dir) / "golf_swing.png"
    with Image.open(asset_path) as img:
        yield img.convert("RGB")


@pytest.fixture
def golf_frame(golf_image: Image.Image) -> av.VideoFrame:
    """Create an av.VideoFrame from the golf image."""
    return av.VideoFrame.from_image(golf_image)


@pytest.fixture
async def vlm_vqa() -> CloudVLM:
    """Create CloudVLM in VQA mode."""
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        pytest.skip("MOONDREAM_API_KEY not set")

    vlm = CloudVLM(api_key=api_key, mode="vqa")
    try:
        yield vlm
    finally:
        vlm.close()


@pytest.fixture
async def vlm_caption() -> CloudVLM:
    """Create CloudVLM in caption mode."""
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        pytest.skip("MOONDREAM_API_KEY not set")

    vlm = CloudVLM(api_key=api_key, mode="caption")
    try:
        yield vlm
    finally:
        vlm.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
class TestMoondreamCloudVLMIntegration:
    async def test_vqa_mode(self, golf_frame: av.VideoFrame, vlm_vqa: CloudVLM):
        """Test VQA mode with a question about the image."""
        vlm_vqa._latest_frame = golf_frame

        question = "What sport is being played in this image?"
        items = [item async for item in vlm_vqa.simple_response(question)]
        final = next(item for item in items if isinstance(item, LLMResponseFinal))

        assert len(final.text) > 0
        assert "golf" in final.text.lower()

    async def test_caption_mode(self, golf_frame: av.VideoFrame, vlm_caption: CloudVLM):
        """Test caption mode to generate a description of the image."""
        vlm_caption._latest_frame = golf_frame

        items = [item async for item in vlm_caption.simple_response("")]
        final = next(item for item in items if isinstance(item, LLMResponseFinal))

        assert len(final.text.strip()) > 0
