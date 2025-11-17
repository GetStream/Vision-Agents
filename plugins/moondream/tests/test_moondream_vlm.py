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
async def test_vqa_mode(golf_frame: av.VideoFrame, vlm_vqa: CloudVLM):
    """Test VQA mode with a question about the image."""
    # Set the latest frame so _process_frame can access it
    vlm_vqa._latest_frame = golf_frame

    # Ask a question about the image
    question = "What sport is being played in this image?"
    response = await vlm_vqa.simple_response(question)

    # Verify we got a response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
    assert response.exception is None

    # Verify the response mentions golf (should be in the image)
    assert "golf" in response.text.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("MOONDREAM_API_KEY"), reason="MOONDREAM_API_KEY not set"
)
async def test_caption_mode(golf_frame: av.VideoFrame, vlm_caption: CloudVLM):
    """Test caption mode to generate a description of the image."""
    # Set the latest frame so _process_frame can access it
    vlm_caption._latest_frame = golf_frame

    # Generate caption (text is not needed for caption mode)
    response = await vlm_caption.simple_response("")

    # Verify we got a response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
    assert response.exception is None

    # Verify the caption is descriptive (not empty)
    assert len(response.text.strip()) > 0
