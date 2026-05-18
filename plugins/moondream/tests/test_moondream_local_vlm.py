"""
Tests for the Moondream LocalVLM plugin.

Integration tests require HF_TOKEN environment variable (for gated model access):

    export HF_TOKEN="your-token-here"
    uv run pytest plugins/moondream/tests/test_moondream_local_vlm.py -m integration -v
"""

import os
from pathlib import Path
from typing import Iterator

import av
import pytest
from PIL import Image
from vision_agents.core.llm.llm import LLMResponseFinal
from vision_agents.plugins.moondream import LocalVLM


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
async def local_vlm_vqa() -> LocalVLM:
    """Create LocalVLM in VQA mode."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set")

    vlm = LocalVLM(mode="vqa")
    try:
        await vlm.warmup()
        yield vlm
    finally:
        vlm.close()


@pytest.fixture
async def local_vlm_caption() -> LocalVLM:
    """Create LocalVLM in caption mode."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        pytest.skip("HF_TOKEN not set")

    vlm = LocalVLM(mode="caption")
    try:
        await vlm.warmup()
        yield vlm
    finally:
        vlm.close()


@pytest.mark.integration
@pytest.mark.skip
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN not set")
class TestMoondreamLocalVLMIntegration:
    async def test_local_vqa_mode(
        self, golf_frame: av.VideoFrame, local_vlm_vqa: LocalVLM
    ):
        """Test LocalVLM VQA mode with a question about the image."""
        local_vlm_vqa._latest_frame = golf_frame

        question = "What sport is being played in this image?"
        items = [item async for item in local_vlm_vqa.simple_response(question)]
        final = items[-1]

        assert isinstance(final, LLMResponseFinal)
        assert len(final.text) > 0
        assert "golf" in final.text.lower()

    async def test_local_caption_mode(
        self, golf_frame: av.VideoFrame, local_vlm_caption: LocalVLM
    ):
        """Test LocalVLM caption mode to generate a description of the image."""
        local_vlm_caption._latest_frame = golf_frame

        items = [item async for item in local_vlm_caption.simple_response("")]
        final = items[-1]

        assert isinstance(final, LLMResponseFinal)
        assert len(final.text.strip()) > 0
