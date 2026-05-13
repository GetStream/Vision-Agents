"""
Tests for the NVIDIA VLM plugin.

Integration tests require NVIDIA_API_KEY environment variable:

    export NVIDIA_API_KEY="your-key-here"
    uv run pytest plugins/nvidia/tests/test_nvidia_vlm.py -m integration -v
"""

import os
from pathlib import Path
from typing import Iterator

import av
import pytest
from dotenv import load_dotenv
from PIL import Image
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.plugins.nvidia import VLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture(scope="session")
def cat_image(assets_dir) -> Iterator[Image.Image]:
    """Load the local cat test image from tests/test_assets."""
    asset_path = Path(assets_dir) / "cat.jpg"
    with Image.open(asset_path) as img:
        yield img.convert("RGB")


@pytest.fixture
def cat_frame(cat_image: Image.Image) -> av.VideoFrame:
    """Create an av.VideoFrame from the cat image."""
    return av.VideoFrame.from_image(cat_image)


@pytest.fixture
async def vlm() -> VLM:
    """Create NvidiaVLM instance for testing."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set")

    vlm_instance = VLM(model="meta/llama-3.2-11b-vision-instruct")
    vlm_instance.set_conversation(InMemoryConversation("be friendly", []))
    try:
        yield vlm_instance
    finally:
        await vlm_instance.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("NVIDIA_API_KEY"), reason="NVIDIA_API_KEY not set")
class TestNvidiaVLMIntegration:
    """Test suite for NvidiaVLM class."""

    async def test_simple_response(self, vlm: VLM):
        """Test streaming responses yield deltas and a final response."""
        deltas, final = await collect_simple_response(
            vlm.simple_response("Explain quantum computing in 1 paragraph")
        )

        assert final.text
        assert len(deltas) > 0

    async def test_memory(self, vlm: VLM):
        """Test conversation memory across multiple messages."""
        await collect_simple_response(
            vlm.simple_response(text="There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            vlm.simple_response(text="How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_with_video_frames(self, vlm: VLM, cat_frame: av.VideoFrame):
        """Test VLM with buffered video frames."""
        vlm._frame_buffer.append(cat_frame)

        _, final = await collect_simple_response(
            vlm.simple_response("What do you see in this image?")
        )

        assert final.text
        assert "cat" in final.text.lower(), f"Expected 'cat' in response: {final.text}"

    async def test_instruction_following(self, vlm):
        """Test that system instructions are respected."""

        vlm.set_conversation(
            InMemoryConversation("only reply in 2 letter country shortcuts", [])
        )
        vlm.set_instructions("only reply in 2 letter country shortcuts")

        _, final = await collect_simple_response(
            vlm.simple_response(
                text="Which country is rainy, protected from water with dikes and below sea level?",
            )
        )
        assert "nl" in final.text.lower()

    async def test_with_participant(self, vlm: VLM):
        """Test that LLM does not duplicate user messages when participant is provided.

        When a participant is provided, the agent layer is responsible for
        adding user messages to the conversation. The LLM should skip adding
        the message to avoid duplicates.
        """
        test_participant = Participant(
            original=None, user_id="test_user_123", id="test_user_123"
        )
        user_question = "What is 2 + 2?"

        # Simulate what the agent does: add user message before calling LLM
        await vlm._conversation.send_message(
            role="user", user_id="test_user_123", content=user_question
        )

        _, final = await collect_simple_response(
            vlm.simple_response(text=user_question, participant=test_participant)
        )

        assert final.text
        assert len(final.text) > 0

        # Verify no duplicate user message was added by the LLM
        user_messages = [
            msg
            for msg in vlm._conversation.messages
            if msg.role == "user" and msg.content == user_question
        ]
        assert len(user_messages) == 1, (
            f"Expected 1 user message, got {len(user_messages)} (LLM should not duplicate)"
        )
