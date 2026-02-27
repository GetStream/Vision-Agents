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
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.nvidia import VLM, events

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

    vlm_instance = VLM(model="nvidia/cosmos-reason2-8b")
    vlm_instance.set_conversation(InMemoryConversation("be friendly", []))
    try:
        yield vlm_instance
    finally:
        await vlm_instance.close()


@pytest.mark.integration
@pytest.mark.skip('The "nvidia/cosmos-reason2-8b" model is not available')
@pytest.mark.skipif(not os.getenv("NVIDIA_API_KEY"), reason="NVIDIA_API_KEY not set")
class TestNvidiaVLMIntegration:
    """Test suite for NvidiaVLM class."""

    async def test_simple(self, vlm: VLM):
        """Test basic text-only response."""
        response = await vlm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        assert response.text
        assert len(response.text) > 0

    async def test_streaming(self, vlm: VLM):
        """Test streaming responses emit chunk and completion events."""
        streaming_works = False

        @vlm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        response = await vlm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        await vlm.events.wait()

        assert response.text
        assert streaming_works

    async def test_memory(self, vlm: VLM):
        """Test conversation memory across multiple messages."""
        await vlm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await vlm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    async def test_events(self, vlm: VLM):
        """Test that LLM events are properly emitted during streaming responses."""
        chunk_events = []
        complete_events = []
        nvidia_stream_events = []
        error_events = []

        @vlm.events.subscribe
        async def handle_chunk_event(event: LLMResponseChunkEvent):
            chunk_events.append(event)

        @vlm.events.subscribe
        async def handle_complete_event(event: LLMResponseCompletedEvent):
            complete_events.append(event)

        @vlm.events.subscribe
        async def handle_nvidia_stream_event(event: events.NvidiaStreamEvent):
            nvidia_stream_events.append(event)

        @vlm.events.subscribe
        async def handle_error_event(event: events.LLMErrorEvent):
            error_events.append(event)

        response = await vlm.simple_response(
            "Create a small story about the weather in the Netherlands. Make it at least 2 paragraphs long.",
        )

        await vlm.events.wait()

        assert response.text, "Response should have text content"
        assert len(response.text) > 50, "Response should be substantial"

        assert len(chunk_events) > 0, (
            "Should have received chunk events during streaming"
        )

        assert len(complete_events) > 0, "Should have received completion event"
        assert len(complete_events) == 1, "Should have exactly one completion event"

        assert len(error_events) == 0, (
            f"Should not have error events, but got: {error_events}"
        )

        total_delta_text = ""
        chunk_item_ids = set()
        for chunk_event in chunk_events:
            assert chunk_event.delta is not None, (
                "Chunk events should have delta content"
            )
            assert isinstance(chunk_event.delta, str), "Delta should be a string"
            assert chunk_event.item_id is not None, (
                "Chunk events should have non-null item_id"
            )
            assert chunk_event.item_id != "", (
                "Chunk events should have non-empty item_id"
            )
            chunk_item_ids.add(chunk_event.item_id)
            total_delta_text += chunk_event.delta

        complete_event = complete_events[0]
        assert complete_event.text == response.text, (
            "Completion event text should match response text"
        )
        assert complete_event.original is not None, (
            "Completion event should have original response"
        )
        assert complete_event.item_id is not None, (
            "Completion event should have non-null item_id"
        )
        assert complete_event.item_id != "", (
            "Completion event should have non-empty item_id"
        )

        assert complete_event.item_id in chunk_item_ids, (
            f"Completion event item_id '{complete_event.item_id}' should match one of the chunk event item_ids: {chunk_item_ids}"
        )

        assert len(total_delta_text) > 0, "Should have accumulated delta text"
        assert len(total_delta_text) >= len(response.text) * 0.8, (
            "Delta text should be substantial portion of final text"
        )

    async def test_with_video_frames(self, vlm: VLM, cat_frame: av.VideoFrame):
        """Test VLM with buffered video frames."""
        vlm._frame_buffer.append(cat_frame)

        response = await vlm.simple_response(
            "What do you see in this image?",
        )

        assert response.text
        assert len(response.text) > 0
        assert "cat" in response.text.lower() or len(response.text.strip()) > 0

    async def test_instruction_following(self):
        """Test that system instructions are respected."""
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            pytest.skip("NVIDIA_API_KEY not set")

        vlm_instance = VLM(model="nvidia/cosmos-reason2-8b")
        vlm_instance.set_conversation(
            InMemoryConversation("only reply in 2 letter country shortcuts", [])
        )
        vlm_instance.set_instructions("only reply in 2 letter country shortcuts")

        try:
            response = await vlm_instance.simple_response(
                text="Which country is rainy, protected from water with dikes and below sea level?",
            )
            assert "nl" in response.text.lower()
        finally:
            await vlm_instance.close()

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

        response = await vlm.simple_response(
            text=user_question,
            participant=test_participant,
        )

        assert response.text
        assert len(response.text) > 0

        # Verify no duplicate user message was added by the LLM
        user_messages = [
            msg
            for msg in vlm._conversation.messages
            if msg.role == "user" and msg.content == user_question
        ]
        assert len(user_messages) == 1, (
            f"Expected 1 user message, got {len(user_messages)} (LLM should not duplicate)"
        )
