import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import Message
from vision_agents.plugins.openai.openai_llm import OpenAILLM
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.openai import events

load_dotenv()


class TestOpenAILLM:
    """Test suite for OpenAILLM class with mocked API calls."""

    def test_message(self):
        messages = OpenAILLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

        advanced = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what do you see in this image?"},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ]
        messages2 = OpenAILLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.fixture
    async def llm(self) -> OpenAILLM:
        llm = OpenAILLM(model="gpt-4o")
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: OpenAILLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: OpenAILLM):
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )

        # Assertions
        assert response.text
        assert hasattr(response.original, "id")  # OpenAI response has id

    @pytest.mark.integration
    async def test_streaming(self, llm: OpenAILLM):
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        await llm.events.wait()

        assert response.text
        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: OpenAILLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: OpenAILLM):
        await llm.create_response(
            input="There are 2 dogs in the room",
        )
        response = await llm.create_response(
            input="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_events(self, llm: OpenAILLM):
        """Test that LLM events are properly emitted during streaming responses."""
        # Track events and their content
        chunk_events = []
        complete_events = []
        openai_stream_events = []
        error_events = []

        # Register event handlers BEFORE making the API call
        @llm.events.subscribe
        async def handle_chunk_event(event: LLMResponseChunkEvent):
            chunk_events.append(event)

        @llm.events.subscribe
        async def handle_complete_event(event: LLMResponseCompletedEvent):
            complete_events.append(event)

        @llm.events.subscribe
        async def handle_openai_stream_event(event: events.OpenAIStreamEvent):
            openai_stream_events.append(event)

        @llm.events.subscribe
        async def handle_error_event(event: events.LLMErrorEvent):
            error_events.append(event)

        # Make API call that should generate streaming events
        response = await llm.create_response(
            input="Create a small story about the weather in the Netherlands. Make it at least 2 paragraphs long.",
        )

        # Wait for all events to be processed
        await llm.events.wait()

        # Verify response was generated
        assert response.text, "Response should have text content"
        assert len(response.text) > 50, "Response should be substantial"

        # Verify chunk events were emitted
        assert len(chunk_events) > 0, (
            "Should have received chunk events during streaming"
        )

        # Verify completion event was emitted
        assert len(complete_events) > 0, "Should have received completion event"
        assert len(complete_events) == 1, "Should have exactly one completion event"

        # Verify OpenAI stream events were emitted
        assert len(openai_stream_events) > 0, (
            "Should have received OpenAI stream events"
        )

        # Verify no error events were emitted
        assert len(error_events) == 0, (
            f"Should not have error events, but got: {error_events}"
        )

        # Verify chunk events have proper content and item_id
        total_delta_text = ""
        chunk_item_ids = set()
        content_indices = []
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

            # Validate content_index: should be sequential (0, 1, 2, ...) or None
            if chunk_event.content_index is not None:
                content_indices.append(chunk_event.content_index)

        # Verify content_index sequencing if any are provided
        if content_indices:
            # Should be sequential starting from 0
            expected_indices = list(range(len(content_indices)))
            assert content_indices == expected_indices, (
                f"content_index should be sequential (0, 1, 2, ...), but got: {content_indices}"
            )

        # Verify completion event has proper content and item_id
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

        # Verify that completion event item_id matches chunk event item_ids
        assert complete_event.item_id in chunk_item_ids, (
            f"Completion event item_id '{complete_event.item_id}' should match one of the chunk event item_ids: {chunk_item_ids}"
        )

        # Verify that chunk deltas reconstruct the final text (approximately)
        # Note: There might be slight differences due to formatting, so we check for substantial overlap
        assert len(total_delta_text) > 0, "Should have accumulated delta text"
        assert len(total_delta_text) >= len(response.text) * 0.8, (
            "Delta text should be substantial portion of final text"
        )

        # Verify event ordering: chunks should come before completion
        # This is implicit since we're using async/await, but we can verify the structure
        assert len(chunk_events) >= 1, (
            "Should have at least one chunk before completion"
        )

        # Verify OpenAI stream events contain expected event types
        stream_event_types = [event.event_type for event in openai_stream_events]
        assert "response.output_text.delta" in stream_event_types, (
            "Should have delta events"
        )
        assert "response.completed" in stream_event_types, (
            "Should have completion event"
        )
