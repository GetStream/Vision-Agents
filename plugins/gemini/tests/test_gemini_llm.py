import pytest
from dotenv import load_dotenv


from vision_agents.core.agents.conversation import InMemoryConversation, Message

from vision_agents.plugins.gemini.gemini_llm import GeminiLLM
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.gemini import events

load_dotenv()


class TestGeminiLLM:
    def test_message(self):
        messages = GeminiLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        advanced = ["say hi"]
        messages2 = GeminiLLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.fixture
    async def llm(self) -> GeminiLLM:
        llm = GeminiLLM(model="gemini-2.0-flash-exp")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: GeminiLLM):
        response = await llm.simple_response("Explain quantum computing in 1 paragraph")
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: GeminiLLM):
        response = await llm.send_message(message="say hi")

        # Assertions
        assert response.text
        assert hasattr(response.original, "text")  # Gemini response has text attribute

    @pytest.mark.integration
    async def test_stream(self, llm: GeminiLLM):
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        await llm.simple_response("Explain magma to a 5 year old")

        # Wait for all events in queue to be processed
        await llm.events.wait()

        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: GeminiLLM):
        await llm.simple_response(text="There are 2 dogs in the room")
        response = await llm.simple_response(
            text="How many paws are there in the room?"
        )

        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: GeminiLLM):
        await llm.send_message(message="There are 2 dogs in the room")
        response = await llm.send_message(
            message="How many paws are there in the room?"
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_instruction_following(self):
        llm = GeminiLLM(model="gemini-2.0-flash-exp")
        llm.set_conversation(InMemoryConversation("be friendly", []))

        llm.set_instructions("only reply in 2 letter country shortcuts")

        response = await llm.simple_response(
            text="Which country is rainy, protected from water with dikes and below sea level?",
        )
        assert "nl" in response.text.lower()

    @pytest.mark.integration
    async def test_events(self, llm: GeminiLLM):
        """Test that LLM events are properly emitted during streaming responses."""
        # Track events and their content
        chunk_events = []
        complete_events = []
        gemini_response_events = []
        error_events = []

        # Register event handlers BEFORE making the API call
        @llm.events.subscribe
        async def handle_chunk_event(event: LLMResponseChunkEvent):
            chunk_events.append(event)

        @llm.events.subscribe
        async def handle_complete_event(event: LLMResponseCompletedEvent):
            complete_events.append(event)

        @llm.events.subscribe
        async def handle_gemini_response_event(event: events.GeminiResponseEvent):
            gemini_response_events.append(event)

        @llm.events.subscribe
        async def handle_error_event(event: events.GeminiErrorEvent):
            error_events.append(event)

        # Make API call that should generate streaming events
        response = await llm.send_message(
            message="Create a small story about the weather in the Netherlands. Make it at least 2 paragraphs long."
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

        # Verify Gemini response events were emitted
        assert len(gemini_response_events) > 0, (
            "Should have received Gemini response events"
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

        # Verify Gemini response events contain expected content
        for gemini_event in gemini_response_events:
            assert gemini_event.response_chunk is not None, (
                "Gemini response events should have response_chunk"
            )
