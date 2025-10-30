import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import InMemoryConversation, Message

from vision_agents.plugins.vertex_ai import LLM as VertexAILLM
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.vertex_ai import events

load_dotenv()


class TestVertexAILLM:

    def test_message(self):
        messages = VertexAILLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        advanced = ["say hi"]
        messages2 = VertexAILLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.fixture
    async def llm(self) -> VertexAILLM:
        import os
        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        if not project:
            pytest.skip("GCP_PROJECT environment variable not set")
        
        llm = VertexAILLM(model="gemini-1.5-flash", project=project, location=location)
        llm._conversation = InMemoryConversation("be friendly", [])
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: VertexAILLM):
        response = await llm.simple_response("Explain quantum computing in 1 paragraph")
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: VertexAILLM):
        response = await llm.generate_content(message="say hi")

        # Assertions
        assert response.text
        assert response.original is not None

    @pytest.mark.integration
    async def test_stream(self, llm: VertexAILLM):
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
    async def test_memory(self, llm: VertexAILLM):
        await llm.simple_response(text="There are 2 dogs in the room")
        response = await llm.simple_response(text="How many paws are there in the room?")

        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: VertexAILLM):
        await llm.generate_content(message="There are 2 dogs in the room")
        response = await llm.generate_content(
            message="How many paws are there in the room?"
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_instruction_following(self):
        import os
        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_LOCATION", "us-central1")
        
        if not project:
            pytest.skip("GCP_PROJECT environment variable not set")
        
        llm = VertexAILLM(model="gemini-1.5-flash", project=project, location=location)
        llm._conversation = InMemoryConversation("be friendly", [])

        llm._set_instructions("only reply in 2 letter country shortcuts")

        response = await llm.simple_response(
            text="Which country is rainy, protected from water with dikes and below sea level?",
        )
        assert "nl" in response.text.lower()

    @pytest.mark.integration
    async def test_events(self, llm: VertexAILLM):
        """Test that LLM events are properly emitted during streaming responses."""
        # Track events and their content
        chunk_events = []
        complete_events = []
        vertex_ai_response_events = []
        error_events = []

        # Register event handlers BEFORE making the API call
        @llm.events.subscribe
        async def handle_chunk_event(event: LLMResponseChunkEvent):
            chunk_events.append(event)

        @llm.events.subscribe
        async def handle_complete_event(event: LLMResponseCompletedEvent):
            complete_events.append(event)

        @llm.events.subscribe
        async def handle_vertex_ai_response_event(event: events.VertexAIResponseEvent):
            vertex_ai_response_events.append(event)

        @llm.events.subscribe
        async def handle_error_event(event: events.VertexAIErrorEvent):
            error_events.append(event)

        # Make API call that should generate streaming events
        response = await llm.generate_content(
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

        # Verify Vertex AI response events were emitted
        assert len(vertex_ai_response_events) > 0, (
            "Should have received Vertex AI response events"
        )

        # Verify no error events were emitted
        assert len(error_events) == 0, (
            f"Should not have error events, but got: {error_events}"
        )
