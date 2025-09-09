import os

import pytest
from dotenv import load_dotenv
from google import genai

from stream_agents.core.llm.llm import LLMResponse
from stream_agents.core.llm.gemini_llm import GeminiLLM

from stream_agents.core.agents.conversation import InMemoryConversation, Message

from stream_agents.core.llm.types import StandardizedTextDeltaEvent

load_dotenv()


class TestGeminiLLM:
    """Test suite for GeminiLLM class with real API calls."""

    @pytest.fixture
    def llm(self) -> GeminiLLM:
        """Test GeminiLLM initialization with a provided client."""
        llm = GeminiLLM(model="gemini-1.5-flash")
        llm._conversation = InMemoryConversation("be friendly", [])
        return llm

    def test_message(self, llm: GeminiLLM):
        messages = GeminiLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content is "say hi"

    def test_advanced_message(self, llm: GeminiLLM):
        advanced = ["say hi"]
        messages2 = GeminiLLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.mark.integration
    async def test_simple(self, llm: GeminiLLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: GeminiLLM):
        response = await llm.send_message(
            message="say hi"
        )

        # Assertions
        assert response.text
        assert hasattr(response.original, 'text')  # Gemini response has text attribute

    @pytest.mark.integration
    async def test_stream(self, llm: GeminiLLM):
        streamingWorks = False
        @llm.on('standardized.output_text.delta')
        def passed(event: StandardizedTextDeltaEvent):
            nonlocal streamingWorks
            streamingWorks = True

        response = await llm.simple_response("Explain magma to a 5 year old")

        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: GeminiLLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: GeminiLLM):
        await llm.send_message(
            message="There are 2 dogs in the room"
        )
        response = await llm.send_message(
            message="How many paws are there in the room?"
        )
        assert "8" in response.text or "eight" in response.text
