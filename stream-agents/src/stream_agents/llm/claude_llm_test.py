
import os

import pytest
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

from stream_agents.llm.llm import LLMResponse
from stream_agents.llm.claude_llm import ClaudeLLM


load_dotenv()


class TestClaudeLLM:
    """Test suite for ClaudeLLM class with real API calls."""

    @pytest.fixture
    def llm(self) -> ClaudeLLM:
        """Test ClaudeLLM initialization with a provided client."""
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022")
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: ClaudeLLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: ClaudeLLM):
        response = await llm.create_message(
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=1000,
        )

        # Assertions
        assert response.text
        assert hasattr(response.original, 'id')  # Claude response has id

    @pytest.mark.integration
    async def test_memory(self, llm: ClaudeLLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: ClaudeLLM):
        await llm.create_message(
            messages=[{"role": "user", "content": "There are 2 dogs in the room"}],
            max_tokens=1000,
        )
        response = await llm.create_message(
            messages=[{"role": "user", "content": "How many paws are there in the room?"}],
            max_tokens=1000,
        )
        assert "8" in response.text or "eight" in response.text

