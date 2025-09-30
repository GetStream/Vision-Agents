import asyncio
import pytest
from dotenv import load_dotenv


from stream_agents.core.agents.conversation import InMemoryConversation

from stream_agents.core.agents.conversation import Message
from stream_agents.core.llm.events import StandardizedTextDeltaEvent
from stream_agents.plugins.anthropic.anthropic_llm import ClaudeLLM

load_dotenv()


class TestClaudeLLM:
    """Test suite for ClaudeLLM class with real API calls."""

    @pytest.fixture
    async def llm(self) -> ClaudeLLM:
        """Test ClaudeLLM initialization with a provided client."""
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022")
        llm._conversation = InMemoryConversation("be friendly", [])
        return llm

    @pytest.mark.asyncio
    async def test_message(self, llm: ClaudeLLM):
        messages = ClaudeLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    @pytest.mark.asyncio
    async def test_advanced_message(self, llm: ClaudeLLM):
        advanced = {
            "role": "user",
            "content": "Explain quantum entanglement in simple terms.",
        }
        messages2 = ClaudeLLM._normalize_message(advanced)
        assert messages2[0].original is not None

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

    @pytest.mark.integration
    async def test_stream(self, llm: ClaudeLLM):
        streamingWorks = False
        
        @llm.events.subscribe
        async def passed(event: StandardizedTextDeltaEvent):
            nonlocal streamingWorks
            streamingWorks = True
        
        # Allow event subscription to be processed
        await asyncio.sleep(0.01)
        
        response = await llm.simple_response("Explain magma to a 5 year old")
        print(response)

        assert streamingWorks


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
            messages=[
                {"role": "user", "content": "How many paws are there in the room?"}
            ],
            max_tokens=1000,
        )
        assert "8" in response.text or "eight" in response.text
