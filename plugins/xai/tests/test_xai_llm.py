import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.xai.llm import XAILLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture
async def llm():
    llm = XAILLM(model="grok-4-latest", api_key=os.getenv("XAI_API_KEY"))
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
class TestXAILLMIntegration:
    """Test suite for XAILLM class with live API calls."""

    async def test_simple_response(self, llm: XAILLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Explain quantum computing in 1 paragraph")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: XAILLM):
        await collect_simple_response(
            llm.simple_response(text="There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response(text="How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_tool_calling(self, llm: XAILLM):
        @llm.register_function()
        async def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        _, final = await collect_simple_response(
            llm.simple_response("What is the weather in San Francisco?")
        )

        assert "sunny" in final.text.lower()
