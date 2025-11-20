import pytest
import os
from dotenv import load_dotenv
from vision_agents.plugins.xai.llm import XAILLM

load_dotenv()


class TestXAITools:
    """Test suite for XAILLM tool calling."""

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_tool_calling(self):
        llm = XAILLM(model="grok-4-latest", api_key=os.getenv("XAI_API_KEY"))

        @llm.register_function()
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        response = await llm.create_response(
            input="What is the weather in San Francisco?",
        )

        assert "sunny" in response.text.lower()
