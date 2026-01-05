from dotenv import load_dotenv
import sys
import os

import pytest

# Add the plugin directory to the path so we can import the plugin
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()


class TestTavus:
    """Integration tests for XAI plugin that make actual API calls."""

    async def test_not_integration(self):
        assert True

    @pytest.mark.integration
    async def test_chat_creation_with_system_message(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        import anthropic

        client = anthropic.AsyncAnthropic()

        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": "What should I search for to find the latest developments in renewable energy?",
                }
            ],
        )
        print(message.content)
