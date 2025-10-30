from dotenv import load_dotenv
import os

import pytest

# Load environment variables
load_dotenv()


class TestAnthropic:
    """Integration tests for Anthropic plugin that make actual API calls."""

    async def test_not_integration(self):
        assert True

    @pytest.mark.integration
    async def test_chat_creation_with_system_message(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        import anthropic

        client = anthropic.Anthropic()

        message = client.messages.create(
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
