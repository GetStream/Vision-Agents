import pytest
from dotenv import load_dotenv
import sys
import os

# Add the plugin directory to the path so we can import the plugin
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()


class TestXAI:
    """Integration tests for XAI plugin that make actual API calls."""

    @pytest.fixture(scope="class")
    def check_api_key(self):
        """Check if XAI API key is available."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            pytest.skip("XAI_API_KEY environment variable not set")
        return api_key

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_xai_client_initialization(self, check_api_key):
        """Test that XAI client can be initialized."""
        from xai_sdk import AsyncClient

        client = AsyncClient()
        assert client is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_creation_with_system_message(self, check_api_key):
        """Test creating a chat with system message."""
        from xai_sdk import AsyncClient
        from xai_sdk.chat import system

        client = AsyncClient()
        chat = client.chat.create(
            model="grok-beta",
            messages=[system("You are a helpful assistant. Keep responses brief.")],
        )

        assert chat is not None
        assert hasattr(chat, "append")
        assert hasattr(chat, "sample")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_response_generation(self, check_api_key):
        """Test generating a response from the chat."""
        from xai_sdk import AsyncClient
        from xai_sdk.chat import system, user

        client = AsyncClient()
        chat = client.chat.create(
            model="grok-beta",
            messages=[
                system("You are a helpful assistant. Keep responses to one sentence.")
            ],
        )

        # Add a user message and get response
        chat.append(user("Say hello in exactly 3 words."))
        response = await chat.sample()

        assert response is not None
        assert hasattr(response, "content")
        assert isinstance(response.content, str)
        assert len(response.content.strip()) > 0
        print(f"Response: {response.content}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_conversation_flow(self, check_api_key):
        """Test a complete conversation flow."""
        from xai_sdk import AsyncClient
        from xai_sdk.chat import system, user

        client = AsyncClient()
        chat = client.chat.create(
            model="grok-beta",
            messages=[
                system(
                    "You are a helpful assistant. Keep all responses to one sentence."
                )
            ],
        )

        # First exchange
        chat.append(user("What is 2+2?"))
        response1 = await chat.sample()
        chat.append(response1)

        # Second exchange
        chat.append(user("What about 3+3?"))
        response2 = await chat.sample()

        # Verify both responses
        assert response1 is not None
        assert response2 is not None
        assert isinstance(response1.content, str)
        assert isinstance(response2.content, str)
        assert len(response1.content.strip()) > 0
        assert len(response2.content.strip()) > 0

        print(f"First response: {response1.content}")
        print(f"Second response: {response2.content}")

    def test_version_import(self):
        """Test that version can be imported."""
        from version import __version__

        assert __version__ == "0.1.0"
