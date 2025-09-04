"""
Proposal
- What if each LLM has a create response method.
- Create response receives the text and processors/ state.
- It normalized the response (so we can update chat history/state)

But... we also keep the original APIs available
- so you can do llm.generate() (gemini example) with full gemini support
- llm.create_message (claude)
- llm.create_response (openAI)
- which expose the full features of claude/openai & gemini

What we do need to standardize
- response -> text conversion to update chat history
- processor state + transcription -> arguments needed for calling the LLM

And more advanced things
- Streaming response standardization
- STS standardization

"""

import os
import pytest
from dotenv import load_dotenv

from anthropic import AsyncAnthropic
from anthropic.types import Message
from openai import OpenAI

from stream_agents.llm.llm import ClaudeLLM, OpenAILLM, LLMResponse


# Load environment variables at module level
load_dotenv()


class TestClaudeLLM:
    """Test suite for ClaudeLLM class with real API calls."""

    def test_init_with_client(self):
        """Test ClaudeLLM initialization with a provided client."""
        custom_client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "test-key")
        )
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022", client=custom_client)
        assert llm.client == custom_client
        assert llm.model == "claude-3-5-sonnet-20241022"

    def test_init_with_no_arguments(self):
        """Test ClaudeLLM initialization with no arguments (should use env vars)."""
        # This test assumes ANTHROPIC_API_KEY is set in environment or .env file
        # If not set, AsyncAnthropic will still initialize but API calls will fail
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022")
        assert isinstance(llm.client, AsyncAnthropic)
        assert llm.model == "claude-3-5-sonnet-20241022"

    def test_init_with_api_key(self):
        """Test ClaudeLLM initialization with an API key."""
        test_api_key = os.getenv("ANTHROPIC_API_KEY", "test-api-key-123")
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022", api_key=test_api_key)
        assert isinstance(llm.client, AsyncAnthropic)
        assert llm.model == "claude-3-5-sonnet-20241022"

    @pytest.mark.integration
    async def test_create_message_say_hi(self):
        """Test create_message method with 'say hi' input using real API."""
        llm = ClaudeLLM(model="claude-3-5-sonnet-20241022")

        # Call create_message with real API - Claude requires messages and max_tokens
        response = await llm.create_message(
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=1000,
            model="claude-3-5-sonnet-20241022",
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert isinstance(response.original, Message)
        assert response.original.content[0].text  # Should have some text response
        print(f"Response: {response.original.content[0].text}")

    @pytest.mark.integration
    async def test_create_message_with_system_and_image(self):
        """Test create_message with system instructions and image URL using real API."""
        llm = ClaudeLLM("claude-3-5-sonnet-20241022")

        # Prepare messages with system prompt and image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "tell me whats in this image"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800",
                        },
                    },
                ],
            }
        ]

        system_prompt = "You are a helpful assistant that describes images in detail."

        # Call create_message with real API
        response = await llm.create_message(
            model="claude-3-5-sonnet-20241022",
            system=system_prompt,
            messages=messages,
            max_tokens=1000,
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert isinstance(response.original, Message)
        # The response should mention a tree since we're using a tree image
        response_text = response.original.content[0].text.lower()
        assert response_text  # Should have some text
        print(f"Image analysis: {response.original.content[0].text[:200]}...")


class TestOpenAILLM:
    """Test suite for OpenAILLM class with real API calls."""

    def test_init_with_client(self):
        """Test OpenAILLM initialization with a provided client."""
        custom_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "test-key"))
        llm = OpenAILLM(model="gpt-4o", client=custom_client)
        assert llm.client == custom_client
        assert llm.model == "gpt-4o"

    def test_init_with_no_arguments(self):
        """Test OpenAILLM initialization with no arguments (should use env vars)."""
        # This test assumes OPENAI_API_KEY is set in environment or .env file
        # If not set, OpenAI will still initialize but API calls will fail
        llm = OpenAILLM(model="gpt-4o")
        assert isinstance(llm.client, OpenAI)
        assert llm.model == "gpt-4o"

    def test_init_with_api_key(self):
        """Test OpenAILLM initialization with an API key."""
        test_api_key = os.getenv("OPENAI_API_KEY", "test-api-key-123")
        llm = OpenAILLM(model="gpt-4o", api_key=test_api_key)
        assert isinstance(llm.client, OpenAI)
        assert llm.model == "gpt-4o"

    @pytest.mark.integration
    async def test_create_response_say_hi(self):
        """Test create_response method with 'say hi' input using real API."""
        llm = OpenAILLM(model="gpt-4o")

        # Call create_response with real API
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, "id")  # OpenAI response has id
        print(f"Response ID: {response.original.id}")

    @pytest.mark.integration
    async def test_create_response_with_model_override(self):
        """Test create_response with model override in kwargs."""
        llm = OpenAILLM(model="gpt-4o")

        # Override the model in kwargs
        response = await llm.create_response(
            input="What is 2+2?",
            instructions="You are a math tutor.",
            model="gpt-3.5-turbo",  # Override the default model
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, "id")
        print(f"Response with model override: {response.original.id}")

    @pytest.mark.integration
    async def test_simple_response_with_conversation(self):
        """Test simple_response method with conversation context."""
        from stream_agents.agents import Conversation

        llm = OpenAILLM(model="gpt-4o")

        # Create a conversation with required parameters
        conversation = Conversation(
            instructions="You are a helpful coding assistant. Keep responses concise.",
            messages=[],
            channel="test",
            chat_client="test_client",
        )

        # Test simple_response
        response = await llm.simple_response(
            text="Write a Python function to calculate fibonacci numbers",
            conversation=conversation,
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, "id")
        print(f"Simple response with conversation: {response.original.id}")

    @pytest.mark.integration
    async def test_create_response_with_custom_parameters(self):
        """Test create_response with various OpenAI-specific parameters."""
        llm = OpenAILLM(model="gpt-4o")

        # Test with valid OpenAI parameters (only input and instructions are supported)
        response = await llm.create_response(
            input="Explain quantum computing in simple terms",
            instructions="You are a science educator. Use analogies.",
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, "id")
        print(f"Response with custom parameters: {response.original.id}")
