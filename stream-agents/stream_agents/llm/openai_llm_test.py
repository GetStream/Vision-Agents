import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from stream_agents.llm.llm import LLMResponse
from stream_agents.llm.openai_llm import OpenAILLM


load_dotenv()

class TestOpenAILLM:
    """Test suite for OpenAILLM class with real API calls."""

    @pytest.fixture
    def llm(self):
        """Test OpenAILLM initialization with a provided client."""
        llm = OpenAILLM(model="gpt-4o")
        return llm

    @pytest.mark.integration
    async def test_create_response_say_hi(self, llm):

        llm.before_response_listener = lambda x: print(x)
        llm.after_response_listener = lambda x: print(x)

        # Call create_response with real API
        response = await llm.create_response(
            input="say hi",
            instructions="You are a helpful assistant."
        )

        # Assertions
        assert response.text
        assert hasattr(response.original, 'id')  # OpenAI response has id
        print(f"Response ID: {response.original.id}")

    @pytest.mark.integration
    async def test_create_response_with_model_override(self):
        """Test create_response with model override in kwargs."""
        llm = OpenAILLM(model="gpt-4o")

        # Override the model in kwargs
        response = await llm.create_response(
            input="What is 2+2?",
            instructions="You are a math tutor.",
            model="gpt-3.5-turbo"  # Override the default model
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, 'id')
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
            chat_client="test_client"
        )

        # Test simple_response
        response = await llm.simple_response(
            text="Write a Python function to calculate fibonacci numbers",
            conversation=conversation
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, 'id')
        print(f"Simple response with conversation: {response.original.id}")


    @pytest.mark.integration
    async def test_create_response_with_custom_parameters(self):
        """Test create_response with various OpenAI-specific parameters."""
        llm = OpenAILLM(model="gpt-4o")

        # Test with valid OpenAI parameters (only input and instructions are supported)
        response = await llm.create_response(
            input="Explain quantum computing in simple terms",
            instructions="You are a science educator. Use analogies."
        )

        # Assertions
        assert isinstance(response, LLMResponse)
        assert hasattr(response.original, 'id')
        print(f"Response with custom parameters: {response.original.id}")