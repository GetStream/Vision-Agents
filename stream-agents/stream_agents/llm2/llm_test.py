'''
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

'''
import os
import pytest
from dotenv import load_dotenv

from anthropic import AsyncAnthropic
from anthropic.types import Message, TextBlock

from stream_agents.llm2.llm import ClaudeLLM, LLMResponse


# Load environment variables at module level
load_dotenv()


class TestClaudeLLM:
    """Test suite for ClaudeLLM class with real API calls."""
    
    def test_init_with_client(self):
        """Test ClaudeLLM initialization with a provided client."""
        custom_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "test-key"))
        llm = ClaudeLLM(client=custom_client)
        assert llm.client == custom_client
    
    def test_init_with_no_arguments(self):
        """Test ClaudeLLM initialization with no arguments (should use env vars)."""
        # This test assumes ANTHROPIC_API_KEY is set in environment or .env file
        # If not set, AsyncAnthropic will still initialize but API calls will fail
        llm = ClaudeLLM()
        assert isinstance(llm.client, AsyncAnthropic)
    
    def test_init_with_api_key(self):
        """Test ClaudeLLM initialization with an API key."""
        test_api_key = os.getenv("ANTHROPIC_API_KEY", "test-api-key-123")
        llm = ClaudeLLM(api_key=test_api_key)
        assert isinstance(llm.client, AsyncAnthropic)
    
    @pytest.mark.integration
    async def test_create_response_say_hi(self):
        """Test create_response method with 'say hi' input using real API."""
        llm = ClaudeLLM()
        
        # Call create_response with real API
        response = await llm.create_response("say hi")
        
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
                    {
                        "type": "text",
                        "text": "tell me whats in this image"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=800"
                        }
                    }
                ]
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
