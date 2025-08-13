"""
Tests for the models package.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from models import OpenAIModel


class MockOpenAIResponse:
    """Mock OpenAI API response."""

    def __init__(self, content: str = "test response"):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content


class MockOpenAIStreamChunk:
    """Mock OpenAI streaming chunk."""

    def __init__(self, content: str = "chunk"):
        self.choices = [Mock()]
        self.choices[0].delta = Mock()
        self.choices[0].delta.content = content


class TestOpenAIModel:
    """Test cases for the OpenAIModel class."""

    def test_openai_model_initialization_with_name_only(self):
        """Test OpenAI model initialization with just a name."""
        model = OpenAIModel(name="gpt-4o")

        assert model.name == "gpt-4o"
        assert model.is_async is True
        assert model.client is not None

    def test_openai_model_initialization_with_client(self):
        """Test OpenAI model initialization with pre-configured client."""
        mock_client = Mock()
        model = OpenAIModel(name="gpt-3.5-turbo", client=mock_client)

        assert model.name == "gpt-3.5-turbo"
        assert model.client == mock_client
        assert model.is_async is False  # Mock is not AsyncOpenAI

    @patch("models.openai.AsyncOpenAI")
    def test_openai_model_initialization_with_api_key(self, mock_async_openai):
        """Test OpenAI model initialization with API key."""
        mock_client = Mock()
        mock_async_openai.return_value = mock_client

        model = OpenAIModel(
            name="gpt-4o",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            organization="test-org",
        )

        assert model.name == "gpt-4o"
        assert model.client == mock_client
        assert model.is_async is True

        mock_async_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            organization="test-org",
            project=None,
        )

    @pytest.mark.asyncio
    async def test_generate_chat_with_async_client(self):
        """Test chat generation with async client."""
        mock_client = AsyncMock()
        mock_response = MockOpenAIResponse("Hello, world!")
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        response = await model.generate_chat(messages)

        assert response == "Hello, world!"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o", messages=messages
        )

    @pytest.mark.asyncio
    async def test_generate_chat_with_parameters(self):
        """Test chat generation with additional parameters."""
        mock_client = AsyncMock()
        mock_response = MockOpenAIResponse("Parameterized response")
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Test"}]
        response = await model.generate_chat(
            messages, max_tokens=100, temperature=0.7, stop=["END"], top_p=0.9
        )

        assert response == "Parameterized response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stop=["END"],
            top_p=0.9,
        )

    @pytest.mark.asyncio
    async def test_generate_simple_prompt(self):
        """Test simple prompt generation (converts to chat format)."""
        mock_client = AsyncMock()
        mock_response = MockOpenAIResponse("Simple response")
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        response = await model.generate("What is AI?")

        assert response == "Simple response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o", messages=[{"role": "user", "content": "What is AI?"}]
        )

    @pytest.mark.asyncio
    async def test_generate_chat_empty_response(self):
        """Test handling of empty response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = []  # Empty choices
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        response = await model.generate_chat(messages)

        assert response == ""

    @pytest.mark.asyncio
    async def test_generate_chat_with_error(self):
        """Test error handling during generation."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception, match="API Error"):
            await model.generate_chat(messages)

    @pytest.mark.asyncio
    async def test_generate_chat_stream_with_async_client(self):
        """Test streaming chat generation with async client."""
        mock_client = AsyncMock()

        # Create an async generator for the stream
        async def mock_stream():
            yield MockOpenAIStreamChunk("Hello")
            yield MockOpenAIStreamChunk(" world")
            yield MockOpenAIStreamChunk("!")

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        chunks = []

        async for chunk in model.generate_chat_stream(messages):
            chunks.append(chunk)

        assert chunks == ["Hello", " world", "!"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o", messages=messages, stream=True
        )

    @pytest.mark.asyncio
    async def test_generate_stream_simple_prompt(self):
        """Test simple prompt streaming (converts to chat format)."""
        mock_client = AsyncMock()

        # Create an async generator for the stream
        async def mock_stream():
            yield MockOpenAIStreamChunk("Streaming")
            yield MockOpenAIStreamChunk(" response")

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        chunks = []
        async for chunk in model.generate_stream("What is AI?"):
            chunks.append(chunk)

        assert chunks == ["Streaming", " response"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is AI?"}],
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_generate_stream_with_parameters(self):
        """Test streaming generation with parameters."""
        mock_client = AsyncMock()

        async def mock_stream():
            yield MockOpenAIStreamChunk("Stream")

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Test"}]
        chunks = []

        async for chunk in model.generate_chat_stream(
            messages, max_tokens=50, temperature=0.5
        ):
            chunks.append(chunk)

        assert chunks == ["Stream"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=messages,
            stream=True,
            max_tokens=50,
            temperature=0.5,
        )

    @pytest.mark.asyncio
    async def test_generate_stream_with_empty_chunks(self):
        """Test streaming with empty content chunks."""
        mock_client = AsyncMock()

        async def mock_stream():
            # Chunk with no content
            empty_chunk = Mock()
            empty_chunk.choices = [Mock()]
            empty_chunk.choices[0].delta = Mock()
            empty_chunk.choices[0].delta.content = None
            yield empty_chunk

            # Chunk with content
            yield MockOpenAIStreamChunk("Content")

        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Test"}]
        chunks = []

        async for chunk in model.generate_chat_stream(messages):
            chunks.append(chunk)

        # Should only get chunks with content
        assert chunks == ["Content"]

    @pytest.mark.asyncio
    async def test_generate_stream_with_error(self):
        """Test error handling during streaming."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Streaming API Error")
        )

        model = OpenAIModel(name="gpt-4o", client=mock_client)

        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception, match="Streaming API Error"):
            async for chunk in model.generate_chat_stream(messages):
                pass  # This should raise before yielding anything

    def test_model_repr(self):
        """Test string representation of the model."""
        mock_client = Mock()
        model = OpenAIModel(name="gpt-4o", client=mock_client)

        repr_str = repr(model)
        assert "OpenAIModel" in repr_str
        assert "gpt-4o" in repr_str
        assert "async=False" in repr_str  # Mock client is not async

    def test_model_properties(self):
        """Test model properties."""
        mock_client = Mock()
        model = OpenAIModel(name="gpt-3.5-turbo", client=mock_client)

        assert model.name == "gpt-3.5-turbo"
        assert model.client == mock_client
        assert model.is_async is False


if __name__ == "__main__":
    pytest.main([__file__])
