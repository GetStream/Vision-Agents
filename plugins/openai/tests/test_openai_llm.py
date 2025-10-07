import pytest
from dotenv import load_dotenv

from vision_agents.core.agents.conversation import Message
from vision_agents.plugins.openai.openai_llm import OpenAILLM
from vision_agents.core.llm.events import LLMResponseChunkEvent

load_dotenv()


class TestOpenAILLM:
    """Test suite for OpenAILLM class with mocked API calls."""

    def test_message(self):
        messages = OpenAILLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

        advanced = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what do you see in this image?"},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ]
        messages2 = OpenAILLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.fixture
    async def llm(self) -> OpenAILLM:
        llm = OpenAILLM(model="gpt-4o")
        return llm

    @pytest.mark.integration
    async def test_simple(self, llm: OpenAILLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )

        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: OpenAILLM):


        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )

        # Assertions
        assert response.text
        assert hasattr(response.original, 'id')  # OpenAI response has id


    @pytest.mark.integration
    async def test_streaming(self, llm: OpenAILLM):

        streamingWorks = False
        
        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True
        
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        
        await llm.events.wait()

        assert response.text
        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: OpenAILLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: OpenAILLM):
        await llm.create_response(
            input="There are 2 dogs in the room",
        )
        response = await llm.create_response(
            input="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text
