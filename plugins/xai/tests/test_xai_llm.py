import pytest
from dotenv import load_dotenv
import os

from vision_agents.core.agents.conversation import Message
from vision_agents.plugins.xai.llm import XAILLM
from vision_agents.core.llm.events import LLMResponseChunkEvent

load_dotenv()


class TestXAILLM:
    """Test suite for XAILLM class with live API calls."""

    def test_message(self):
        messages = XAILLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    async def test_advanced_message(self):
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
        messages2 = XAILLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_simple(self):
        llm = XAILLM(model="grok-4", api_key=os.getenv("XAI_API_KEY"))
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        assert response.text

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_native_api(self):
        llm = XAILLM(model="grok-4", api_key=os.getenv("XAI_API_KEY"))
        response = await llm.create_response(
            input="say hi", instructions="You are a helpful assistant."
        )
        assert response.text

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_streaming(self):
        llm = XAILLM(model="grok-4", api_key=os.getenv("XAI_API_KEY"))
        streaming_works = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        await llm.events.wait()

        assert response.text
        assert streaming_works

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_memory(self):
        llm = XAILLM(model="grok-4", api_key=os.getenv("XAI_API_KEY"))
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        await llm.events.wait()
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
    async def test_native_memory(self):
        llm = XAILLM(model="grok-4", api_key=os.getenv("XAI_API_KEY"))
        await llm.create_response(
            input="There are 2 dogs in the room",
        )
        await llm.events.wait()
        response = await llm.create_response(
            input="How many paws are there in the room?",
        )
        assert "8" in response.text or "eight" in response.text
