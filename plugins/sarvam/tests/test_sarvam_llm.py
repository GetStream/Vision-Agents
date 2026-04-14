"""Tests for the Sarvam LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.sarvam import LLM

load_dotenv()


class TestSarvamLLM:
    """Unit tests for Sarvam LLM configuration."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            LLM()

    async def test_default_model(self):
        llm = LLM(api_key="sk_test")
        assert llm.model == "sarvam-m"

    async def test_custom_model(self):
        llm = LLM(api_key="sk_test", model="sarvam-30b")
        assert llm.model == "sarvam-30b"

    async def test_base_url_points_to_sarvam(self):
        llm = LLM(api_key="sk_test")
        assert str(llm._client.base_url).startswith("https://api.sarvam.ai")

    async def test_subscription_key_header_injected(self):
        llm = LLM(api_key="sk_test")
        headers = llm._client.default_headers
        assert headers.get("api-subscription-key") == "sk_test"


@pytest.mark.skipif(not os.getenv("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set")
@pytest.mark.integration
class TestSarvamLLMIntegration:
    """Integration tests hitting the real Sarvam Chat Completions endpoint."""

    @pytest.fixture
    async def llm(self):
        llm = LLM(model="sarvam-m")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    async def test_simple_response(self, llm):
        response = await llm.simple_response("Greet the user in English")
        assert response.text
        assert response.exception is None

    async def test_streaming_chunks(self, llm):
        saw_chunk = False

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            nonlocal saw_chunk
            saw_chunk = True

        await llm.simple_response("Say hi in one word")
        await llm.events.wait()
        assert saw_chunk
